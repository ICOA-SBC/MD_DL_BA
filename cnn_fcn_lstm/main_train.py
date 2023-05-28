### IMPORT LIBRARIES
##### BUILT_IN
import time
import os
import copy
import mlflow
import seaborn as sns
import pandas as pd
import numpy as np
##### EXTERNAL
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
##### LOCAL
from codes.cnn_fcn_lstm import CNN_FCN_LSTM
from codes.complex_dataset import Complexes_4DDataset
from codes.tools import Distribution as DIST
from codes.tools import convert_byte, convert_time, master_print
from codes.transformations import build_rotations
from main_test import analyse, predict


### SETUP PROCESS COMMUNICATION
def setup():
    DIST()
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=DIST.size,
                            rank=DIST.rank)
    torch.cuda.set_device(DIST.local_rank)

def cleanup():
    dist.destroy_process_group()


### UTILS
def load_model(cfg, dev):
    if cfg.network.pretrained_path == 'True':
        filename = os.path.join(cfg.experiment.model_path, f"{cfg.experiment.run}.pth")
    else:
        filename = f"{cfg.network.pretrained_path.strip('.pth')}.pth"
    print(f"\tloading model {filename}")
    checkpoint = torch.load(filename, map_location=dev)
    model = CNN_FCN_LSTM(in_frames=cfg.data_setup.frames,
                         in_channels_per_frame=cfg.data_setup.features,
                         model_architecture=cfg.model.architecture)

    model.load_state_dict(checkpoint['model'])
    print(f"Model successfully loaded")
    model.to(dev)
    return model

def save_model(model, epoch, model_name, cfg, optimizer=None, lr_scheduler=None):
    filename = os.path.join(cfg.experiment.model_path, f"{model_name}.pth")
    print(f"\tsaving model {filename} on rank {DIST.rank}")
    model_without_ddp = model.module

    if optimizer is None:
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': None,
            'lr_scheduler': None,
            'epoch': epoch,
            'cfg': cfg}
    else:
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'cfg': cfg}

    torch.save(checkpoint, filename)


### TRAIN
def train(model, dl, ds_size, cfg, device):

    ### TRAINING SETUP
    cfg_train, cfg_exp = cfg.training, cfg.experiment
    best_mse, best_epoch = 100, -1
    patience, max_patience = 0, cfg_train.patience
    metric = torch.nn.MSELoss(reduction='sum')
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=cfg_train.learning_rate, weight_decay=cfg_train.weight_decay)

    # https://github.com/Lance0218/Pytorch-DistributedDataParallel-Training-Tricks
    master_print(
        f"WARM UP during {cfg_train.warmup_epochs} over max {cfg_train.num_epochs} with patience at {max_patience}"
        f"\nlearning_rate will reach {cfg_train.learning_rate} after {cfg_train.warmup_epochs} epochs")
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
    # A function which computes a multiplicative factor given an integer parameter epoch
    warm_up = lambda _epoch: _epoch / cfg_train.warmup_epochs if _epoch <= cfg_train.warmup_epochs else 1  # (L6)

    scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up, verbose=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, cooldown=2,
                                                     verbose=True)
    ### TRAINING PHASE
    time_train = time.time()

    for epoch in range(cfg_train.num_epochs):
        master_print(f"Epoch {epoch + 1}/{cfg_train.num_epochs}")
        time_epoch = time.time()

        for phase in ['train', 'val']:
            master_print(f"Phase: {phase} ")
            model.train() if phase == 'train' else model.eval()

            running_metrics = 0.0

            for (inputs, affinities) in dl[phase]:
                inputs, affinities = inputs.to(device, non_blocking=True), affinities.to(device, non_blocking=True)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), affinities.float())

                    # statistics
                    running_metrics += metric(outputs, affinities)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # synchronization
            # https://discuss.pytorch.org/t/reducelronplateau-usage-with-distributeddataparallel/117509
            dist.barrier()
            dist.all_reduce(running_metrics, op=dist.ReduceOp.SUM)
            epoch_metric = running_metrics.cpu().item() / ds_size[phase]

            if phase == 'val':
                if epoch <= cfg_train.warmup_epochs:
                    master_print("scheduler LambdaLR warmup in action")
                    scheduler_wu.step()
                # Learning rate scheduling should be applied after optimizer’s update (pytorch doc)
                master_print("scheduler ReduceLROnPlateau in action")
                scheduler.step(epoch_metric)

            if phase == 'train':
                master_print(f"\t[{phase}] MSELoss {epoch_metric:.4f}")
            else:
                master_print(f"\t[{phase}] MSELoss {epoch_metric:.4f} \t Duration: {time.time() - time_epoch:.2f}")
                master_print(
                    f"Current learning rate on GPU {DIST.rank}: wu \t{scheduler_wu.get_last_lr()} (from scheduler_wu)")

            # check if the model is improving
            if phase == 'val':
                if epoch_metric < best_mse:
                    master_print(f"*** /\\ Better loss {best_mse} --> {epoch_metric}")
                    best_mse, best_epoch = epoch_metric, epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if DIST.master_rank:
                        model_name = f"{cfg_exp.run}_{best_mse:.4f}_{best_epoch}"
                        save_model(model, best_epoch, model_name, cfg, optimizer, scheduler)
                    patience = 0
                else:
                    patience += 1

            mlflow.log_metric(f"{phase}_mse", epoch_metric, epoch)

        if patience > max_patience:
            master_print(f"----------- Early stopping activated ! ({patience} > {max_patience}")
            break
    duration = time.time() - time_train
    master_print("\n\n_____________________________________________")
    master_print(f"[{epoch + 1} / {cfg_train.num_epochs}] Best mean MSE: {best_mse:.4f} at epoch {best_epoch} \
       \n\tTotal duration: {convert_time(duration)}")
    master_print("_____________________________________________")

    model.load_state_dict(best_model_wts)

    return model, best_epoch, best_mse


def test(model, test_dataloader, test_samples, device):
    metric = torch.nn.MSELoss(reduction='sum')

    model.eval()
    running_metrics = 0.0

    for (inputs, labels) in test_dataloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            running_metrics += metric(outputs, labels).cpu().item()

    metric = running_metrics / test_samples
    master_print(f"\t[Test] MSELoss {metric:.4f}")
    mlflow.log_metric(f"test_mse", metric)


def mlflow_setup(cfg):
    if DIST.master_rank:
        mlflow.log_param("learning_rate", cfg.training.learning_rate)
        mlflow.log_param("patience", cfg.training.patience)
        mlflow.log_param("max_epoch", cfg.training.num_epochs)
        # log configuration as an artefact
        with open('./configuration.yaml', 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))
        mlflow.log_artifact("configuration.yaml")


@hydra.main(config_path="./configs", config_name=f"cnnlstm_v6")
def main(cfg: DictConfig) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(DIST.local_rank)
    output_pdf = f"{os.environ['SCRATCH']}/_pafnucy/cnn_fcn_lstm/"

    master_print(f"Training with {DIST.size} gpu !")

    if cfg.debug:
        print(OmegaConf.to_yaml(cfg))

    # either run training with one random simulation from each complex per epoch or all simulations per epoch
    by_complex = cfg.experiment.by_complex
    # batch_size per gpu
    batch_size = cfg.training.batch_size

    # create transformations
    rotations_matrices = build_rotations()
    master_print(f"Number of available rotations: {len(rotations_matrices)}")

    # Dataset creation
    train_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex,
                                   rotations_matrices, mode="train", debug=cfg.debug)
    val_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex, mode="val", debug=cfg.debug)
    ds_size = {'train': len(train_ds), 'val': len(val_ds)}

    train_sampler = DistributedSampler(train_ds, num_replicas=DIST.size, rank=DIST.rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=DIST.size, rank=DIST.rank, shuffle=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          num_workers=4, pin_memory=True, sampler=train_sampler)
    val_dl = DataLoader(val_ds, batch_size=batch_size,
                        num_workers=4, pin_memory=True, sampler=val_sampler)
    dl = {'train': train_dl, 'val': val_dl}

    if cfg.network.pretrained_path: #TRUE or Path
        model = load_model(cfg, dev)
    else:
        model = CNN_FCN_LSTM(in_frames=cfg.data_setup.frames,
                         in_channels_per_frame=cfg.data_setup.features,
                         model_architecture=cfg.model.architecture)
    model.to(DIST.local_rank)
    # Convert BatchNorm to SyncBatchNorm.
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    ddp_model = DDP(model, device_ids=[DIST.local_rank])

    try:
        mlflow.end_run()
    except:
        master_print("mlflow not running")

    if DIST.master_rank:
        mlflow.set_tracking_uri(cfg.experiment.path)
        mlflow.set_experiment(cfg.experiment.name)

    with mlflow.start_run(run_name=cfg.experiment.run) as run:
        mlflow_setup(cfg)

        # train
        best_model, best_epoch, best_MSE = train(ddp_model, dl, ds_size, cfg, DIST.local_rank)

        if DIST.master_rank:
            print("Saving best model")
            save_model(best_model, best_epoch, cfg.experiment.run, cfg)

        # test on one GPU (for simplicity)
        if DIST.master_rank:
            if cfg.network.sim_test:
                test_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex=False, mode="test", debug=cfg.debug) #test on all simulation from test set
            else:
                test_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex=True, mode="test", debug=cfg.debug)
            test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True,
                                 persistent_workers=True)
            print(f"--------------------- Running test")
            test(best_model.module, test_dl, len(test_ds), dev)
            print(f"--------------------- Running predict")
            # apply model on test
            affinities, predictions = predict(best_model.module, test_dl, len(test_ds), dev)
            # compute metrics
            rmse, mae, corr = analyse(affinities, predictions)

            if cfg.network.mean_test:
                pdbs = [pdb.split('/')[11] for pdb in test_ds.samples_list]
                assert len(affinities) == len(pdbs)
                DFresults = pd.DataFrame([pdbs,affinities,predictions]).T
                DFresults.columns = ['pdbid','real','predictions']
                DFgroupby = DFresults.groupby('pdbid')
                DFresults = pd.concat([DFgroupby['real'].first(), DFgroupby['predictions'].apply(list), DFgroupby['predictions'].mean().round(2).rename('mean_pred')], axis=1)
                DFresults.to_csv(f'{work}/deep_learning/MD_ConvLSTM/cnn_fcn_lstm/correlation_plot/CNN-LSTM_mean_{cfg.experiment.run}.csv')
                affinities = DFresults.real
                predictions = DFresults.mean_pred

            mlflow.log_param("best_epoch", best_epoch)
            mlflow.log_metric("best_val_MSELoss", best_MSE)
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r", corr[0])

            mlflow.pytorch.log_model(best_model, "model")

            Lrun_name = cfg.experiment.run.replace('-',' ').split('_')
            grid = sns.jointplot(x=affinities, y=predictions, space=0.0, height=3, s=10, edgecolor='w', ylim=(0, 16), xlim=(0, 16),alpha=.5)
            grid.ax_joint.text(0.5, 11, f'cnn-lstm {Lrun_name[2]}\nR= {corr[0]:.2f} - RMSE= {rmse:.2f}\n{Lrun_name[0]}\n{Lrun_name[1]}',size=8)
            grid.set_axis_labels('real','predicted',size=8)
            grid.ax_joint.set_xticks(range(0, 16, 5))
            grid.ax_joint.set_yticks(range(0, 16, 5))
            grid.ax_joint.xaxis.set_label_coords(0.5,-0.13)
            grid.ax_joint.yaxis.set_label_coords(-0.15,0.5)
            if cfg.network.mean_test:
                grid.fig.savefig(f'{output_pdf}/correlation_plot/CNN-LSTM_mean_{cfg.experiment.run}.pdf')
            elif cfg.network.sim_test:
                grid.fig.savefig(f'{output_pdf}/correlation_plot/CNN-LSTM_all_{cfg.experiment.run}.pdf')
            else:
                grid.fig.savefig(f'{output_pdf}/correlation_plot/CNN-LSTM_random_{cfg.experiment.run}.pdf')

    gpu_memory = torch.cuda.max_memory_allocated()
    print(f"--\nGPU usage on GPU {DIST.local_rank}: {convert_byte(gpu_memory)}\n--")


if __name__ == "__main__":
    setup()
    main()
    cleanup()

