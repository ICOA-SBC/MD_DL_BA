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
from torch import nn, device, cuda, set_grad_enabled
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
##### LOCAL
from codes.videonucy import create_videonucy
from codes.complex_dataset import Complexes_4DDataset
from codes.tools import Distribution as DIST
from codes.tools import convert_byte, convert_time, master_print
from codes.transformations import build_rotations
from videonucy_test import analyse, predict


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
        filename = os.path.join(cfg.experiment.model_path, f"convlstm_{cfg.experiment.run}.pth")
    else:
        filename = f"{cfg.network.pretrained_path.rstrip('.pth')}.pth"
    print(f"\tloading model {filename}")
    checkpoint = torch.load(filename, map_location=dev)
    model = create_videonucy(cfg.network.convlstm_cfg, cfg.network.fc_cfg)

    model.load_state_dict(checkpoint['model'])
    print(f"Model successfully loaded")
    model.to(dev)
    return model

def save_model(model, epoch, model_name, cfg, optimizer=None):
    filename = os.path.join(cfg.experiment.model_path, f"convlstm_{model_name}.pth")
    print(f"\tsaving model {filename} on rank {DIST.rank}")
    model_without_ddp = model.module

    if optimizer is None:
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': None,
            'epoch': epoch,
            'cfg': cfg}
    else:
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'cfg': cfg}

    torch.save(checkpoint, filename)

### TRAIN
def train(model, dl, ds_size, cfg, device, batch_size):

    ### TRAINING SETUP
    metric = nn.MSELoss(reduction='sum')
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.training.learning_rate,
                           weight_decay=cfg.training.weight_decay)

    best_mse, best_epoch = 100, -1
    patience, max_patience = 0, cfg.training.patience

    ### TRAINING PHASE
    time_train = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=1, warmup=3, active=12, repeat=2),
                 on_trace_ready=tensorboard_trace_handler('./profiler'),
                 profile_memory=True) as prof:
        for epoch in range(cfg.training.num_epochs):
            master_print(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
            time_epoch = time.time()
            for phase in ['train', 'val']:
                master_print(f"\tPhase {phase}")
                model.train() if phase == 'train' else model.eval()

                running_metrics = 0.0

                for (*inputs, labels) in dl[phase]:
                    inputs = torch.stack(inputs, dim=1)
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    optimizer.zero_grad()

                    with set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        gpu_loss = criterion(outputs.float(), labels.float())
                        loss = gpu_loss / batch_size

                        # statistics
                        running_metrics += metric(outputs, labels).cpu().item()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        prof.step()

                running_metrics = torch.tensor(running_metrics).to(device)
                dist.all_reduce(running_metrics, op=dist.ReduceOp.SUM)
                epoch_metric = running_metrics.cpu().item() / ds_size[phase]
                if phase == 'train':
                    master_print(f"\t\t[{phase}] MSELoss {epoch_metric:.4f}")
                else:
                    master_print(f"\t\t[{phase}] MSELoss {epoch_metric:.4f}\
                    \t Duration: {time.time() - time_epoch:.2f}")

                # check if the model is improving
                if phase == 'val':
                    if epoch_metric < best_mse:
                        master_print(f"\t\t/Better loss {best_mse} --> {epoch_metric}\\ ")
                        best_mse, best_epoch = epoch_metric, epoch
                        best_model_wts = copy.deepcopy(model.state_dict())
                        if DIST.master_rank:
                            model_name=f"{cfg.experiment.run}_{best_mse:.4f}_{best_epoch}"
                            save_model(model, best_epoch, model_name, cfg, optimizer)
                        patience = 0
                    else:
                        patience += 1

                if DIST.master_rank:
                    mlflow.log_metric(f"{phase}_mse", epoch_metric, epoch)

            if patience > max_patience:
                master_print("----------- Early stopping activated !")
                break

    duration = time.time() - time_train
    master_print(f"[{epoch + 1} / {cfg.training.num_epochs}] Best mean MSE: {best_mse:.4f} at \
            epoch {best_epoch} \n\tTotal duration: {convert_time(duration)}")

    model.load_state_dict(best_model_wts)
    return model, best_epoch, best_mse

def mlflow_setup(cfg):
    mlflow.log_param("n_frames", cfg.data_setup.frames)
    mlflow.log_param("learning_rate", cfg.training.learning_rate)
    mlflow.log_param("weight_decay", cfg.training.weight_decay)
    mlflow.log_param("batch_size", cfg.training.batch_size * DIST.size)
    mlflow.log_param("patience", cfg.training.patience)
    mlflow.log_param("max_epoch", cfg.training.num_epochs)
    # log configuration as an artefact
    with open('./configuration.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    mlflow.log_artifact("configuration.yaml")

@hydra.main(config_path="./configs", config_name=f"videonucy")
def main(cfg: DictConfig) -> None:
    by_complex = cfg.experiment.by_complex

    # create transformations
    rotations_matrices = build_rotations()
    master_print(f"Number of available rotations: {len(rotations_matrices)}")

    train_ds = Complexes_4DDataset(cfg.io,
                                   cfg.data_setup,
                                   by_complex,
                                   rotations_matrices,
                                   mode="train",
                                   debug=cfg.debug)
    val_ds = Complexes_4DDataset(cfg.io,
                                 cfg.data_setup,
                                 by_complex,
                                 mode="val",
                                 debug=cfg.debug)

    train_sampler = DistributedSampler(train_ds,
                                       num_replicas=DIST.size,
                                       rank=DIST.rank,
                                       shuffle=True)
    val_sampler = DistributedSampler(val_ds,
                                     num_replicas=DIST.size,
                                     rank=DIST.rank,
                                     shuffle=False)

    ds_size = {'train': len(train_ds), 'val': len(val_ds)}
    batch_size_per_gpu = cfg.training.batch_size
    batch_size = batch_size_per_gpu * DIST.size

    train_dl = DataLoader(train_ds,
                          batch_size=batch_size_per_gpu,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True,
                          drop_last=True,
                          persistent_workers=True,
                          sampler=train_sampler)
    val_dl = DataLoader(val_ds,
                        batch_size=batch_size_per_gpu,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True,
                        persistent_workers=True,
                        sampler=val_sampler)

    dl = {'train': train_dl, 'val': val_dl}

    if cuda.is_available():
        dev = device("cuda")
        print('Running DL with gpu')
    else:
        dev = device("cpu")
        print('Running DL with cpu')
    cuda.set_device(DIST.local_rank)

    master_print(f"Training with {DIST.size} gpu !")

    if cfg.network.pretrained_path: #TRUE or Path
        model = load_model(cfg,dev)
    else:
        model = create_videonucy(cfg.network.convlstm_cfg, cfg.network.fc_cfg)
    model.to(DIST.local_rank)
    model.to(dev)

    ddp_model = DDP(model, device_ids=[DIST.local_rank])

    if DIST.master_rank:
        try:
            mlflow.end_run()
        except:
            print("mlflow not running")

        mlflow.set_tracking_uri(cfg.experiment.path)
        mlflow.set_experiment(cfg.experiment.name)


    with mlflow.start_run(run_name=cfg.experiment.run) as run:
        if DIST.master_rank:
            mlflow_setup(cfg)

        #train
        best_model, best_epoch, best_MSE = train(ddp_model,
                                       dl,
                                       ds_size,
                                       cfg,
                                       DIST.local_rank,
                                       batch_size)

        if DIST.master_rank:
            # the best model is saved without rmse (easier to load for testing)
            print('saving best model')
            save_model(best_model, best_epoch, cfg.experiment.run, cfg)
            mlflow.log_param("best_epoch", best_epoch)
            mlflow.log_metric("best_val_MSELoss", best_MSE)

            # test dataset
            if cfg.network.sim_test:
                test_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex=False, mode="test", debug=cfg.debug) #test on all simulation from test set
            else:
                test_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex=True, mode="test", debug=cfg.debug)
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

            # apply model on test
            affinities, predictions = predict(model, test_dl, len(test_ds), dev)

            # compute metrics
            rmse, mae, corr = analyse(affinities, predictions)

            if cfg.network.mean_test or cfg.network.sim_test:
                pdbs = [pdb.split('/')[-2] for pdb in test_ds.samples_list]
            else:
                pdbs = [pdb.split('/')[-1] for pdb in test_ds.samples_list]
            assert len(affinities) == len(pdbs)
            DFresults = pd.DataFrame([pdbs,affinities,predictions]).T
            DFresults.columns = ['pdbid','real','predictions']
            DFresults[['real','predictions']] = DFresults[['real','predictions']].astype(float)

            if cfg.network.mean_test:
                DFgroupby = DFresults.groupby('pdbid')
                DFresults = pd.concat([DFgroupby['real'].first(), DFgroupby['predictions'].apply(list), DFgroupby['predictions'].mean().round(2).rename('mean_pred')], axis=1)
                affinities = DFresults.real
                predictions = DFresults.mean_pred
            else:
                DFresults = DFresults.set_index('pdbid')

            mlflow.log_param("best_epoch", best_epoch)
            mlflow.log_metric("best_val_MSELoss", best_MSE)
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r", corr[0])

            Lrun_name = cfg.experiment.run.replace('-',' ').split('_')
            grid = sns.jointplot(x=affinities, y=predictions, space=0.0, height=3, s=10, edgecolor='w', ylim=(0, 16), xlim=(0, 16), alpha=.5)
            if len(Lrun_name) == 3:
                grid.ax_joint.text(0.5, 11, f'convlstm {Lrun_name[2]}\nR= {corr[0]:.2f} - RMSE= {rmse:.2f}\n{Lrun_name[0]}\n{Lrun_name[1]}', size=8)
            elif len(Lrun_name) == 2:
                grid.ax_joint.text(0.5, 11, f'convlstm {Lrun_name[1]}\nR= {corr[0]:.2f} - RMSE= {rmse:.2f}\n{Lrun_name[0]}', size=8)
            elif len(Lrun_name) == 1:
                grid.ax_joint.text(0.5, 12.5, f'convlstm {Lrun_name[0]}\nR= {corr[0]:.2f} - RMSE= {rmse:.2f}', size=8)

            grid.set_axis_labels('real','predicted', size=8)
            grid.ax_joint.set_xticks(range(0, 16, 5))
            grid.ax_joint.set_yticks(range(0, 16, 5))
            grid.ax_joint.xaxis.set_label_coords(0.5, -0.13)
            grid.ax_joint.yaxis.set_label_coords(-0.15, 0.5)
            if cfg.network.mean_test:
                grid.fig.savefig(f'../../../results/ConvLSTM_mean_{cfg.experiment.run}.pdf')
                DFresults.to_csv(f'../../../results/CNN-LSTM_mean_{cfg.experiment.run}.csv')
            elif cfg.network.sim_test:
                grid.fig.savefig(f'../../../results/ConvLSTM_all_{cfg.experiment.run}.pdf')
                DFresults.to_csv(f'../../../results/CNN-LSTM_all_{cfg.experiment.run}.csv')
            else:
                grid.fig.savefig(f'../../../results/ConvLSTM_random_{cfg.experiment.run}.pdf')
                DFresults.to_csv(f'../../../results/CNN-LSTM_random_{cfg.experiment.run}.csv')

            print(f"GPU usage: {convert_byte(cuda.max_memory_allocated(device=None))}")


if __name__ == "__main__":
    setup()
    main()
    cleanup()

