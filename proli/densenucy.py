### IMPORT LIBRARIES
##### BUILT_IN
import time
import os
import copy
import mlflow
import seaborn as sns
import pandas as pd
##### EXTERNAL
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
import torch.optim as optim
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
##### LOCAL
from codes.densenucy import create_densenucy
from codes.pt_data import ProteinLigand_3DDataset
from codes.raw_data import RawDataset
from codes.tools import Distribution as DIST
from codes.tools import convert_byte, convert_time, master_print
from codes.transformations import build_rotations
from proli_test import analyse, predict

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
    master_print(f"\tloading model {filename}")
    checkpoint = torch.load(filename, map_location=dev)
    model = create_densenucy(
            cfg.network.growth_rate,
            cfg.network.dense_cfg,
            cfg.network.fc_cfg,
            cfg.network.pretrained_path
        )

    model.load_state_dict(checkpoint['model'])
    master_print(f"Model successfully loaded")
    model.to(dev)
    return model

def save_model(model, pathname, experiment_name, run_name, rmse):
    filename = os.path.join(pathname, f"{experiment_name}_{run_name}_{rmse:.4f}.pth")
    master_print(f"\tsaving model {filename} on rank {DIST.rank}")

    torch.save(model, filename)

### TRAIN
def train(model, dataloaders, dataset_size, cfg, device):

    ### TRAINING SETUP
    metric = nn.MSELoss(reduction='sum')
    criterion = nn.MSELoss(reduction='mean')
    cfg_train, cfg_model_path, cfg_exp = cfg.training, cfg.io.model_path, cfg.experiment_name
    optimizer = optim.Adam(model.parameters(), lr=cfg_train.learning_rate, weight_decay=cfg_train.weight_decay)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_MSE, best_epoch = 100.0, -1
    patience, max_patience = 0, cfg_train.patience

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model.to(device)

    ### TRAINING PHASE
    time_train = time.time()

    for epoch in range(cfg_train.num_epochs):
        master_print(f"Epoch {epoch + 1}/{cfg_train.num_epochs}")
        time_epoch = time.time()

        for phase in ['train', 'val']:
            master_print(f"Phase: {phase} ")
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_metrics = 0.0

            for (inputs, labels) in dataloaders[phase]:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels.float())

                    # statistics
                    running_metrics += metric(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # synchronization
            # https://discuss.pytorch.org/t/reducelronplateau-usage-with-distributeddataparallel/117509
            dist.barrier()
            dist.all_reduce(running_metrics, op=dist.ReduceOp.SUM)
            epoch_metric = running_metrics.cpu().item() / dataset_size[phase]

            if phase == 'train':
                master_print(f"\t[{phase}] MSELoss {epoch_metric:.4f}")
            else:
                master_print(f"\t[{phase}] MSELoss {epoch_metric:.4f} \t Duration: {time.time() - time_epoch:.2f}")

            # check if the model is improving
            if phase == 'val':
                if epoch_metric < best_MSE:
                    master_print(f"/\\ Better loss {best_MSE} --> {epoch_metric}")
                    best_MSE, best_epoch = epoch_metric, epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if DIST.master_rank:
                        save_model(model, cfg_model_path, cfg_exp, best_epoch, best_MSE)
                    patience = 0
                else:
                    patience += 1

            mlflow.log_metric(f"{phase}_MSELoss", epoch_metric, epoch)

        if patience > max_patience:
            master_print("----------- Early stopping activated !")
            break

    duration = time.time() - time_train
    master_print("\n\n_____________________________________________")
    master_print(f"[{epoch + 1} / {cfg_train.num_epochs}] Best mean MSE: {best_MSE:.4f} at epoch {best_epoch} \
            \n\tTotal duration: {convert_time(duration)}")
    master_print("_____________________________________________")

    model.load_state_dict(best_model_wts)

    return model, best_epoch, best_MSE

def test(model, test_dataloader, test_samples, device):
    metric = nn.MSELoss(reduction='sum')

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    running_metrics = 0.0

    for (inputs, labels) in test_dataloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            running_metrics += metric(outputs, labels).cpu().item()

    metric = running_metrics / test_samples
    master_print(f"\t[Test] MSELoss {metric:.4f}")
    mlflow.log_metric(f"test_MSELoss", metric)

def mlflow_setup(cfg):
    if DIST.master_rank:
        mlflow.log_param("learning_rate", cfg.training.learning_rate)
        mlflow.log_param("patience", cfg.training.patience)
        mlflow.log_param("max_epoch", cfg.training.num_epochs)
        mlflow.log_param("batch_size", cfg.training.batch_size)
        mlflow.log_param("weight_decay", cfg.training.weight_decay)
        mlflow.log_param("growth_rate", cfg.network.growth_rate)
        mlflow.log_param("dense_cfg", cfg.network.dense_cfg)
        mlflow.log_param("fc_cfg", cfg.network.fc_cfg)

        # log configuration as an artefact
        with open('./configuration.yaml', 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))
        mlflow.log_artifact("configuration.yaml")

@hydra.main(config_path="./configs", config_name="dense")
def main(cfg: DictConfig) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(DIST.local_rank)
    master_print(f"Training with {DIST.size} gpu !")
    work = os.environ['WORK']

    # load hdf data in memory
    raw_data_train = RawDataset(cfg.io.input_dir, 'training', cfg.data.max_dist)
    raw_data_train.load_data()
    master_print(raw_data_train)
    batch_size = cfg.training.batch_size

    if not cfg.training.only_test:
        raw_data_valid = RawDataset(cfg.io.input_dir, 'validation', cfg.data.max_dist)
        raw_data_valid.load_data()
        raw_data_valid.set_normalization_params(*raw_data_train.get_normalization_params())
        master_print(raw_data_valid)

        # update raw data
        raw_data_train.charge_normalization()
        raw_data_valid.charge_normalization()

        # create transformations
        rotations_matrices = build_rotations()
        master_print(f"Number of available rotations: {len(rotations_matrices)}")

        transform, target_transform = None, None

        # create dataset (pt) and dataloader
        train_dataset = ProteinLigand_3DDataset(raw_data_train,
                                            grid_spacing=cfg.data.grid_spacing, rotations=rotations_matrices)
        valid_dataset = ProteinLigand_3DDataset(raw_data_valid,
                                            grid_spacing=cfg.data.grid_spacing, rotations=None)

        dataset_size = {'train': len(train_dataset), 'val': len(valid_dataset)}

        train_sampler = DistributedSampler(train_dataset, num_replicas=DIST.size, rank=DIST.rank, shuffle=True)
        val_sampler = DistributedSampler(valid_dataset, num_replicas=DIST.size, rank=DIST.rank, shuffle=True)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        dataloaders = {'train': train_dataloader, 'val': valid_dataloader}

    # create model

    if cfg.network.pretrained_path:
        model = load_model(cfg, dev)
    else:
        model = create_densenucy(
               cfg.network.growth_rate,
               cfg.network.dense_cfg,
               cfg.network.fc_cfg,
               cfg.network.pretrained_path
           )

    if DIST.master_rank:
        summary(model, input_size=(batch_size, 19, 25, 25, 25))

    model.to(DIST.local_rank)
    model.to(dev)
    # Convert BatchNorm to SyncBatchNorm.
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    ddp_model = DDP(model, device_ids=[DIST.local_rank])

    try:
        mlflow.end_run()
    except:
        master_print("mlflow not running")

    if DIST.master_rank:
        mlflow.set_tracking_uri(cfg.mlflow.path)
        mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        mlflow_setup(cfg)

        # train
        if cfg.training.only_test:
            best_model = model
        else:
            best_model, best_epoch, best_MSE = train(model, dataloaders, dataset_size, cfg, DIST.local_rank)
            mlflow.log_param("best_epoch", best_epoch)
            mlflow.log_metric("best_val_MSELoss", best_MSE)

        # test on one GPU (for simplicity)
        if DIST.master_rank:
            if cfg.io.specific_test_dir:
                raw_data_test = RawDataset(cfg.io.specific_test_dir, 'test', cfg.data.max_dist) #by default CoG_12
            else:
                raw_data_test = RawDataset(cfg.io.input_dir, 'test', cfg.data.max_dist) #if none: augmented test set
            raw_data_test.load_data()
            raw_data_test.set_normalization_params(*raw_data_train.get_normalization_params())
            raw_data_test.charge_normalization()
            print(raw_data_test)

            test_dataset = ProteinLigand_3DDataset(raw_data_test,
                                               grid_spacing=cfg.data.grid_spacing, rotations=None)

            test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 4,
                                     shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
            print(f"--------------------- Running test")
            test(best_model, test_dataloader, len(test_dataset),dev)
            print(f"--------------------- Running predict")
            affinities, predictions = predict(best_model, test_dataloader, len(test_dataset))
            pdbs = [pdb for pdb in raw_data_test.ids]
            DFresults = pd.DataFrame([pdbs,affinities,predictions]).T
            DFresults.columns= ['pdbid','real','prediction']
            DFresults.to_csv(f'{work}/deep_learning/MD_ConvLSTM/proli/correlation_plot/Densenucy_{cfg.mlflow.run_name}.csv')

            rmse, mae, corr = analyse(affinities, predictions)
            mlflow.log_metric(f"test_rmse", rmse)
            mlflow.log_metric(f"test_mae", mae)
            mlflow.log_metric(f"test_r", corr[0])
            mlflow.log_metric(f"test_p-value", corr[1])

            print(f"[TEST] rmse: {rmse:.4f} mae: {mae:.4f} corr: {corr}")

            mlflow.pytorch.log_model(best_model, "model")

            if not cfg.training.only_test:
                save_model(best_model, cfg.io.model_path, cfg.experiment_name, cfg.mlflow.run_name, rmse)

            Lrun_name = cfg.mlflow.run_name.replace('-',' ').split('_')
            grid = sns.jointplot(x=affinities, y=predictions, space=0.0, height=3, s=10, edgecolor='w', ylim=(0, 16), xlim=(0, 16), alpha=.5)
            if len(Lrun_name) == 4:
                grid.ax_joint.text(0.5, 11, f'proli - R= {corr[0]:.2f} - RMSE= {rmse:.2f}\n{Lrun_name[0]}\n{Lrun_name[1]}\n{Lrun_name[2]}\n{Lrun_name[3]}', size=7)
            elif len(Lrun_name) == 3:
                grid.ax_joint.text(0.5, 11, f'proli - R= {corr[0]:.2f} - RMSE= {rmse:.2f}\n{Lrun_name[0]}\n{Lrun_name[1]}\n{Lrun_name[2]}', size=7)
            elif len(Lrun_name) == 2:
                grid.ax_joint.text(0.5, 11, f'proli - R= {corr[0]:.2f} - RMSE= {rmse:.2f}\n{Lrun_name[0]}\n{Lrun_name[1]}', size=7)
            elif len(Lrun_name) == 1:
                grid.ax_joint.text(0.5, 11, f'proli - R= {corr[0]:.2f} - RMSE= {rmse:.2f}\n{Lrun_name[0]}', size=7)
            grid.set_axis_labels('real','predicted', size=8)
            grid.ax_joint.set_xticks(range(0, 16, 5))
            grid.ax_joint.set_yticks(range(0, 16, 5))
            grid.ax_joint.xaxis.set_label_coords(0.5, -0.13)
            grid.ax_joint.yaxis.set_label_coords(-0.15, 0.5)
            grid.fig.savefig(f'{work}/deep_learning/MD_ConvLSTM/proli/correlation_plot/Densenucy_{cfg.mlflow.run_name}.pdf')
    gpu_memory = torch.cuda.max_memory_allocated()
    master_print(f"--\nGPU usage on GPU {DIST.local_rank}: {convert_byte(gpu_memory)}\n--")


if __name__ == "__main__":
    setup()
    main()
    cleanup()

