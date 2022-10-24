### IMPORT LIBRARIES
#----- BUILT_IN
import time
import os
import copy
#----- EXTERNAL
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
#----- LOCAL
from codes.densenucy import create_densenucy
from codes.pt_data import ProteinLigand_3DDataset
from codes.raw_data import RawDataset
from codes.transformations import build_rotations
import idr_torch


### SETUP PROCESS COMMUNICATION
dist.init_process_group(backend='nccl', 
                        init_method='env://', 
                        world_size=idr_torch.size, 
                        rank=idr_torch.rank)
# We suppose that for each task, we only use one GPU so we set here
# which GPU we will use among the GPUs available in a single computing node.
# For example, it is possible that we have 4 GPUs in a single node,
# to avoid that 4 executions of this script run into conflict for using the 
# same GPU, we make an explicit declaration.
torch.cuda.set_device(idr_torch.local_rank)


### UTILS
def master_print(msg):
    if idr_torch.rank == 0:
        print(msg)
        
def convert_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


### TRAIN
def multi_train(model, 
          train_dataloader, 
          valid_dataloader, 
          dataset_size, 
          batch_size, 
          cfg, 
          model_path, 
          name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ddp_model = DDP(model, device_ids=[idr_torch.local_rank])
    
    best_mse, best_epoch = 100, -1
    patience, max_patience = 0, cfg.patience
    
    dataloaders = {'train': train_dataloader, 'val': valid_dataloader}
    metric = nn.MSELoss(reduction='sum')
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(ddp_model.parameters(), 
                           lr=cfg.learning_rate, 
                           weight_decay=cfg.weight_decay)
    
    ### TRAINING PHASE
    time_train = time.time()
    for epoch in range(cfg.num_epochs):
        master_print(f"Epoch {epoch + 1}/{cfg.num_epochs}")
        
        time_epoch = time.time()
        for phase in ['train', 'val']:
            master_print(f"\tPhase {phase}")
            ddp_model.train() if phase == 'train' else ddp_model.eval()

            running_metrics = 0.0
            for (inputs, labels) in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = ddp_model(inputs)
                    gpu_loss = criterion(outputs.float(), labels.float())
                    loss = gpu_loss / batch_size
                    
                    # statistics
                    running_metrics += metric(outputs, labels).cpu().item()

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # metrics computation
            running_metrics = torch.tensor(running_metrics).to(device)
            dist.all_reduce(running_metrics, op=dist.ReduceOp.SUM)
            epoch_metric = running_metrics.cpu().item() / dataset_size[phase]
            
            if phase == 'train':
                master_print(f"\t\t[{phase}] MSELoss {epoch_metric:.4f}")
            else:
                master_print(f"\t\t[{phase}] MSELoss {epoch_metric:.4f}\
                \t Duration: {time.time() - time_epoch:.2f}")

            # check if the model is improving
            if phase == 'val' and epoch_metric < best_mse:
                master_print(f"\t\t/Better loss {best_mse} --> {epoch_metric}\\")
                best_mse, best_epoch = epoch_metric, epoch
                best_model_wts = copy.deepcopy(ddp_model.state_dict())
                
                filename = os.path.join(model_path, f"{name}_{best_mse:.4f}_{best_epoch}.pth")
                if idr_torch.rank == 0:
                    print(f"\tSaving model {filename}")
                    torch.save(ddp_model, filename)

                patience = 0
            elif phase == 'val':
                patience += 1
            
            if idr_torch.rank == 0:
                mlflow.log_metric(f"{phase}_MSELoss", epoch_metric, epoch)

        if patience > max_patience:
            print("----------- Early stopping activated !")
            break

    duration = time.time() - time_train

    master_print(f"[{epoch + 1} / {cfg.num_epochs}] Best mean MSE: {best_mse:.4f} \
            \n\tTotal duration: {convert_time(duration)}")

    model.load_state_dict(best_model_wts)
    return model, best_epoch


def test(model, test_dataloader, test_samples):
    metric = nn.MSELoss(reduction='sum')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    running_metrics = 0.0
    for (inputs, labels) in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            running_metrics += metric(outputs, labels).cpu().item()

    metric = running_metrics / test_samples
    print(f"\t[Test] MSELoss {metric:.4f}")
    mlflow.log_metric(f"test_MSELoss", metric)


@hydra.main(config_path="./configs", config_name="dense_multi")
def main(cfg: DictConfig) -> None:
    master_print(OmegaConf.to_yaml(cfg))
    
    # load hdf data in memory
    raw_data_train = RawDataset(cfg.io.input_dir, 'training', cfg.data.max_dist)
    raw_data_train.load_data()
    master_print(raw_data_train)

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
                                            grid_spacing=cfg.data.grid_spacing, 
                                            rotations=rotations_matrices)
    valid_dataset = ProteinLigand_3DDataset(raw_data_valid,
                                            grid_spacing=cfg.data.grid_spacing, 
                                            rotations=None)

    dataset_size = {'train': len(train_dataset), 'val': len(valid_dataset)}

    batch_size = cfg.training.batch_size
    batch_size_per_gpu = batch_size // idr_torch.size
    
    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas=idr_torch.size,
                                       rank=idr_torch.rank,
                                       shuffle=True,
                                       drop_last=True)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size_per_gpu,
                                  shuffle=False, 
                                  num_workers=4, 
                                  pin_memory=True,
                                  persistent_workers=True,
                                  sampler=train_sampler)
    val_sampler = DistributedSampler(valid_dataset,
                                     num_replicas=idr_torch.size,
                                     rank=idr_torch.rank,
                                     shuffle=False,
                                     drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, 
                                  batch_size=batch_size_per_gpu,
                                  shuffle=False, 
                                  num_workers=4, 
                                  pin_memory=True,
                                  persistent_workers=True,
                                  sampler=val_sampler)

    # create model
    model = create_densenucy(
        cfg.network.growth_rate,
        cfg.network.dense_cfg,
        cfg.network.fc_cfg
    )
    summary(model, input_size=(batch_size, 19, 25, 25, 25))

    # SET MLFLOW
    if idr_torch.rank == 0:
        try:
            mlflow.end_run()
        except:
            print("mlflow not running")
    
        mlflow.set_tracking_uri(cfg.mlflow.path)
        mlflow.set_experiment(cfg.mlflow.run_name)

        mlflow.start_run()
        mlflow.log_param("name", cfg.experiment_name)
        mlflow.log_param("batch_size", cfg.training.batch_size)
        mlflow.log_param("learning_rate", cfg.training.learning_rate)
        mlflow.log_param("weight_decay", cfg.training.weight_decay)
        mlflow.log_param("growth_rate", cfg.network.growth_rate)
        mlflow.log_param("dense_cfg", cfg.network.dense_cfg)
        mlflow.log_param("fc_cfg", cfg.network.fc_cfg)
        mlflow.log_param("patience", cfg.training.patience)
        mlflow.log_param("max_epoch", cfg.training.num_epochs)

    # train
    best_model, best_epoch = multi_train(model, 
                                   train_dataloader, 
                                   valid_dataloader, 
                                   dataset_size,
                                   batch_size,
                                   cfg.training,
                                   cfg.io.model_path, 
                                   cfg.experiment_name)
    
    # test
    if idr_torch.rank == 0:
        raw_data_test = RawDataset(cfg.io.input_dir, 'test', cfg.data.max_dist)
        raw_data_test.load_data()
        raw_data_test.set_normalization_params(*raw_data_train.get_normalization_params())
        raw_data_test.charge_normalization()
        print(raw_data_test)

        test_dataset = ProteinLigand_3DDataset(raw_data_test,
                                               grid_spacing=cfg.data.grid_spacing,
                                               rotations=None)

        test_dataloader = DataLoader(test_dataset, 
                                     batch_size=batch_size,
                                     shuffle=False, 
                                     num_workers=4, 
                                     pin_memory=True, 
                                     persistent_workers=True)

        test(best_model, test_dataloader, len(test_dataset))
        mlflow.log_param("best_epoch", best_epoch)
        mlflow.pytorch.log_model(best_model, "model")
        mlflow.end_run()


if __name__ == "__main__":
    main()
