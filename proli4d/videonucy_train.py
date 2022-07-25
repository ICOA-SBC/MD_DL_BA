### IMPORT LIBRARIES
##### BUILT_IN
import time
import os
import copy
import mlflow
##### EXTERNAL
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch import nn, device, cuda, set_grad_enabled
from torch.utils.data import DataLoader
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
##### LOCAL
import idr_torch 
from codes.videonucy import create_videonucy
from codes.complex_dataset import Complexes_4DDataset
from codes.tools import convert_byte
from codes.transformations import build_rotations


### SETUP PROCESS COMMUNICATION
dist.init_process_group(backend='nccl', 
                        init_method='env://', 
                        world_size=idr_torch.size, 
                        rank=idr_torch.rank)

torch.cuda.set_device(idr_torch.local_rank)


### UTILS
def master_print(msg):
    if idr_torch.rank == 0:
        print(msg)
        
def convert_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def save_model(model, model_path, model_name):
    filename = os.path.join(model_path, f"{model_name}.pth")
    print(f"\t\t\tSaving model {filename}")
    torch.save(model, filename)
    
def mlflow_setup(cfg):
    mlflow.log_param("n_frames", cfg.data_setup.frames)
    mlflow.log_param("learning_rate", cfg.training.learning_rate)
    mlflow.log_param("weight_decay", cfg.training.weight_decay)
    mlflow.log_param("batch_size", cfg.training.batch_size * idr_torch.size)
    mlflow.log_param("patience", cfg.training.patience)
    mlflow.log_param("max_epoch", cfg.training.num_epochs)
    # log configuration as an artefact
    with open('./configuration.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    mlflow.log_artifact("configuration.yaml")


### TRAIN
def train(model, dl, ds_size, cfg, cfg_exp, device, batch_size):
    
    ### TRAINING SETUP
    metric = nn.MSELoss(reduction='sum')
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), 
                           lr=cfg.learning_rate, 
                           weight_decay=cfg.weight_decay)

    best_mse, best_epoch = 100, -1
    patience, max_patience = 0, cfg.patience
    
    ### TRAINING PHASE
    time_train = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=1, warmup=3, active=12, repeat=5),
                 on_trace_ready=tensorboard_trace_handler('./profiler'),
                 profile_memory=True) as prof:
        for epoch in range(cfg.num_epochs):
            master_print(f"Epoch {epoch + 1}/{cfg.num_epochs}")
            time_epoch = time.time()
            for phase in ['train', 'val']:
                master_print(f"\tPhase {phase} ")
                model.train() if phase == 'train' else model.eval()

                running_metrics = 0.0

                for (*inputs, labels) in dl[phase]:
                    inputs = torch.stack(inputs, dim=1)
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
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
                if phase == 'val' and epoch_metric < best_mse:
                    master_print(f"\t\t/Better loss {best_mse} --> {epoch_metric}\\")
                    best_mse, best_epoch = epoch_metric, epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if idr_torch.rank == 0:
                        save_model(model, 
                                   cfg_exp.model_path,
                                   f"{cfg_exp.name}_{cfg_exp.run}_\
                                   {best_mse:.4f}_{best_epoch}")                 
                    patience = 0
                else:
                    patience += 1

                if idr_torch.rank == 0:
                    mlflow.log_metric(f"{phase}_mse", epoch_metric, epoch)

            if patience > max_patience:
                master_print("----------- Early stopping activated !")
                break
                
    duration = time.time() - time_train
    master_print(f"[{epoch + 1} / {cfg.num_epochs}] Best mean MSE: {best_mse:.4f} at \
            epoch {best_epoch} \n\tTotal duration: {convert_time(duration)}")

    model.load_state_dict(best_model_wts)
    return model, best_epoch


@hydra.main(config_path="./configs", config_name="videonucy")
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
                                       num_replicas=idr_torch.size,
                                       rank=idr_torch.rank,
                                       shuffle=True)
    val_sampler = DistributedSampler(val_ds,
                                     num_replicas=idr_torch.size,
                                     rank=idr_torch.rank,
                                     shuffle=False)
    
    ds_size = {'train': len(train_ds), 'val': len(val_ds)}
    batch_size_per_gpu = cfg.training.batch_size
    batch_size = batch_size_per_gpu * idr_torch.size

    train_dl = DataLoader(train_ds, 
                          batch_size=batch_size_per_gpu,
                          shuffle=False, 
                          num_workers=4, 
                          pin_memory=True, 
                          persistent_workers=True,
                          sampler=train_sampler)
    val_dl = DataLoader(val_ds, 
                        batch_size=batch_size_per_gpu,
                        shuffle=False, 
                        num_workers=4, 
                        pin_memory=True, 
                        persistent_workers=True,
                        sampler=val_sampler)
    
    dl = {'train': train_dl, 'val': val_dl}

    dev = device("cuda" if cuda.is_available() else "cpu")
    model = create_videonucy(cfg.network.convlstm_cfg, cfg.network.fc_cfg)
    model.to(dev)
    ddp_model = DDP(model, device_ids=[idr_torch.local_rank])

    if idr_torch.rank == 0:
        try:
            mlflow.end_run()
        except:
            print("mlflow not running")

        mlflow.set_tracking_uri(cfg.experiment.path)
        mlflow.set_experiment(cfg.experiment.name)    


    with mlflow.start_run(run_name=cfg.experiment.run) as run:
        
        if idr_torch.rank == 0:
            mlflow_setup(cfg)
        
        best_model, best_epoch = train(ddp_model,
                                       dl, 
                                       ds_size, 
                                       cfg.training,
                                       cfg.experiment, 
                                       dev,
                                       batch_size)

        if idr_torch.rank == 0:
            # the best model is saved without rmse (easier to load for testing)
            save_model(model, cfg.experiment.model_path, cfg.experiment.model_name) 
            mlflow.log_param("best_epoch", best_epoch)
            print(f"GPU usage: {convert_byte(cuda.max_memory_allocated(device=None))}")


if __name__ == "__main__":
    main()