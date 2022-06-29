import time

import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
from torch import nn, device, cuda, set_grad_enabled
from torch.utils.data import DataLoader
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import idr_torch 
from codes.videonucy import create_videonucy
from codes.complex_dataset import Complexes_4DDataset
from codes.tools import convert_byte
from codes.transformations import build_rotations


dist.init_process_group(backend='nccl', 
                        init_method='env://', 
                        world_size=idr_torch.size, 
                        rank=idr_torch.rank)

torch.cuda.set_device(idr_torch.local_rank)


def convert_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def train(model, dl, ds_size, cfg, device, batch_size):
    
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
                 schedule=schedule(wait=1, warmup=1, active=12, repeat=5),
                 on_trace_ready=tensorboard_trace_handler('./profiler'),
                 profile_memory=True) as prof:
        for epoch in range(cfg.num_epochs):
            print(f"Epoch {epoch + 1}/{cfg.num_epochs}")
            time_epoch = time.time()
            for phase in ['train', 'val']:
                print(f"Phase: {phase} ")
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

                dist.all_reduce(running_metrics, op=dist.ReduceOp.SUM)
                epoch_metric = running_metrics / ds_size[phase]
                if phase == 'train':
                    print(f"\t[{phase}] MSELoss {epoch_metric:.4f}")
                else:
                    print(f"\t[{phase}] MSELoss {epoch_metric:.4f} \t\
                            Duration: {time.time() - time_epoch:.2f}")

                # check if the model is improving
                if phase == 'val' and epoch_metric < best_mse:
                    print(f"/\\ Better loss {best_mse} --> {epoch_metric}")
                    best_mse, best_epoch = epoch_metric, epoch
                    patience = 0
                else:
                    patience += 1

            if patience > max_patience:
                print("----------- Early stopping activated !")
                break
                
    duration = time.time() - time_train
    print(f"[{epoch + 1} / {cfg.num_epochs}] Best mean MSE: {best_mse:.4f} at \
            epoch {best_epoch} \n\tTotal duration: {convert_time(duration)}")

    return best_mse, best_epoch


@hydra.main(config_path="./configs", config_name="videonucy")
def main(cfg: DictConfig) -> None:
    
    by_complex = cfg.experiment.by_complex

    # create transformations
    rotations_matrices = build_rotations()
    print(f"Number of available rotations: {len(rotations_matrices)}")

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

    # train
    best_model, best_epoch = train(ddp_model, 
                                   dl, 
                                   ds_size, 
                                   cfg.training, 
                                   dev,
                                   batch_size)

    print(f"GPU usage: {convert_byte(cuda.max_memory_allocated(device=None))}")


if __name__ == "__main__":
    main()