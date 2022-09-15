import os

import hydra
import numpy as np
import scipy
import scipy.stats
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from codes.cnn_fcn_lstm import CNN_FCN_LSTM
from codes.complex_dataset import Complexes_4DDataset
from codes.tools import Distribution as DIST


def setup():
    # use slurm variables
    DIST()

    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=DIST.size,
                            rank=DIST.rank)
    torch.cuda.set_device(DIST.local_rank)


def cleanup():
    dist.destroy_process_group()


def load_model(cfg, dev):
    filename = os.path.join(cfg.experiment.model_path, f"{cfg.experiment.run}.pth")
    print(f"\tloading model {filename}")
    checkpoint = torch.load(filename, map_location=dev)
    model = CNN_FCN_LSTM(in_frames=cfg.data_setup.frames,
                         in_channels_per_frame=cfg.data_setup.features,
                         model_architecture=cfg.model.architecture)

    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch'] + 1
    print(f"Model successfully loaded")
    model.to(dev)
    return model


def predict(model, dataloader, no_of_samples, device):
    print(f"device : {device}")
    # model.to(device)
    model.eval()
    affinities = np.empty(0, dtype=np.float32)
    predictions = []

    for (inputs, labels) in dataloader:
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            preds = model(inputs)

        affinities = np.append(affinities, labels.numpy())
        predictions = np.append(predictions, preds.cpu().detach().numpy())

    print(f"Computed preds on {len(predictions)}/{len(affinities)} samples! (expected: {no_of_samples})")
    return affinities, predictions


def analyse(affinities, predictions):
    rmse = ((predictions - affinities) ** 2).mean() ** 0.5
    mae = (np.abs(predictions - affinities)).mean()
    corr = scipy.stats.pearsonr(predictions, affinities)

    print(f"""
    Analysis:
        rmse= {rmse}
        mae= {mae}
        corr= {corr}
    """)

    return rmse, mae, corr


@hydra.main(config_path="./configs", config_name="default_datasetv3")
def main(cfg: DictConfig) -> None:
    torch.cuda.set_device(int(os.environ['SLURM_LOCALID']))
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(OmegaConf.to_yaml(cfg))
    by_complex = cfg.experiment.by_complex
    batch_size = cfg.training.batch_size

    model = load_model(cfg, dev)

    # test dataset
    test_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex, mode="test", debug=cfg.debug)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                         persistent_workers=True)

    # apply model on test
    affinities, predictions = predict(model, test_dl, len(test_ds), dev)

    # compute metrics
    analyse(affinities, predictions)


if __name__ == "__main__":
    if int(os.environ['SLURM_PROCID']) == 0:
        print(f"Running test on one GPU SLURM_PROCID: {os.environ['SLURM_PROCID']}  SLURM_LOCALID {os.environ['SLURM_LOCALID']}")
        main()
