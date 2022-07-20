import os
import time

import hydra
import numpy as np
import scipy
import scipy.stats
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from codes.complex_dataset import Complexes_4DDataset
from codes.videonucy import Videonucy

def predict(model, dataloader, no_of_samples, device):
    model.eval()
    affinities = np.empty(0, dtype=np.float32)
    predictions = []

    for (*inputs, labels) in dataloader:
        inputs = torch.stack(inputs, dim=1)
        inputs = inputs.to(device, non_blocking=True)
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


@hydra.main(config_path="./configs", config_name="videonucy")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    by_complex = cfg.experiment.by_complex
    batch_size = cfg.training.batch_size
    # load model
    model_path = os.path.join(cfg.experiment.model_path, f"{cfg.experiment.model_name}.pth")
    model = torch.load(model_path)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    print(f"Model {model_path} loaded")
    # test dataset
    test_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex, mode="test", debug=cfg.debug)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # apply model on test
    affinities, predictions = predict(model, test_dl, len(test_ds), dev)

    # compute metrics
    analyse(affinities, predictions)


if __name__ == "__main__":
    main()
