import os
import time

import hydra
import numpy as np
import scipy
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchinfo import summary

from codes.pt_data import ProteinLigand_3DDataset
from codes.raw_data import RawDataset


def convert_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def test(model, dataloader, no_of_samples):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    # lr = LinearRegression()
    # lr.fit(predictions, affinities)
    # y_ = lr.predict(predictions)
    # sd = (((affinities - y_) ** 2).sum() / (len(affinities) - 1)) ** 0.5

    print(f"""
    Analysis:
        rmse= {rmse}
        mae= {mae}
        corr= {corr}
    """)


@hydra.main(config_path="./configs", config_name="default_test")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # load model
    model_path = os.path.join(cfg.path, cfg.model)
    model = torch.load(model_path)

    summary(model, input_size=(10, 19, 25, 25, 25))

    # test dataset
    raw_data_test = RawDataset(cfg.data.input_dir, 'test', cfg.data.max_dist)
    raw_data_test.load_data()
    raw_data_test.set_normalization_params(cfg.partialcharge.m, cfg.partialcharge.std)
    raw_data_test.charge_normalization()
    print(raw_data_test)

    test_dataset = ProteinLigand_3DDataset(raw_data_test,
                                           grid_spacing=cfg.data.grid_spacing, rotations=None)
    batch_size = min(len(test_dataset), cfg.data.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                 persistent_workers=True)

    # apply model on test
    affinities, predictions = test(model, test_dataloader, len(test_dataset))

    # compute metrics
    analyse(affinities, predictions)


if __name__ == "__main__":
    my_app()
