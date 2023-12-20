import os
import time
import seaborn as sns
import pandas as pd
import hydra
import numpy as np
import scipy
import scipy.stats
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch import device, cuda

from codes.complex_dataset import Complexes_4DDataset
from codes.videonucy import create_videonucy

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


@hydra.main(config_path="./configs", config_name=f"videonucy")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    batch_size = cfg.training.batch_size
    # load model
    if cuda.is_available():
        dev = device("cuda")
        print('Running DL with gpu')
    else:
        dev = device("cpu")
        print('Running DL with cpu')
    
    if cfg.network.pretrained_path == 'True':
        model_path = os.path.join(cfg.experiment.model_path, f"convlstm_{cfg.experiment.run}.pth")
    else:
        model_path = f"{cfg.network.pretrained_path.rstrip('.pth')}.pth" #../../../models/complex/convlstm_5932-complexes_58300-simulations_by-complex.pth
    checkpoint = torch.load(model_path, map_location=dev)
    model = create_videonucy(cfg.network.convlstm_cfg, cfg.network.fc_cfg)
    model.load_state_dict(checkpoint['model'])
    model.to(dev)

    print(f"Model {model_path} loaded")
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
        DFresults.to_csv(f'../../../results/ConvLSTM_mean_{cfg.experiment.run}.csv')
    elif cfg.network.sim_test:
        grid.fig.savefig(f'../../../results/ConvLSTM_all_{cfg.experiment.run}.pdf')
        DFresults.to_csv(f'../../../results/ConvLSTM_all_{cfg.experiment.run}.csv')
    else:
        grid.fig.savefig(f'../../../results/ConvLSTM_random_{cfg.experiment.run}.pdf')
        DFresults.to_csv(f'../../../results/ConvLSTM_random_{cfg.experiment.run}.csv')


if __name__ == "__main__":
    main()
