import os
import seaborn as sns
import pandas as pd
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


@hydra.main(config_path="./configs", config_name=f"default_datasetv4")
def main(cfg: DictConfig) -> None:
    torch.cuda.set_device(int(os.environ['SLURM_LOCALID']))
    work = os.environ['WORK']
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(OmegaConf.to_yaml(cfg))
    by_complex = cfg.experiment.by_complex
    batch_size = cfg.training.batch_size

    model = load_model(cfg, dev)

    # test dataset
    if cfg.network.sim_test:
        test_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex=False, mode="test", debug=cfg.debug) #test on all simulations of test set
    else:
        test_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex=True, mode="test", debug=cfg.debug)

    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                         persistent_workers=True)

    # apply model on test
    affinities, predictions = predict(model, test_dl, len(test_ds), dev)
    
    # compute metrics
    rmse, mae, corr = analyse(affinities, predictions)
    print(affinities)
    print(predictions)
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
        print(pdbs)
    Lrun_name = cfg.experiment.run.replace('-',' ').split('_')
    grid = sns.jointplot(x=affinities, y=predictions, space=0.0, height=3, s=10, edgecolor='w', ylim=(0, 16), xlim=(0, 16),alpha=.5)
    grid.ax_joint.text(0.5, 11, f'cnn-lstm {Lrun_name[2]}\nR= {corr[0]:.2f} - RMSE= {rmse:.2f}\n{Lrun_name[0]}\n{Lrun_name[1]}',size=8)
    grid.set_axis_labels('real','predicted',size=8)
    grid.ax_joint.set_xticks(range(0, 16, 5))
    grid.ax_joint.set_yticks(range(0, 16, 5))
    grid.ax_joint.xaxis.set_label_coords(0.5,-0.13)
    grid.ax_joint.yaxis.set_label_coords(-0.15,0.5)
    if cfg.network.mean_test:
        grid.fig.savefig(f'{work}/deep_learning/MD_ConvLSTM/cnn_fcn_lstm/correlation_plot/CNN-LSTM_mean_{cfg.experiment.run}.pdf')
    elif cfg.network.sim_test:
        grid.fig.savefig(f'{work}/deep_learning/MD_ConvLSTM/cnn_fcn_lstm/correlation_plot/CNN-LSTM_all_{cfg.experiment.run}.pdf')
    else:
        grid.fig.savefig(f'{work}/deep_learning/MD_ConvLSTM/cnn_fcn_lstm/correlation_plot/CNN-LSTM_random_{cfg.experiment.run}.pdf')


if __name__ == "__main__":
    if int(os.environ['SLURM_PROCID']) == 0:
        print(f"Running test on one GPU SLURM_PROCID: {os.environ['SLURM_PROCID']}  SLURM_LOCALID {os.environ['SLURM_LOCALID']}")
        main()
