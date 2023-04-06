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
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
##### LOCAL
from codes.pafnucy import create_pafnucy
from codes.pt_data import ProteinLigand_3DDataset
from codes.raw_data import RawDataset
from codes.tools import convert_byte, convert_time, master_print
from codes.transformations import build_rotations
from proli_test import analyse, predict


### UTILS
def save_model(model, pathname, experiment_name, run_name, rmse, epoch, cfg):
    filename = os.path.join(pathname, f"{experiment_name}_{run_name}_{rmse:.4f}.pth")
    print(f"\tsaving model {filename}")

    torch.save(model, filename)


### TRAIN
def train(model, train_dataloader, valid_dataloader, dataset_size, cfg):

    ### TRAINING SETUP
    dataloaders = {'train': train_dataloader, 'val': valid_dataloader}
    metric = nn.MSELoss(reduction='sum')
    criterion = nn.MSELoss(reduction='mean')
    cfg_train, cfg_model_path, cfg_exp = cfg.training, cfg.io.model_path, cfg.experiment_name
    optimizer = optim.Adam(model.parameters(), lr=cfg_train.learning_rate, weight_decay=cfg_train.weight_decay)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_MSE, best_epoch = 100.0, -1
    patience, max_patience = 0, cfg_train.patience

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ### TRAINING PHASE
    time_train = time.time()

    for epoch in range(cfg_train.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg_train.num_epochs}")

        time_epoch = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_metrics = 0.0

            for (inputs, labels) in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # statistics
                    running_metrics += metric(outputs, labels).cpu().item()

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            epoch_metric = running_metrics / dataset_size[phase]
            if phase == 'train':
                print(f"\t[{phase}] MSELoss {epoch_metric:.4f}")
            else:
                print(f"\t[{phase}] MSELoss {epoch_metric:.4f} \t Duration: {time.time() - time_epoch:.2f}")

            # check if the model is improving
            if phase == 'val':
                if epoch_metric < best_MSE:
                    print(f"/\\ Better loss {best_MSE} --> {epoch_metric}")
                    best_MSE, best_epoch = epoch_metric, epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f"\tsaving model {cfg_exp}")
                    save_model(model, cfg_model_path, cfg_exp, best_epoch, best_MSE, best_epoch, cfg)

                    patience = 0
                else:
                    patience += 1

            mlflow.log_metric(f"{phase}_MSELoss", epoch_metric, epoch)

        if patience > max_patience:
            print("----------- Early stopping activated !")
            break

    duration = time.time() - time_train
    master_print("\n\n_____________________________________________")
    master_print(f"[{epoch + 1} / {cfg_train.num_epochs}] Best mean MSE: {best_MSE:.4f} at epoch {best_epoch} \
            \n\tTotal duration: {convert_time(duration)}")
    master_print("_____________________________________________")

    model.load_state_dict(best_model_wts)

    return model, best_epoch, best_MSE

def test(model, test_dataloader, test_samples):
    metric = nn.MSELoss(reduction='sum')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

@hydra.main(config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    work = os.environ['WORK']
    
    # load hdf data in memory
    raw_data_train = RawDataset(cfg.io.input_dir, 'training', cfg.data.max_dist)
    raw_data_train.load_data()
    print(raw_data_train)
    batch_size = cfg.training.batch_size

    if not cfg.training.only_test:
        raw_data_valid = RawDataset(cfg.io.input_dir, 'validation', cfg.data.max_dist)
        raw_data_valid.load_data()
        raw_data_valid.set_normalization_params(*raw_data_train.get_normalization_params())
        print(raw_data_valid)

        # update raw data
        raw_data_train.charge_normalization()
        raw_data_valid.charge_normalization()

        # create transformations
        rotations_matrices = build_rotations()
        print(f"Number of available rotations: {len(rotations_matrices)}")

        # create dataset (pt) and dataloader
        train_dataset = ProteinLigand_3DDataset(raw_data_train,
                                            grid_spacing=cfg.data.grid_spacing, rotations=rotations_matrices)
        valid_dataset = ProteinLigand_3DDataset(raw_data_valid,
                                            grid_spacing=cfg.data.grid_spacing, rotations=None)

        dataset_size = {'train': len(train_dataset), 'val': len(valid_dataset)}

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # create model

    model = create_pafnucy(
            conv_cfg=cfg.network.conv_channels,
            conv_kernel_size=cfg.network.conv_kernel_size,
            pool_kernel_size=cfg.network.pool_kernel_size,
            fc_cfg=cfg.network.dense_sizes,
            dropout_prob=cfg.network.drop_p
        )

    if cfg.network.pretrained_path:
        checkpoint = torch.load(f"{cfg.network.pretrained_path.strip('.pth')}.pth")
        model.load_state_dict(checkpoint['model'])
        print(f"Model successfully loaded")

    summary(model, input_size=(batch_size, 19, 25, 25, 25))

    try:
        mlflow.end_run()
    except:
        print("mlflow not running")

    mlflow.set_tracking_uri(cfg.mlflow.path)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        mlflow.log_param("batch_size", cfg.training.batch_size)
        mlflow.log_param("learning_rate", cfg.training.learning_rate)
        mlflow.log_param("weight_decay", cfg.training.weight_decay)
        mlflow.log_param("conv_cfg", cfg.network.conv_channels)
        mlflow.log_param("conv_kernel_size", cfg.network.conv_kernel_size)
        mlflow.log_param("pool_kernel_size", cfg.network.pool_kernel_size)
        mlflow.log_param("fc_cfg", cfg.network.dense_sizes)
        mlflow.log_param("dropout", cfg.network.drop_p)
        mlflow.log_param("patience", cfg.training.patience)
        mlflow.log_param("max_epoch", cfg.training.num_epochs)

        # train
        if cfg.training.only_test:
            best_model = model
        else:
            best_model, best_epoch, best_MSE = train(model, train_dataloader, valid_dataloader, dataset_size, cfg)
            mlflow.log_param("best_epoch", best_epoch)
            mlflow.log_metric("best_val_MSELoss", best_MSE)

        # test
        if cfg.io.specific_test_dir:
            raw_data_test = RawDataset(cfg.io.specific_test_dir, 'test', cfg.data.max_dist) #CoG_12 by default
        else:
            raw_data_test = RawDataset(cfg.io.input_dir, 'test', cfg.data.max_dist) #test set augmented
        raw_data_test.load_data()
        raw_data_test.set_normalization_params(*raw_data_train.get_normalization_params())
        raw_data_test.charge_normalization()
        print(raw_data_test)

        test_dataset = ProteinLigand_3DDataset(raw_data_test,
                                               grid_spacing=cfg.data.grid_spacing, rotations=None)
        print(test_dataset)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 4,
                                     shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        print(f"--------------------- Running test")
        test(best_model, test_dataloader, len(test_dataset))
        print(f"--------------------- Running predict")
        affinities, predictions = predict(best_model, test_dataloader, len(test_dataset))

        pdbs = [pdb for pdb in raw_data_test.ids]
        DFresults = pd.DataFrame([pdbs,affinities,predictions])
        DFresults.columns = ['pdbid','real','prediction']
        DFresults.to_csv(f'{work}/deep_learning/MD_ConvLSTM/proli/correlation_plot/Proli_{cfg.mlflow.run_name}.csv')

        rmse, mae, corr = analyse(affinities, predictions)

        mlflow.log_metric(f"test_rmse", rmse)
        mlflow.log_metric(f"test_mae", mae)
        mlflow.log_metric(f"test_r", corr[0])
        mlflow.log_metric(f"test_p-value", corr[1])

        print(f"[TEST] rmse: {rmse:.4f} mae: {mae:.4f} corr: {corr}")

        mlflow.pytorch.log_model(best_model, "model")

        if not cfg.training.only_test:
            save_model(best_model, cfg.io.model_path, cfg.experiment_name, cfg.mlflow.run_name, rmse, best_epoch, cfg)

        Lrun_name = cfg.mlflow.run_name.replace('-',' ').split('_')
        grid = sns.jointplot(x=affinities, y=predictions, space=0.0, height=3, s=10, edgecolor='w', ylim=(0, 16), xlim=(0, 16), alpha=.5)
        if 'complexes-with-MD' in cfg.mlflow.run_name:
            grid.ax_joint.text(0.5, 11, f'proli - R= {corr[0]:.2f} - RMSE= {rmse:.2f}\n{Lrun_name[0]}\n{Lrun_name[1]}\n{Lrun_name[2]}\n{Lrun_name[3]}', size=7)
        else:
            grid.ax_joint.text(0.5, 11, f'proli - R= {corr[0]:.2f} - RMSE= {rmse:.2f}\n{Lrun_name[0]}\n{Lrun_name[1]}\n{Lrun_name[2]}', size=7)
        grid.set_axis_labels('real','predicted', size=7)
        grid.ax_joint.set_xticks(range(0, 16, 5))
        grid.ax_joint.set_yticks(range(0, 16, 5))
        grid.ax_joint.xaxis.set_label_coords(0.5, -0.13)
        grid.ax_joint.yaxis.set_label_coords(-0.15, 0.5)
        grid.fig.savefig(f'{work}/deep_learning/MD_ConvLSTM/proli/correlation_plot/Proli_{cfg.mlflow.run_name}.pdf')

    gpu_memory = torch.cuda.max_memory_allocated()
    print(f"--\nGPU usage on GPU {DIST.local_rank}: {convert_byte(gpu_memory)}\n--")


if __name__ == "__main__":
    main()
