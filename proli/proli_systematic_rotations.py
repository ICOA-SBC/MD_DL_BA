import copy
import os.path
import time

import hydra
import mlflow
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from codes.pafnucy import create_pafnucy
from codes.pt_data import ProteinLigand_3DDataset
from codes.raw_data import RawDataset
from codes.transformations import build_rotations
from proli_test import analyse, predict


def convert_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def train_with_rotations(model, raw_data_train, valid_dataloader, dataset_size, cfg, rotations, grid_spacing,
                         batch_size, model_path, name):
    metric = nn.MSELoss(reduction='sum')
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    no_of_rotations = len(rotations)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_MSE = 100.0
    best_epoch = -1
    patience = 0
    max_patience = cfg.patience * no_of_rotations

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    time_train = time.time()

    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}")
        time_epoch = time.time()

        for rot in range(no_of_rotations):
            # print(f"\tRotation {rot} / {no_of_rotations}")
            # recreate train_dataloader with one rotation
            train_dataset = ProteinLigand_3DDataset(raw_data_train,
                                                    grid_spacing=grid_spacing,
                                                    rotations=[rotations[rot]])  # only one rotation
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
            dataloaders = {'train': train_dataloader, 'val': valid_dataloader}
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
                    print(f"\t[{phase} - {rot}/{no_of_rotations}] MSELoss {epoch_metric:.4f}")
                else:
                    if rot == no_of_rotations - 1:
                        print(f"\t[{phase} - {rot}/{no_of_rotations}] MSELoss {epoch_metric:.4f} \
                                \t Duration: {time.time() - time_epoch:.2f}")

                # deep copy the model only at the end of the rotations
                if phase == 'val' and epoch_metric < best_MSE and rot == no_of_rotations - 1:
                    print(f"/\\ Better loss {best_MSE} --> {epoch_metric} at the end of an epoch * 24 rotations")
                    best_MSE, best_epoch = epoch_metric, epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    filename = os.path.join(model_path, f"{name}_{best_MSE:.4f}_{best_epoch}.pth")
                    print(f"\tsaving model {filename}")
                    torch.save(model, filename)
                    patience = 0
                else:
                    patience += 1

                mlflow.log_metric(f"{phase}_MSELoss", epoch_metric, epoch)

            if patience > max_patience:
                print("----------- Early stopping activated !")
                break

    duration = time.time() - time_train

    print(f"[{epoch + 1} / {cfg.num_epochs}] Best mean MSE: {best_MSE:.4f} at {best_epoch} epoch.\
            \n\tTotal duration: {convert_time(duration)}")

    model.load_state_dict(best_model_wts)

    return model, best_epoch


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


def save_model(model, pathname, experiment_name, run_name, rmse):
    filename = os.path.join(pathname, f"{experiment_name}_{run_name}_{rmse:.4f}.pth")
    print(f"\tsaving model {filename}")
    torch.save(model, filename)


@hydra.main(config_path="./configs", config_name="default_rotations")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # load hdf data in memory
    raw_data_train = RawDataset(cfg.io.input_dir, 'training', cfg.data.max_dist)
    raw_data_train.load_data()
    print(raw_data_train)

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
                                            grid_spacing=cfg.data.grid_spacing,
                                            rotations=None)  # done manually during training
    valid_dataset = ProteinLigand_3DDataset(raw_data_valid,
                                            grid_spacing=cfg.data.grid_spacing, rotations=None)

    dataset_size = {'train': len(train_dataset), 'val': len(valid_dataset)}

    batch_size = cfg.training.batch_size
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
    #                              shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # create model
    model = create_pafnucy(
        conv_cfg=cfg.network.conv_channels,
        conv_kernel_size=cfg.network.conv_kernel_size,
        pool_kernel_size=cfg.network.pool_kernel_size,
        fc_cfg=cfg.network.dense_sizes,
        dropout_prob=cfg.network.drop_p)
    summary(model, input_size=(batch_size, 19, 25, 25, 25))

    try:
        mlflow.end_run()
    except:
        print("mlflow not running")

    mlflow.set_tracking_uri(cfg.mlflow.path)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        mlflow.log_param("val_samples", len(valid_dataset))
        mlflow.log_param("max_epoch", cfg.training.num_epochs)
        mlflow.log_param("learning_rate", cfg.training.learning_rate)
        mlflow.log_param("weight_decay", cfg.training.weight_decay)
        mlflow.log_param("dropout", cfg.network.drop_p)
        mlflow.log_param("patience", cfg.training.patience)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("name", cfg.experiment_name)

        # train
        best_model, best_epoch = train_with_rotations(model, raw_data_train, valid_dataloader, dataset_size,
                                                      cfg.training, rotations_matrices,
                                                      cfg.data.grid_spacing, cfg.training.batch_size,
                                                      cfg.io.model_path, cfg.experiment_name)
        mlflow.log_param("best_epoch", best_epoch)
        # test
        raw_data_test = RawDataset(cfg.io.input_dir, 'test', cfg.data.max_dist)
        raw_data_test.load_data()
        raw_data_test.set_normalization_params(*raw_data_train.get_normalization_params())
        raw_data_test.charge_normalization()
        print(raw_data_test)

        test_dataset = ProteinLigand_3DDataset(raw_data_test,
                                               grid_spacing=cfg.data.grid_spacing, rotations=None)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 4,
                                     shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        test(best_model, test_dataloader, len(test_dataset))
        affinities, predictions = predict(best_model, test_dataloader, len(test_dataset))
        rmse, mae, corr = analyse(affinities, predictions)
        mlflow.log_metric(f"test_rmse", rmse)
        mlflow.log_metric(f"test_mae", mae)
        mlflow.log_metric(f"test_r", corr[0])
        mlflow.log_metric(f"test_p-value", corr[1])

        print(f"[TEST] rmse: {rmse:.4f} mae: {mae:.4f} corr: {corr}")

        mlflow.pytorch.log_model(best_model, "model")
        save_model(model, cfg.io.model_path, cfg.experiment_name, cfg.mlflow.run_name, rmse)


if __name__ == "__main__":
    my_app()
