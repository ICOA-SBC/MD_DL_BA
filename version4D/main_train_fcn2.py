import copy
import os
import time

import hydra
import mlflow
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from codes.cnn_fcn2_lstm import CNN_FCN_LSTM
from codes.complex_dataset import Complexes_4DDataset
from codes.tools import convert_byte
from codes.transformations import build_rotations
from main_test import analyse, predict


def convert_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def train(model, dl, ds_size, cfg_train, cfg_exp, device):
    metric = torch.nn.MSELoss(reduction='sum')
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=cfg_train.learning_rate, weight_decay=cfg_train.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


    best_mse, best_epoch = 100, -1
    patience, max_patience = 0, cfg_train.patience
    time_train = time.time()
    for epoch in range(cfg_train.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg_train.num_epochs}")
        time_epoch = time.time()
        for phase in ['train', 'val']:
            print(f"Phase: {phase} ")
            model.train() if phase == 'train' else model.eval()

            running_metrics = 0.0

            for (*inputs, labels) in dl[phase]:
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels.float())
                    # statistics
                    running_metrics += metric(outputs, labels).cpu().item()

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            epoch_metric = running_metrics / ds_size[phase]
            if phase == 'train':
                print(f"\t[{phase}] MSELoss {epoch_metric:.4f}")
            else:
                print(f"\t[{phase}] MSELoss {epoch_metric:.4f} \t Duration: {time.time() - time_epoch:.2f}")
                # Learning rate scheduling should be applied after optimizer’s update (pytorch doc)
                scheduler.step(loss)

            # deep copy the model
            if phase == 'val' and epoch_metric < best_mse:
                print(f"/\\ Better loss {best_mse} --> {epoch_metric}")
                best_mse, best_epoch = epoch_metric, epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                filename = os.path.join(cfg_exp.model_path, f"{cfg_exp.model_name}_{best_mse:.4f}_{best_epoch}.pth")
                print(f"\tsaving model {filename}")
                torch.save(model, filename)
                patience = 0
            else:
                patience += 1

            mlflow.log_metric(f"{phase}_mse", epoch_metric, epoch)

        if patience > max_patience:
            print("----------- Early stopping activated !")
            break
    duration = time.time() - time_train
    print(f"[{epoch + 1} / {cfg_train.num_epochs}] Best mean MSE: {best_mse:.4f} at epoch {best_epoch} \
       \n\tTotal duration: {convert_time(duration)}")

    model.load_state_dict(best_model_wts)

    return model, best_epoch


def test(model, test_dataloader, test_samples, device):
    metric = torch.nn.MSELoss(reduction='sum')

    model.eval()
    running_metrics = 0.0

    for (*inputs, labels) in test_dataloader:
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            running_metrics += metric(outputs, labels).cpu().item()

    metric = running_metrics / test_samples
    print(f"\t[Test] MSELoss {metric:.4f}")
    mlflow.log_metric(f"test_mse", metric)


def save_model(model, model_path, model_name):
    filename = os.path.join(model_path, f"{model_name}.pth")
    print(f"\tsaving model {filename}")
    torch.save(model, filename)


def mlflow_setup(cfg):
    mlflow.log_param("learning_rate", cfg.training.learning_rate)
    mlflow.log_param("patience", cfg.training.patience)
    mlflow.log_param("max_epoch", cfg.training.num_epochs)
    # log configuration as an artefact
    with open('./configuration.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    mlflow.log_artifact("configuration.yaml")


@hydra.main(config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    by_complex = cfg.experiment.by_complex

    # create transformations
    rotations_matrices = build_rotations()
    print(f"Number of available rotations: {len(rotations_matrices)}")

    train_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex,
                                   rotations_matrices, mode="train", debug=cfg.debug)
    val_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex, mode="val", debug=cfg.debug)

    ds_size = {'train': len(train_ds), 'val': len(val_ds)}
    batch_size = cfg.training.batch_size

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    dl = {'train': train_dl, 'val': val_dl}

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN_FCN_LSTM(in_frames=cfg.data_setup.frames,
                         in_channels_per_frame=cfg.data_setup.features, device=dev)
    model.to(dev)

    try:
        mlflow.end_run()
    except:
        print("mlflow not running")

    mlflow.set_tracking_uri(cfg.experiment.path)
    mlflow.set_experiment(cfg.experiment.name)

    with mlflow.start_run(run_name=cfg.experiment.run) as run:
        mlflow_setup(cfg)

        # train
        best_model, best_epoch = train(model, dl, ds_size, cfg.training, cfg.experiment, dev)

        save_model(model, cfg.experiment.model_path, cfg.experiment.model_name)

        # test
        test_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex, mode="test", debug=cfg.debug)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                             persistent_workers=True)
        test(best_model, test_dl, len(test_ds), dev)

        affinities, predictions = predict(best_model, test_dl, len(test_ds), dev)
        rmse, mae, corr = analyse(affinities, predictions)

        mlflow.log_param("best_epoch", best_epoch)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r", corr[0])

        mlflow.pytorch.log_model(best_model, "model")

    print(f"GPU usage: {convert_byte(torch.cuda.max_memory_allocated(device=None))}")


if __name__ == "__main__":
    main()

# TODO
# - test by_complexe=false
# - calculer rmse, mae et corr (voir proli_test) sur val ?
# - pb pour ddp : envoi des frames dans gpu durant le forward !

"""
a voir    
https://cnvrg.io/pytorch-lstm/
https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
https://towardsdatascience.com/pytorch-lstms-for-time-series-data-cd16190929d7
https://stackoverflow.com/questions/65596522/lstm-for-time-series-prediction-failing-to-learn-pytorch
https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
https://stackoverflow.com/questions/65144346/feeding-multiple-inputs-to-lstm-for-time-series-forecasting-using-pytorch
"""
