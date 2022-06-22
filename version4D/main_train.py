import time

import hydra
import torch.optim as optim
from omegaconf import DictConfig
from torch import nn, device, cuda, set_grad_enabled
from torch.utils.data import DataLoader

from codes.cnn_fcn_lstm import CNN_FCN_LSTM
from codes.complex_dataset import Complexes_4DDataset
from codes.tools import convert_byte
from codes.transformations import build_rotations


def convert_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def train(model, dl, ds_size, cfg, device):
    metric = nn.MSELoss(reduction='sum')
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_mse, best_epoch = 100, -1
    patience, max_patience = 0, cfg.patience
    time_train = time.time()
    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}")
        time_epoch = time.time()
        for phase in ['train', 'val']:
            print(f"Phase: {phase} ")
            model.train() if phase == 'train' else model.eval()

            running_metrics = 0.0

            for (*inputs, labels) in dl[phase]:
                labels = labels.to(device)
                optimizer.zero_grad()

                with set_grad_enabled(phase == 'train'):
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

            # deep copy the model
            if phase == 'val' and epoch_metric < best_mse:
                print(f"/\\ Better loss {best_mse} --> {epoch_metric}")
                best_mse, best_epoch = epoch_metric, epoch
                # best_model_wts = copy.deepcopy(model.state_dict())
                # filename = os.path.join(model_path, f"{name}_{best_MSE:.4f}_{best_epoch}.pth")
                # print(f"\tsaving model {filename}")
                # torch.save(model, filename)
                patience = 0
            else:
                patience += 1

        if patience > max_patience:
            print("----------- Early stopping activated !")
            break
    duration = time.time() - time_train
    print(f"[{epoch + 1} / {cfg.num_epochs}] Best mean MSE: {best_mse:.4f} at epoch {best_epoch} \
       \n\tTotal duration: {convert_time(duration)}")

    return None, None


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

    dev = device("cuda:0" if cuda.is_available() else "cpu")
    model = CNN_FCN_LSTM(in_frames=cfg.data_setup.frames,
                         in_channels_per_frame=cfg.data_setup.features, device=dev)
    model.to(dev)

    # train
    best_model, best_epoch = train(model, dl, ds_size, cfg.training, dev)

    print(f"GPU usage: {convert_byte(cuda.max_memory_allocated(device=None))}")


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
