import time

import hydra
from omegaconf import DictConfig
from torch import device, cuda
from torch.utils.data import DataLoader

from codes.cnn_fcn_lstm import CNN_FCN_LSTM
from codes.complex_dataset import Complexes_4DDataset
from codes.tools import convert_byte
from codes.transformations import build_rotations


def convert_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


@hydra.main(config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    by_complex = cfg.experiment.by_complex

    rotations_matrices = build_rotations()
    print(f"Number of available rotations: {len(rotations_matrices)}")

    test_ds = Complexes_4DDataset(cfg.io, cfg.data_setup, by_complex, rotations_matrices, mode="test", debug=cfg.debug)

    *inputs, affinity = test_ds[4]
    print(f"single sample: aff={affinity} no_of_frames= {len(inputs)} frame_shape={inputs[0].shape}")

    batch_size = 4  # cfg.training.batch_size

    test_dl = DataLoader(test_ds, batch_size=batch_size,
                         shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    *batch, affinity = next(iter(test_dl))

    print(f"one batch (batch_size= {batch_size}) -------------\n \
                len= {len(batch)}, shape_batch[0]= {batch[0].shape} len_label={affinity.shape}")

    dev = device("cuda:0" if cuda.is_available() else "cpu")
    model = CNN_FCN_LSTM(in_frames=cfg.data_setup.frames,
                         in_channels_per_frame=cfg.data_setup.features, device=dev)

    # print(model)

    model.to(dev)
    pred = model(batch)

    print(f"Model output: {pred}")

    print(
        f"GPU usage for batch size= {batch_size} : {convert_byte(cuda.max_memory_allocated(device=None))} - prediction")
    # GPU usage for batch size= 3 : 2.6GiB - prediction


if __name__ == "__main__":
    main()

    """
    bash-4.4$ python main_test_dataset.py 
test_ds[4](unpacked) : aff=6.17 no_of_frames= 50 frame_shape=torch.Size([19, 25, 25, 25])
test_ds[4](packed) : 51 (x inputs + 1 label: 6.17)
batch --------------------------
len_batch= 50, shape_batch[0]= torch.Size([3, 19, 25, 25, 25]) len_label=torch.Size([3])
Model: x len 50 shape x[0] torch.Size([3, 19, 25, 25, 25])
all_frame_cnn_output= 50, one_frame torch.Size([3, 1])



    """
