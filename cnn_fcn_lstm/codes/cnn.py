import torch
import torch.nn as nn


def conv_layer(in_c, out_c):
    _conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(5, 5, 5), padding='same'),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        # nn.Tanh(),
        nn.MaxPool3d((2, 2, 2)))
    return _conv_layer


def fc_layer(in_c, out_c):
    _fc_layer = nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.LeakyReLU(),
        # nn.Tanh(),
        nn.Dropout(p=0.5)
        # nn.BatchNorm1d(out_c)
    )
    return _fc_layer


class Single_CNN(nn.Module):
    # all frames of a complex will go through a single CNN

    def __init__(self, in_frames, in_channels_per_frame) -> None:
        super().__init__()
        self.in_frames = in_frames
        self.in_channels = in_channels_per_frame

        self.conv1 = conv_layer(in_channels_per_frame, out_c=64)
        self.conv2 = conv_layer(64, 128)
        self.conv3 = conv_layer(128, 256)

        self.fcn1 = fc_layer(256 * 3 * 3 * 3, 256 * 3)
        self.fcn2 = fc_layer(256 * 3, 256)
        self.fcn3 = nn.Sequential(nn.Linear(256, 1))

    def forward(self, x):
        frame_cnn_output = []
        frame_length = x.shape[1]
        for i in range(frame_length):
            f = self.conv1(x[:, i, :])
            f = self.conv2(f)
            f = self.conv3(f)
            f = torch.flatten(f, 1)
            f = self.fcn1(f)
            f = self.fcn2(f)
            f = self.fcn3(f)

            frame_cnn_output.append(f)
        return frame_cnn_output


# NOT OPERATIONAL
class Multiple_CNN(nn.Module):
    # each frames will go through one CNN (total CNN is "in_frames")

    def __init__(self, in_frames, in_channels_per_frame) -> None:
        super().__init__()
        self.in_frames = in_frames
        self.in_channels = in_channels_per_frame

        # create as many cnn as there are frames
        self.cnns = torch.nn.ModuleList([nn.Sequential(
            conv_layer(in_channels_per_frame, out_c=64),
            conv_layer(64, 128),
            conv_layer(128, 256),
            fc_layer(256 * 3 * 3 * 3, 256 * 3),
            fc_layer(256 * 3, 256),
            nn.Linear(256, 1)
        ) for _ in range(self.in_frames)])

        print(f"cnns {self.cnns}")

    def forward(self, x):
        frame_cnn_output = []
        frame_length = x.shape[1]

        # frame_cnn_output = [cnn(x) for x, cnn in zip(x, self.cnns)]
        print(f"x[:, 0, :] {x[:, 0, :].shape}")
        for i in range(frame_length):
            f = self.cnns[i](x[:, i, :])
            print(f"f {f.shape}")
            frame_cnn_output.append(f)

        return frame_cnn_output
