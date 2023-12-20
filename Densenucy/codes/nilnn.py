import torch
import torch.nn as nn


class NilNN(nn.Module):

    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv1 = self.__conv_layer(in_channels, out_c=64)
        self.conv2 = self.__conv_layer(64, 128)
        self.conv3 = self.__conv_layer(128, 256)
        self.fcn1 = self.__fc_layer(256 * 3 * 3 * 3, 256 * 3)
        self.fcn2 = self.__fc_layer(256 * 3, 256)
        self.fcn3 = nn.Sequential(nn.Linear(256, 1))

        for l in self.modules():
            if isinstance(l, nn.Conv3d):
                nn.init.kaiming_normal_(l.weight,
                                        mode="fan_in",
                                        nonlinearity="leaky_relu")
                nn.init.constant_(l.bias, 0.1)
            elif isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight,
                                        mode="fan_in",
                                        nonlinearity="leaky_relu")
                nn.init.constant_(l.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fcn3(x)
        return x.squeeze()

    def __conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(5, 5, 5), padding='same'),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.MaxPool3d((2, 2, 2)))
        return conv_layer

    def __fc_layer(self, in_c, out_c):
        fc_layer = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Dropout(p=0.5)
            # nn.BatchNorm1d(out_c)
        )
        return fc_layer
