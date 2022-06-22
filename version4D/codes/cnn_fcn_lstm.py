import torch
import torch.nn as nn
from torch import device


# from torch.autograd import Variable


class CNN_FCN_LSTM(nn.Module):

    def __init__(self, in_frames, in_channels_per_frame, device) -> None:
        super().__init__()
        self.in_frames = in_frames
        self.in_channels = in_channels_per_frame

        self.conv1 = self.conv_layer(in_channels_per_frame, out_c=64)
        self.conv2 = self.conv_layer(64, 128)
        self.conv3 = self.conv_layer(128, 256)

        self.fcn1 = self.fc_layer(256 * 3 * 3 * 3, 256 * 3)
        self.fcn2 = self.fc_layer(256 * 3, 256)
        self.fcn3 = nn.Sequential(nn.Linear(256, 1))  # TODO try without ?

        # LSTM declaration (input_size depends on lasty fcn)
        self.input_size = in_channels_per_frame  # number of features
        self.hidden_size = 10  # number of features in hidden state
        self.num_layers = 1

        self.lstm = nn.LSTM(input_size=self.in_frames, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        # hidden size is also output_size
        self.dropout = nn.Dropout(0.2)  # TODO parameter
        self.fcn4 = nn.Linear(self.hidden_size, 16)  # after lstm
        self.fcn5 = nn.Linear(16, 1)  # after lstm

        self.device = device

    def forward(self, x):
        # TODO : kept implicit in lstm function (check if ok)
        # h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_() #hidden state
        # c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        # print(f"Model: x len {len(x)} shape x[0] {x[0].shape}")
        # x contains up to 50 frames, each has : [batch_size, 19, x, y, z]
        frame_cnn_output = []
        for f in x:
            f = f.to(self.device)
            f = self.conv1(f)
            f = self.conv2(f)
            f = self.conv3(f)
            f = torch.flatten(f, 1)
            f = self.fcn1(f)
            f = self.fcn2(f)
            f = self.fcn3(f)

            frame_cnn_output.append(f)

        # format into (batch_size, seg_length, features) for LSTM
        frame_cnn_output = torch.cat(frame_cnn_output, dim=1)
        lstm_input = frame_cnn_output.view([frame_cnn_output.shape[0], -1, frame_cnn_output.shape[1]])
        # print(f"all_frame_cnn_output= {len(frame_cnn_output)}, one_frame {frame_cnn_output[0].shape}")
        # print(f"lstm input {lstm_input.shape}")

        lstm_out, (hn, cn) = self.lstm(lstm_input)
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # hn = hn.view(-1, self.hidden_size)
        out = self.dropout(lstm_out)
        out = self.fcn4(out)
        out = self.fcn5(out)

        # out = out.view(batch_size, -1)
        # out = out[:,-1]

        # y = self.lstm(lstm_input)

        # print(f"lstm output {lstm_out.shape}")
        # print(f"Final output: {out.shape}")
        return out.squeeze()

    # COntinuer sur https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
    # plus simple (sur l'init de l'hidden state : https://cnvrg.io/pytorch-lstm/)

    # https://www.kaggle.com/code/orkatz2/cnn-lstm-pytorch-train/notebook

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

    @staticmethod
    def conv_layer(in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(5, 5, 5), padding='same'),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.MaxPool3d((2, 2, 2)))
        return conv_layer

    @staticmethod
    def fc_layer(in_c, out_c):
        fc_layer = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Dropout(p=0.5)
            # nn.BatchNorm1d(out_c)
        )
        return fc_layer

    """
        def forward(self, x, y):
            x1 = self.features(x)
            x2 = self.features(y)
            x = torch.cat((x1, x2), 1)
            return x

    net = torch.cat((x,y,z),1)

    return net

You have to control your parameters while feeding the network. Layers couldn't be feed with more than a parameter. Therefore, you need to extract features from your input one by one and concatenate with torch.cat((x,y),1)(1 for dimension) them.


        voir aussi les inputs d'un lstm
        output = model(input1,input2)

        https://stackoverflow.com/questions/63443348/how-to-train-pytorch-cnn-with-two-or-more-inputs
        https://stackoverflow.com/questions/66786787/pytorch-multiple-branches-of-a-model
        https://stackoverflow.com/questions/65144346/feeding-multiple-inputs-to-lstm-for-time-series-forecasting-using-pytorch
    """
