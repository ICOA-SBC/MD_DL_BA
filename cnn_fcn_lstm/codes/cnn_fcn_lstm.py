import torch
import torch.nn as nn

from codes.cnn import Single_CNN, Multiple_CNN
from codes.densecnn import create_densenucy


class CNN_FCN_LSTM(nn.Module):

    def __init__(self, in_frames, in_channels_per_frame, model_architecture="single_cnn") -> None:
        super().__init__()
        self.model_name = model_architecture
        self.in_frames = in_frames
        self.in_channels = in_channels_per_frame

        if model_architecture == "single_cnn":
            self.cnn_layers = Single_CNN(in_frames, in_channels_per_frame)
        elif model_architecture == "multi_cnn":  # NOT WORKING
            self.cnn_layers = Multiple_CNN(in_frames, in_channels_per_frame)
        elif model_architecture == "single_densecnn":
            self.cnn_layers = create_densenucy(in_frames)
        else:
            print("ERROR")
            exit()

        # LSTM declaration (input_size depends on last fcn)
        self.input_size = in_channels_per_frame  # number of features
        self.hidden_size = 10  # number of features in hidden state
        self.num_layers = 1

        self.lstm = nn.LSTM(input_size=self.in_frames, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        # hidden size is also output_size
        self.dropout = nn.Dropout(0.2)
        self.fcn4 = nn.Linear(self.hidden_size, 16)  # after lstm
        self.fcn5 = nn.Linear(16, 1)  # after lstm

    def forward(self, x):
        frame_cnn_output = self.cnn_layers(x)
        # format into (batch_size, seg_length, features) for LSTM
        frame_cnn_output = torch.cat(frame_cnn_output, dim=1)
        lstm_input = frame_cnn_output.view([frame_cnn_output.shape[0], -1, frame_cnn_output.shape[1]])

        lstm_out, _ = self.lstm(lstm_input)
        out = self.dropout(lstm_out)
        out = self.fcn4(out)
        out = self.fcn5(out)

        return out.squeeze()
