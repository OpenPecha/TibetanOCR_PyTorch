"""
A collection of OCR architectures to be used by the pipeline.

"""

import torch
from torch import nn
from torch.nn import functional as F


"""
1. A classic implementation of a CRNN model using a CNN backbone with recurrent layers and ctc loss.
"""

class ConvRelu(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel: int = 3, 
                 strides: int = 1, 
                 padding: int = 1, 
                 use_bn: bool = False, 
                 leaky_relu: bool = False):
        super(ConvRelu, self).__init__()
        
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=strides, padding=padding)
        self.use_bn = use_bn
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = leaky_relu
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv2d(x)

        if self.use_bn:
            x = self.bn(x)

        if self.leaky_relu:
            x = nn.LeakyReLU(negative_slope=0.2)(x)
        else:
            x = nn.ReLU(inplace=True)(x)
        
        return x


class VanillaCRNN(nn.Module):

    def __init__(self, img_height: int = 80, img_width: int = 2000, img_channels: int = 1, charset_size: int = 68,
                 map_to_seq_hidden: int =64, rnn_hidden: int =256, leaky_relu: bool = False, rnn: str = "lstm"):
        super(VanillaCRNN, self).__init__()


        self.input_channels = img_channels
        self.input_height = img_height
        self.input_width = img_width
        self.classes = charset_size
        self.map_to_seq_hidden = map_to_seq_hidden

        self.conv_block_0= ConvRelu(in_channels=self.input_channels, out_channels=64, leaky_relu=leaky_relu)
        self.max_pool_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block_1 = ConvRelu(in_channels=64, out_channels=128, leaky_relu=leaky_relu)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block_2 = ConvRelu(in_channels=128, out_channels=256, leaky_relu=leaky_relu)
        self.conv_block_3 = ConvRelu(in_channels=256, out_channels=256, leaky_relu=leaky_relu)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv_block_4= ConvRelu(in_channels=256, out_channels=512, use_bn=True, leaky_relu=leaky_relu)
        self.conv_block_5= ConvRelu(in_channels=512, out_channels=512, use_bn=True, leaky_relu=leaky_relu)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv_block_6 = ConvRelu(in_channels=512, out_channels=512, kernel=2, padding=0,leaky_relu=leaky_relu)
        self.linear = nn.Linear(512 * ( self.input_height // 16 - 1), self.map_to_seq_hidden)

        if rnn == "lstm":
            self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
            self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)
        else:
            self.rnn1 = nn.GRU(map_to_seq_hidden, rnn_hidden, bidirectional=True)
            self.rnn2 = nn.GRU(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, charset_size)


    def forward(self, images):
        x = self.conv_block_0(images)
        x = self.max_pool_0(x)
        x = self.conv_block_1(x)
        x = self.max_pool_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.max_pool_2(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.max_pool_3(x)
        x = self.conv_block_6(x)

        batch, channel, height, width = x.size()
        
        x = x.view(batch, channel * height, width)
        x = x.permute(2, 0, 1)
        x = self.linear(x)

        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.dense(x)

        return x