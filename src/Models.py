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
                 dropout_rate: float = 0.2, 
                 leaky_relu: bool = False):
        super(ConvRelu, self).__init__()
        
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=strides, padding=padding)
        self.use_bn = use_bn
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = leaky_relu
        self.dropout = nn.Dropout(p=dropout_rate)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv2d(x)

        if self.use_bn:
            x = self.bn(x)

        if self.leaky_relu:
            x = nn.LeakyReLU(negative_slope=0.2)(x)
        else:
            x = nn.ReLU(inplace=True)(x)
        x = self.dropout(x)

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


"""
2. Experimental Implementation of the Easter2 Architecture
    -   Adaption of the original Easter2-Architecture: https://github.com/kartikgill/Easter2
    -   On the padding behaviour of the Conv-layers in Pytorch, see here: https://github.com/pytorch/pytorch/issues/67551. I opted for manually padding the output.
"""


class GlobalContext(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalContext, self).__init__()
        self.pool = nn.AvgPool1d(kernel_size=out_channels)
        self.linear1 = nn.Linear(in_channels, out_channels // 8)
        self.linear2 = nn.Linear(out_channels // 8, out_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        pool = self.pool(data)
        pool = pool[:, :, 0] # TODO: maybe not just drop the last dimension, but average the values across the depth dim?
        pool = self.linear1(pool)
        pool = self.relu(pool)
        pool = self.linear2(pool)
        pool = self.sigmoid(pool)
        pool = torch.unsqueeze(pool, -1)
        pool = torch.multiply(pool, data)

        return pool


class EasterUnit(nn.Module):
    """
    TODO: the padding value calculation can be done on init of the EasterUnit instead of the forward pass
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, bn_eps=1e-5, bn_decay=0.997):
        super(EasterUnit, self).__init__()
        self.bn_eps = bn_eps
        self.bn_decay = bn_decay
        self.dropout = dropout

        self.conv1d_1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, groups=1)
        self.conv1d_2 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, groups=1)
        self.conv1d_3 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, groups=1)
        self.conv1d_4 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel, stride=stride, groups=1)
        self.conv1d_5 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel, stride=stride, groups=1)

        self.bn_1 = nn.BatchNorm1d(num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay)
        self.bn_2 = nn.BatchNorm1d(num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay)
        self.bn_3 = nn.BatchNorm1d(num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay)
        self.bn_4 = nn.BatchNorm1d(num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay)
        self.bn_5 = nn.BatchNorm1d(num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay)

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu_3 = nn.ReLU()

        self.drop_1 = nn.Dropout(p=self.dropout)
        self.drop_2 = nn.Dropout(p=self.dropout)
        self.drop_3 = nn.Dropout(p=self.dropout)
        self.global_context = GlobalContext(in_channels=out_channels, out_channels=out_channels)

    def forward(self, input):
        old, data = input
        old = torch.squeeze(old)
        old = self.conv1d_1(old)
        old = self.bn_1(old)

        this = self.conv1d_2(data)
        this = self.bn_2(this)
        old = torch.add(old, this)

        # First Block
        data = self.conv1d_3(data)
        pad_val = old.shape[-1] - data.shape[-1]
        data = nn.ZeroPad1d(padding=(0, pad_val))(data)
        data = self.bn_3(data)
        data = torch.squeeze(data)
        data = self.relu_1(data)
        data = self.drop_1(data)

        # Second Block
        data = self.conv1d_4(data)
        pad_val = old.shape[-1] - data.shape[-1]
        data = nn.ZeroPad1d(padding=(0, pad_val))(data)
        data = self.bn_4(data)
        data = torch.squeeze(data)
        data = self.relu_2(data)
        data = self.drop_2(data)

        # Third Block
        data = self.conv1d_5(data)
        pad_val = old.shape[-1] - data.shape[-1]
        data = nn.ZeroPad1d(padding=(0, pad_val))(data)
        data = self.bn_5(data)
        data = torch.squeeze(data)
        data = self.global_context(data)
        data = torch.add(old, data)
        data = self.relu_3(data)
        data = self.drop_3(data)

        return data, old


class Easter2(nn.Module):
    """
    Easter2 model adapted from the original Keras implementation (see link abvove).
    Note: This is an early experimental version that needs some optimization and better ways to parameterize the model from the cli.

    """

    def __init__(self, input_width: int = 2000, input_height: int = 80, bn_eps=1e-5, bn_decay=0.997,
                 vocab_size: int = 77):
        super(Easter2, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.vocab_size = vocab_size
        self.bn_eps = bn_eps
        self.bn_decay = bn_decay
        self.zero_pad = nn.ZeroPad1d(padding=(0, 1))
        self.dropout = 0.2

        self.conv1d_1 = nn.Conv1d(self.input_height, 128, kernel_size=3, stride=2,
                                  groups=1)  # NxCxL (N=BatchSize, C = number of channels, L = length of the signal)
        self.conv1d_2 = nn.Conv1d(128, 128, kernel_size=3, stride=2, groups=1)
        self.conv1d_3 = nn.Conv1d(256, 512, kernel_size=11, stride=1, dilation=2)
        self.conv1d_4 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding="same")
        self.conv1d_5 = nn.Conv1d(512, self.vocab_size, kernel_size=1, stride=1, padding="same")

        self.bn_1 = nn.BatchNorm1d(num_features=128, eps=self.bn_eps, momentum=self.bn_decay)
        self.bn_2 = nn.BatchNorm1d(num_features=128, eps=self.bn_eps, momentum=self.bn_decay)
        self.bn_3 = nn.BatchNorm1d(num_features=512, eps=self.bn_eps, momentum=self.bn_decay)
        self.bn_4 = nn.BatchNorm1d(num_features=512, eps=self.bn_eps, momentum=self.bn_decay)

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu_3 = nn.ReLU()
        self.relu_4 = nn.ReLU()

        self.easter1 = EasterUnit(128, 128, 5, 1, 0.2)
        self.easter2 = EasterUnit(128, 256, 5, 1, 0.2)
        self.easter3 = EasterUnit(256, 256, 7, 1, 0.2)
        self.easter4 = EasterUnit(256, 256, 9, 1, 0.3)

        self.drop_1 = nn.Dropout(p=self.dropout)
        self.drop_2 = nn.Dropout(p=self.dropout)
        self.drop_3 = nn.Dropout(p=0.4)
        self.drop_4 = nn.Dropout(p=0.4)

        #self.soft_max = nn.Softmax(dim=0)

    def forward(self, inputs):
        """
        Note: The model inputs should correspond to BxHxW (B = batch size, H = image_height = self.input_height, W = image width = self.image_width) to match the specification of NxCxL of nn.conv1d
        TODO: handle the case of a single image input, which currently crashes the network
        """
        x = self.conv1d_1(inputs)
        x = self.zero_pad(x)
        x = self.bn_1(x)
        x = torch.squeeze(x)
        x = self.relu_1(x)
        x = self.drop_1(x)

        x = self.conv1d_2(x)
        x = self.zero_pad(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.drop_2(x)
        x = torch.squeeze(x)

        old = x
        data, old = self.easter1((old, x))
        data, old = self.easter2((old, data))
        data, old = self.easter3((old, data))
        data, old = self.easter4((old, data))

        x = self.conv1d_3(data)

        x = nn.ZeroPad1d(padding=(10, 10))(x)
        x = self.bn_3(x)
        x = self.relu_3(x)
        x = self.drop_3(x)

        x = self.conv1d_4(x)
        x = self.bn_4(x)
        x = self.relu_4(x)
        x = self.drop_4(x)
        x = self.conv1d_5(x)
        x = torch.squeeze(x)

        return x