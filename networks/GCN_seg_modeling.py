# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import torch
import torch.nn as nn
from torch.nn import Conv2d


from .GCN_seg_modeling_resnet_skip import ResNetV2
from .GCN import CTRGCN

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.hybrid = None
        self.config = config
        self.scale = 4


        self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)

        self.patch_embeddings4 = Conv2d(in_channels=1024,
                                       out_channels=1024//4,
                                       kernel_size=(1,1),
                                       stride=(1,1))



    def forward(self, x):

        x, _features = self.hybrid_model(x)

        x = self.patch_embeddings4(x)

        return x, _features



class GCN_bridge(nn.Module):
    def __init__(self, config):
        super(GCN_bridge, self).__init__()
        self.Encoder = Encoder(config)
        self.Bridge4 = CTRGCN(length=14, in_channels=256, adaptive=True,layer=3)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)



    def forward(self, input_ids):
        encoder_output, _features = self.Encoder(input_ids)

        x = self.Bridge4(encoder_output)

        return x, _features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        x = self.conv_more(hidden_states)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class Model(nn.Module):
    def __init__(self, config, num_classes=21843, zero_head=False):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.GCN_bridge = GCN_bridge(config)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config


    def forward(self, x):

        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, features = self.GCN_bridge(x)

        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

if __name__ == '__main__':
    import ml_collections
    import argparse

    def get_config():
        config = ml_collections.ConfigDict()
        config.hidden_size = 256
        config.representation_size = None
        config.resnet_pretrained_path = None
        config.resnet = ml_collections.ConfigDict()
        config.resnet.num_layers = (3, 4, 9)
        config.resnet.width_factor = 1
        config.classifier = 'seg'
        config.decoder_channels = (256, 128, 64, 16)
        config.skip_channels = [512, 256, 64, 16]
        config.n_classes = 9
        config.n_skip = 3
        config.activation = 'softmax'

        return config



    config = get_config()

    net = Model(config, num_classes=config.n_classes).cuda()
    x = torch.rand(1, 1, 224, 224).cuda()
    out = net(x)
    print(out.shape)







