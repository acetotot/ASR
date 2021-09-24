"""QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions
https://arxiv.org/abs/1910.10261
"""

__author__ = 'Erdene-Ochir Tuguldur'
__all__ = ['QuartzNet5x5', 'QuartzNet10x5', 'QuartzNet15x5']

from .jasper_encoder_decoder import JasperEncoderDecoder

_quartznet15x5_config = [
    {'filters': 256, 'repeat': 1, 'kernel': [33], 'stride': [2], 'dilation': [1], 'residual': False, 'separable': True},

    {'filters': 256, 'repeat': 5, 'kernel': [33], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 256, 'repeat': 5, 'kernel': [33], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 256, 'repeat': 5, 'kernel': [33], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 256, 'repeat': 5, 'kernel': [39], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 256, 'repeat': 5, 'kernel': [39], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 256, 'repeat': 5, 'kernel': [39], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': [51], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 512, 'repeat': 5, 'kernel': [51], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 512, 'repeat': 5, 'kernel': [51], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': [63], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 512, 'repeat': 5, 'kernel': [63], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 512, 'repeat': 5, 'kernel': [63], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': [75], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 512, 'repeat': 5, 'kernel': [75], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 512, 'repeat': 5, 'kernel': [75], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 1, 'kernel': [87], 'stride': [1], 'dilation': [2], 'residual': False, 'separable': True},
    {'filters': 1024, 'repeat': 1, 'kernel': [1], 'stride': [1], 'dilation': [1], 'residual': False, 'separable': False}
]

_quartznet10x5_config = [
    {'filters': 256, 'repeat': 1, 'kernel': [33], 'stride': [2], 'dilation': [1], 'residual': False, 'separable': True},

    {'filters': 256, 'repeat': 5, 'kernel': [33], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 256, 'repeat': 5, 'kernel': [33], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 256, 'repeat': 5, 'kernel': [39], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 256, 'repeat': 5, 'kernel': [39], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': [51], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 512, 'repeat': 5, 'kernel': [51], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': [63], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 512, 'repeat': 5, 'kernel': [63], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': [75], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},
    {'filters': 512, 'repeat': 5, 'kernel': [75], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 1, 'kernel': [87], 'stride': [1], 'dilation': [2], 'residual': False, 'separable': True},
    {'filters': 1024, 'repeat': 1, 'kernel': [1], 'stride': [1], 'dilation': [1], 'residual': False, 'separable': False}
]

_quartznet5x5_config = [
    {'filters': 256, 'repeat': 1, 'kernel': [33], 'stride': [2], 'dilation': [1], 'residual': False, 'separable': True},

    {'filters': 256, 'repeat': 5, 'kernel': [33], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 256, 'repeat': 5, 'kernel': [39], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': [51], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': [63], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': [75], 'stride': [1], 'dilation': [1], 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 1, 'kernel': [87], 'stride': [1], 'dilation': [2], 'residual': False, 'separable': True},
    {'filters': 1024, 'repeat': 1, 'kernel': [1], 'stride': [1], 'dilation': [1], 'residual': False, 'separable': False}
]


class QuartzNet15x5(JasperEncoderDecoder):

    def __init__(self, vocab, num_features):
        if type(vocab) == int:
            num_classes = vocab
        else:
            num_classes = len(vocab)
        super(QuartzNet15x5, self).__init__(feat_in=num_features, num_classes=num_classes, activation='relu',
                                            jasper=_quartznet15x5_config)


class QuartzNet10x5(JasperEncoderDecoder):

    def __init__(self, vocab, num_features):
        if type(vocab) == int:
            num_classes = vocab
        else:
            num_classes = len(vocab)
        super(QuartzNet10x5, self).__init__(feat_in=num_features, num_classes=num_classes, activation='relu',
                                            jasper=_quartznet10x5_config)


class QuartzNet5x5(JasperEncoderDecoder):

    def __init__(self, vocab, num_features):
        if type(vocab) == int:
            num_classes = vocab
        else:
            num_classes = len(vocab)
        super(QuartzNet5x5, self).__init__(feat_in=num_features, num_classes=num_classes, activation='relu',
                                           jasper=_quartznet5x5_config)
