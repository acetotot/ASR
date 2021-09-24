
from argparse import Namespace
from distutils.util import strtobool

import logging
import math

import torch
import pdb

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from einops import rearrange


class MyTransformer(torch.nn.Module):
    def __init__(self, idim, odim, adim, elayers=6, dlayers=6, ignore_id=-1, linear_units=1024):
        super().__init__()
        self.encoder = Encoder(
            idim=idim,
            attention_dim=adim,
            input_layer="linear",
            linear_units=linear_units,
            num_blocks=elayers,
        )
        self.y_embedding = torch.nn.Embedding(odim, adim)
        self.y_output_prj = torch.nn.Linear(adim, odim)
        # self.y_output_prj.weight = self.y_embedding.weight
        self.decoder = Decoder(
            odim=odim,
            attention_dim=adim,
            linear_units=1024,
            num_blocks=dlayers,
            use_output_layer=False,
            input_layer=self.y_embedding
        )

        self.sos = 1
        self.eos = 2
        self.odim = odim
        self.ignore_id = ignore_id
        self.criterion = LabelSmoothingLoss(odim, ignore_id, smoothing=0.1, normalize_length=True)
        # self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_id)
        # self.criterion = torch.nn.CrossEntropyLoss()

        self.ctc = CTC(odim, adim, ctc_type="bulitin", dropout_rate=0.1, reduce=True)

    def forward(self, x_btd, len_b, y_bl):#训练过程：输入音频特征向量x_btd，批量pad到同一长度后输入到encoder生成隐藏向量h，再encoder 计算ctc损失
        # x_btd: Audio输入特征b*n*d               然后对真是标签加上sos变成y_in_bl 去预测 加上eos的y_out_bl 在decoder错位推理
        # len_b: Length for each audio =n                  并对真实标签的上三角进行mask
        # y_bl: Text
        x_btd = x_btd[:, :max(len_b)]
        src_mask = make_non_pad_mask(len_b.tolist()).unsqueeze(-2).to(x_btd.device)
        h_btd, h_mask = self.encoder(x_btd, src_mask)
        h_len = rearrange(h_mask, "b d l ->(b d) l").sum(1)
        loss_ctc = self.ctc(h_btd, h_len, y_bl)
        # y_bl [1, 2, 3]
        # y_in_bl Decoder Input [SOS, 1, 2, 3]
        # y_out_bl Decoder Output [1, 2, 3, EOS] (训练标签)
        y_in_bl, y_out_bl = add_sos_eos(y_bl, self.sos, self.eos, self.ignore_id)
        y_mask = target_mask(y_in_bl, self.ignore_id)
        pred_bld, pred_mask = self.decoder(y_in_bl, y_mask, h_btd, h_mask)
        pred_bld = self.y_output_prj(pred_bld)  #激活函数

        loss_att = self.criterion(pred_bld, y_out_bl)

        return {"loss_ctc":loss_ctc, "loss_att":loss_att, "pred":pred_bld}

    def inference(self, x_btd, len_b):

        x_btd = x_btd[:, :max(len_b)]

        src_mask = make_non_pad_mask(len_b.tolist()).unsqueeze(-2).to(x_btd.device)
        h_btd, h_mask = self.encoder(x_btd, src_mask)
        h_len = rearrange(h_mask, "b d l ->(b d) l").sum(1)
        y_in_bl = (torch.ones(1, 1) * self.sos).long().to(x_btd.device)
        # Input 从 <SOS> 开始，依次开始得到下一个输出的字符
        for _ in range(500):
            y_mask = target_mask(y_in_bl, self.ignore_id)
            pred_bld, pred_mask = self.decoder(y_in_bl, y_mask, h_btd, h_mask)
            pred_bld = self.y_output_prj(pred_bld)
            pred_bdl = rearrange(pred_bld, "b l d -> b d l")
            val, ix = pred_bdl[:, :, -1].data.topk(1)  #取最后一个字符看是否是eos
            char = ix[0][0]
            if (char == self.eos) or (y_in_bl.shape[1] > 500):
                break
            else:
                char = char.unsqueeze(0).unsqueeze(0)
                y_in_bl = torch.cat((y_in_bl, char), dim=-1)
        return y_in_bl[:, 1:] #扔掉加上的sos eos没有加上，碰到就break了

if __name__ == "__main__":
    model = MyTransformer(idim=40, odim=1442, adim=512)
    # input = torch.rand(4, 200, 40)
    # labels = torch.randint(0, 1000, (4, 20))
    # lens = torch.randint(100, 200, (4,))
    # out = model(input, lens, labels)
    # print(out["pred"].shape)
    print(model)
