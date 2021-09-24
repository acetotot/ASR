# from model.jasper import Jasper
# from model.espnet_transformer.transformer_conv2d import MyTransformer
from model.espnet_transformer.transformer_linear_sharing import MyTransformer
import torch
import pytorch_lightning as pl
from utils.text_utils import Mapper, max_decode_last_dim, cal_cer_batch, translate_to_string
from utils.feature_utils import int_to_tensor
from dataset.spm_data import AudioDataset
from config import Config
from torch.utils.data import DataLoader
from einops import rearrange
from pytorch_lightning.callbacks import ModelCheckpoint


class LitModel(pl.LightningModule):
    def __init__(self, alpha, lr, elayers, adim, linear_units):
        super().__init__()
        self.loss = torch.nn.CTCLoss()
        # self.mapper = Mapper("dataset/librispeech/librispeech_31.model")
        self.mapper = Mapper("dataset/librispeech/librispeech_1k.model")
        vocab_size = 1000
        self.model = MyTransformer(idim=40*4, odim=vocab_size, adim=adim, elayers=elayers, linear_units=linear_units)
        self.save_hyperparameters()
        self.alpha = alpha
        self.lr = lr

    def forward(self, x_btd, len_b, y_bl):
        d = self.model(x_btd, len_b, y_bl)
        return d

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0005)
        self.opt = opt
        return opt

    def new_lr(self):
        d_model = 512
        step = self.global_step + 1
        warmup_steps = 4000
        lr = 1.0 * d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr
        return lr

    def run_batch(self, batch):
        features_btd = batch["features"].squeeze(0)
        # labels_len = batch["labels_len"].squeeze(0)
        features_len = batch["features_len"].squeeze(0)
        labels = batch["labels"].squeeze(0)
        d = self(features_btd, features_len, labels)
        d["labels"] = labels
        loss_ctc = d["loss_ctc"]
        loss_att = d["loss_att"]
        loss = self.alpha * loss_ctc + (1 - self.alpha) * loss_att
        # if self.alpha < 0.01 or not (loss_ctc < 2000):
        #     loss = loss_att
        # else:
        #     loss = self.alpha * loss_ctc + (1 - self.alpha) * loss_att
        d["loss"] = loss
        return d

    def training_step(self, batch, batch_idx):
        d = self.run_batch(batch)
        loss_ctc = d["loss_ctc"]
        loss_att = d["loss_att"]
        loss = d["loss"]

        result = pl.TrainResult(minimize=loss)
        result.log("loss", loss)
        result.log("loss_ctc", loss_ctc)
        result.log("loss_att", loss_att)
        lr = self.new_lr()
        result.log("lr", lr)

        return result

    def validation_step(self, batch, batch_idx):
        d = self.run_batch(batch)
        logits_bld = d["pred"]
        # labels = d["labels"]
        loss = d["loss"]
        p_label = max_decode_last_dim(logits_bld)
        p_text = []
        for label_len, p in zip(batch["labels_len"].squeeze(0), p_label):
            p_text.append(translate_to_string(p[:label_len]))
        # gt_text = self.mapper.translate_batch_to_strings(labels)
        gt_text = [s[0] for s in batch["texts"]]
        cer = cal_cer_batch(p_text, gt_text)
        cer = int_to_tensor(cer)

        # result = pl.EvalResult(checkpoint_on=cer)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss)
        result.log("cer", cer)
        print("CER", cer)
        logger = self.logger.experiment
        for i in range(1):
            logger.add_text(F"gt_{i}", gt_text[i], self.global_step)
            logger.add_text(F"p_{i}", p_text[i], self.global_step)

        return result

    def train_dataloader(self):
        train_dataset = AudioDataset(cfg.data, cfg.data.train_dataset_path, val=False)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)
        return train_loader

    def val_dataloader(self):
        val_dataset = AudioDataset(cfg.data, cfg.data.val_dataset_path, val=True)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=False)
        return val_loader


if __name__ == "__main__":
    cfg = Config()
    n_gpus = 8
    # model = LitModel(alpha=0.1)
    model = LitModel(alpha=0.1, lr=6e-4, elayers=12, adim=256, linear_units=2048)
    checkpoint_callback = ModelCheckpoint(save_top_k=5)
    # checkpoint = "lightning_logs/version_48/checkpoints/epoch=73.ckpt" # char
    # checkpoint = "lightning_logs/version_26/checkpoints/epoch=107.ckpt"
    # checkpoint = "lightning_logs/version_51/checkpoints/epoch=26.ckpt"
    # model = model.load_from_checkpoint(checkpoint, alpha=0.1, lr=1e-4)
    # debug = True
    debug = False
    if not debug:
        trainer = pl.Trainer(gpus=n_gpus, checkpoint_callback=checkpoint_callback,
                             gradient_clip_val=5,
                             distributed_backend='ddp',)
                             # distributed_backend='ddp',
                             # resume_from_checkpoint=checkpoint)
    else:
        trainer = pl.Trainer(gpus=1, fast_dev_run=True)
    trainer.fit(model)
