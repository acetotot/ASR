# from model.jasper import Jasper
# from model.espnet_transformer.transformer_conv2d import MyTransformer
import torchaudio
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from torchaudio.compliance.kaldi import fbank as compute_fbank
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
from conformer import Conformer


class LitModel(pl.LightningModule):
    def __init__(self, alpha, lr):
        super().__init__()
        self.loss = torch.nn.CTCLoss()
        vocab_size = 1442
        # vocab_size = 6844
        self.model = MyTransformer(idim=40, odim=vocab_size, adim=512) #40是输入语音特征维度，odim是输出分类拼音数
        # self.model = Conformer(num_classes=vocab_size, input_dim=40,
        #                        encoder_dim=144, num_encoder_layers=16,
        #                        decoder_dim=320)
        self.save_hyperparameters()
        self.alpha = alpha
        self.lr = lr

    def forward(self, x_btd, len_b, y_bl):
        d = self.model(x_btd, len_b, y_bl)
        return d

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0005)
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
        d["loss"] = loss
        # d["loss"] = loss_att
        return d

    def training_step(self, batch, batch_idx):
        d = self.run_batch(batch)
        loss_ctc = d["loss_ctc"]
        loss_att = d["loss_att"]
        loss = d["loss"]
        self.log("loss", loss)
        self.log("loss_ctc", loss_ctc)
        self.log("loss_att", loss_att)
        lr = self.new_lr()
        self.log("lr", lr)
        return loss

    def list_to_string(self, l):
        if type(l[0]) is str:
            return " ".join(map(str, l))
        else:
            l = [word[0] for word in l]
            return " ".join(map(str, l)) #否则可能是[[1,2,3]]这种元组类型的，就先取出来aa再拼接

    def test_step(self, batch, batch_idx):
        features_btd = batch["features"].squeeze(0)
        features_len = batch["features_len"].squeeze(0)
        labels = batch["labels"].squeeze(0)
        p_labels = []
        for x_1td, len_1 in zip(features_btd, features_len):
            y_bl = self.model.inference(x_1td.unsqueeze(0), len_1.unsqueeze(0)) #与forword维度一致 批量*n时长*d特征长度
            p_labels.append(y_bl)
        p_text = []
        # for label_len, p in zip(batch["labels_len"].squeeze(0), p_labels):
        #     p_string = self.list_to_string(translate_to_string(p[0][:label_len])) #P=[[1,2,3]],先选p[0]=[1,2，3]，再选标签长度，从0~len（lable）
        #     p_text.append(p_string)
        for p in p_labels:
            p = p.cpu().numpy().tolist()
            p_string = self.list_to_string(translate_to_string(p[0])) #p=[[1,2,3]] p[o]=[1,2,3]
            p_text.append(p_string)
        texts = batch["texts"]
        gt_text = [self.list_to_string(s) for s in texts]

        out_file = open("text_output-2.txt", "a")
        for gt, p in zip(gt_text, p_text):
            print("GT/Predicted:", gt, "/", p)
            out_file.write(f"GT/Pred: {gt}/{p}\n")
        cer = cal_cer_batch(p_text, gt_text)
        cer = int_to_tensor(cer)
        self.log("cer", cer)


    def validation_step(self, batch, batch_idx):
        d = self.run_batch(batch)
        logits_bld = d["pred"]
        # labels = d["labels"]
        loss = d["loss"]
        p_label = max_decode_last_dim(logits_bld)
        p_text = []
        for label_len, p in zip(batch["labels_len"].squeeze(0), p_label):
            p_string = self.list_to_string(translate_to_string(p[:label_len]))
            p_text.append(p_string)
        # gt_text = self.mapper.translate_batch_to_strings(labels)
        gt_text = [self.list_to_string(s) for s in batch["texts"]]
        cer = cal_cer_batch(p_text, gt_text)
        cer = int_to_tensor(cer)
        self.log("val_loss", loss)
        self.log("cer", cer)
        print("CER", cer)
        logger = self.logger.experiment
        for i in range(1):
            logger.add_text(F"gt_{i}", gt_text[i], self.global_step)
            logger.add_text(F"p_{i}", p_text[i], self.global_step)


    def train_dataloader(self):
        train_dataset = AudioDataset(cfg.data, cfg.data.train_dataset_path, val=False)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)
        return train_loader

    def val_dataloader(self):
        val_dataset = AudioDataset(cfg.data, cfg.data.val_dataset_path, val=True)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=False)
        return val_loader

    def test_dataloader(self):
        test_dataset = AudioDataset(cfg.data, cfg.data.test_dataset_path, val=True)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, drop_last=False)
        return test_loader




if __name__ == "__main__":
    cfg = Config()
    n_gpus = [1]
    model = LitModel(alpha=0.1, lr=6e-4)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="/data2/lzl/project/speechz/model_sppech/check_point/hanzi/",
        filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    # checkpoint = None
    checkpoint = "/data2/lzl/project/speech-xiuz/model_speech/check_point/sample-mnist-epoch=48-val_loss=0.28.ckpt"
    if checkpoint is not None:
        model = model.load_from_checkpoint(checkpoint)

    # debug = True
    debug = False          #改debug模式就直接在pycharmdebug
    if not debug:
        trainer = pl.Trainer(gpus=n_gpus, callbacks=[checkpoint_callback],
                             limit_train_batches=10000,  # 每20000个batch验证保存一次防止NaN
                             distributed_backend='ddp',)
    else:
        trainer = pl.Trainer(gpus=[1], fast_dev_run=True)
    # trainer.fit(model)
    trainer.test(model)

