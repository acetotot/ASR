import os

class DataConfig:
    def __init__(self):
        self.train_dataset_path = "/data/train_processing_hanzi.cvs"
        self.val_dataset_path = "/data/test_hanzi.cvs"
        self.test_dataset_path = "/data/test-aishell-aitang.cvs"
        self.batch_frames = 6000
        self.LFR = False
        self.LFR_m = 4
        self.LFR_n = 3


class TrainConfig:
    def __init__(self):
        self.lr = 1e-4
        self.save_epoch = 15
        self.saved_model_path = "saved_models"
        self.evts_path = "logevts"
        self.epochs = 1001
        self.log_n_iters = 100
        self.val_n_epochs = 20
        self.n_gpus = [0,1]
        # self.ddp = False
        self.ddp = True


class Config:
    def __init__(self):
        self.data = DataConfig()
        self.train = TrainConfig()

# if __name__ == "__main__":
#     c = ConvTasNetConfig()
#     from torchsummary import summary
#     conv_tasnet = c.model
#     summary(conv_tasnet, tuple([16384]))  # Forward/backward pass size (MB): 1561.02, Params size (MB): 32.98
