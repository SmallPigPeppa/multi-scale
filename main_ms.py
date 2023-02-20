import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from data_modules.imagenet_dali import DALIDataset
from models.msnet_l1 import MultiScaleNet
from args import parse_args


class MSNetPL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_threads = args.num_threads
        self.num_gpus = args.num_gpus
        self.lr = args.lr
        self.args = args
        self.encoder = MultiScaleNet()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.dali_dataset = DALIDataset(data_dir=args.data_dir,
                                        batch_size=args.batch_size * self.num_gpus,
                                        num_threads=args.num_threads,
                                        num_gpus=args.num_gpus)
        self.dali_dataset.setup()

    def forward(self, x):
        z1, z2, z3, y1, y2, y3 = self.encoder(x)
        return z1, z2, z3, y1, y2, y3


    def share_step(self, batch, batch_idx):
        x, target = batch
        z1, z2, z3, y1, y2, y3 = self.forward(x)


        si_loss1 = self.mse_loss(z1, z2)
        si_loss2 = self.mse_loss(z1, z3)
        si_loss3 = self.mse_loss(z2, z3)
        ce_loss1 = self.ce_loss(y1, target)
        ce_loss2 = self.ce_loss(y2, target)
        ce_loss3 = self.ce_loss(y3, target)
        total_loss = si_loss1 + si_loss2 + si_loss3 + ce_loss1 + ce_loss2 + ce_loss3

        acc1 = (torch.argmax(y1, dim=1) == target).float().mean()
        acc2 = (torch.argmax(y2, dim=1) == target).float().mean()
        acc3 = (torch.argmax(y3, dim=1) == target).float().mean()
        avg_acc = torch.mean([acc1, acc2, acc3])

        result_dict = {
            "si_loss1": si_loss1,
            "si_loss2": si_loss2,
            "si_loss3": si_loss3,
            "ce_loss1": ce_loss1,
            "ce_loss2": ce_loss2,
            "ce_loss3": ce_loss3,
            "total_loss": total_loss,
            "acc1": acc1,
            "acc2": acc2,
            "acc3": acc3,
            "avg_acc": avg_acc
        }
        return result_dict

    def training_step(self, batch, batch_idx):
        result_dict = self.share_step(batch, batch_idx)
        train_result_dict = {f'train_{k}': v for k, v in result_dict.items()}
        self.log_dict(train_result_dict)
        return result_dict['total_loss']

    def validation_step(self, batch, batch_idx):
        result_dict = self.share_step(batch, batch_idx)
        val_result_dict = {f'val_{k}': v for k, v in result_dict.items()}
        self.log_dict(val_result_dict)
        return val_result_dict

        # def validation_epoch_end(self, outputs):
        #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #     avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        #     tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        #     return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),
                              lr=self.lr * self.num_gpus,
                              momentum=0.9,
                              weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dali_dataset.train_dataloader()

    def val_dataloader(self):
        return self.dali_dataset.val_dataloader()


if __name__ == '__main__':
    args = parse_args()
    model = MSNetPL(args)
    wandb_logger = WandbLogger(name=args.name, project=args.project, entity=args.entity, offline=args.offline)
    wandb_logger.watch(model, log="gradients", log_freq=100)
    wandb_logger.log_hyperparams(args)
    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir, save_last=True, save_top_k=2, monitor="val_total_loss")
    trainer = pl.Trainer(gpus=args.num_gpus,
                         max_epochs=args.max_epochs,
                         check_val_every_n_epoch=5,
                         gradient_clip_val=0.5,
                         strategy='ddp',
                         logger=wandb_logger,
                         callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback])

    trainer.fit(model)
