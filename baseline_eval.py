import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from models.baseline_net import BaselineNet
from data_modules.imagenet_dali import ClassificationDALIDataModule
from pytorch_lightning.strategies.ddp import DDPStrategy
from args import parse_args


class BaselineNetPL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.num_gpus = args.num_gpus
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.args = args
        self.encoder = BaselineNet()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        y1, y2, y3 = self.encoder(x)
        return y1, y2, y3

    def share_step(self, batch, batch_idx):
        x, target = batch
        y1, y2, y3 = self.forward(x)
        ce_loss1 = self.ce_loss(y1, target)
        ce_loss2 = self.ce_loss(y2, target)
        ce_loss3 = self.ce_loss(y3, target)
        total_loss = ce_loss1 + ce_loss2 + ce_loss3

        acc1 = (torch.argmax(y1, dim=1) == target).float().mean()
        acc2 = (torch.argmax(y2, dim=1) == target).float().mean()
        acc3 = (torch.argmax(y3, dim=1) == target).float().mean()
        avg_acc = (acc1 + acc2 + acc3) / 3

        result_dict = {
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
        scale_factor = self.batch_size * self.num_gpus  / 256
        lr = self.lr * scale_factor
        optimizer = optim.SGD(self.parameters(),
                              lr=lr,
                              momentum=0.9,
                              weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]










if __name__ == '__main__':
    pl.seed_everything(5)
    args = parse_args()
    model = BaselineNetPL(args)
    wandb_logger = WandbLogger(name=args.name, project=args.project, entity=args.entity, offline=args.offline)
    wandb_logger.watch(model, log="gradients", log_freq=100)
    wandb_logger.log_hyperparams(args)
    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir, save_last=True, save_top_k=2, monitor="val_total_loss")
    trainer = pl.Trainer(gpus=args.num_gpus,
                         max_epochs=args.max_epochs,
                         check_val_every_n_epoch=5,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         precision=16,
                         logger=wandb_logger,
                         callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback])

    # dali_datamodule = DALIDataset(data_dir=args.data_dir, batch_size=args.batch_size * args.num_gpus,
    #                               num_threads=args.num_threads)
    dali_datamodule = ClassificationDALIDataModule(
        train_data_path=os.path.join(args.data_dir,'train'),
        val_data_path=os.path.join(args.data_dir,'val'),
        num_workers=args.num_workers,
        batch_size=args.batch_size)

    # trainer.validate(model, datamodule=dali_datamodule)
    trainer.validate(model, datamodule=dali_datamodule)