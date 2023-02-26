import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from models.msnet_l2 import MultiScaleNet
from data_modules.imagenet_dali import ClassificationDALIDataModule
from data_modules.not_dali import prepare_data
from pytorch_lightning.strategies.ddp import DDPStrategy
from args import parse_args
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class MSNetPL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = MultiScaleNet()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

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

        if si_loss1 < 0.01:
            si_loss1 = 0

        if si_loss2 < 0.01:
            si_loss2 = 0

        if si_loss3 < 0.01:
            si_loss3 = 0

        total_loss = si_loss1 + si_loss2 + si_loss3 + ce_loss1 + ce_loss2 + ce_loss3

        acc1 = (torch.argmax(y1, dim=1) == target).float().mean()
        acc2 = (torch.argmax(y2, dim=1) == target).float().mean()
        acc3 = (torch.argmax(y3, dim=1) == target).float().mean()
        avg_acc = (acc1 + acc2 + acc3) / 3

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
        return result_dict['ce_loss1']+result_dict['ce_loss2']+result_dict['ce_loss3']

    def validation_step(self, batch, batch_idx):
        result_dict = self.share_step(batch, batch_idx)
        val_result_dict = {f'val_{k}': v for k, v in result_dict.items()}
        self.log_dict(val_result_dict)
        return val_result_dict

    def configure_optimizers(self):
        scale_factor = self.args.batch_size * self.args.num_gpus / 256
        lr = self.args.lr * scale_factor
        wd = self.args.weight_decay
        optimizer = optim.SGD(self.parameters(),
                              lr=lr,
                              momentum=0.9,
                              weight_decay=wd)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=self.args.max_epochs,
            warmup_start_lr=0.01 * lr,
            eta_min=0.01 * lr,
        )
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        return [optimizer], [scheduler]


if __name__ == '__main__':
    pl.seed_everything(5)
    args = parse_args()
    model = MSNetPL(args)
    wandb_logger = WandbLogger(name=args.name, project=args.project, entity=args.entity, offline=args.offline)
    wandb_logger.watch(model, log="gradients", log_freq=100)
    wandb_logger.log_hyperparams(args)
    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir, save_last=True, save_top_k=2, mode='max',
                                          monitor="val_acc3")
    trainer = pl.Trainer(gpus=args.num_gpus,
                         max_epochs=args.max_epochs,
                         check_val_every_n_epoch=5,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         precision=16,
                         gradient_clip_val=0.5,
                         logger=wandb_logger,
                         callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback])

    try:
        from pytorch_lightning.loops import FitLoop


        class WorkaroundFitLoop(FitLoop):
            @property
            def prefetch_batches(self) -> int:
                return 1


        trainer.fit_loop = WorkaroundFitLoop(
            trainer.fit_loop.min_epochs, trainer.fit_loop.max_epochs
        )
    except:
        pass

    dali_datamodule = ClassificationDALIDataModule(
        train_data_path=os.path.join(args.data_dir, 'train'),
        val_data_path=os.path.join(args.data_dir, 'val'),
        num_workers=args.num_workers,
        batch_size=args.batch_size)

    trainer.fit(model, datamodule=dali_datamodule)
