import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from models.msnet_l1 import MultiScaleNet
from data_modules.imagenet_dali import ClassificationDALIDataModule
from data_modules.not_dali import prepare_data
from pytorch_lightning.strategies.ddp import DDPStrategy
from args import parse_args
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from l1_train import MSNetPL
import torch.nn.functional as F


class MSNetValPL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ms_model = MSNetPL(args)
        self.ms_model = self.ms_model.load_from_checkpoint(args.val_ckpt_path)
        self.size_list = list(range(args.start_size, args.end_size, args.interval))

    def forward(self, x):
        z1, z2, z3, y1, y2, y3 = self.ms_model.encoder(x)
        return z1, z2, z3, y1, y2, y3

    def share_step(self, x, target):
        z1, z2, z3, y1, y2, y3 = self.forward(x)

        acc1 = (torch.argmax(y1, dim=1) == target).float().mean()
        acc2 = (torch.argmax(y2, dim=1) == target).float().mean()
        acc3 = (torch.argmax(y3, dim=1) == target).float().mean()

        result_dict = {
            "acc1": acc1,
            "acc2": acc2,
            "acc3": acc3,
        }
        return result_dict

    def validation_step(self, batch, batch_idx):
        x, target = batch
        dict_list = []
        for size_i in self.size_list:
            x_size_i = F.interpolate(x, size=int(size_i), mode='bilinear')
            dict_i = self.share_step(x_size_i, target)
            dict_size_i = {f'{str(size_i)}_{k}': v for k, v in dict_i.items()}
            dict_list.append(dict_size_i)

        all_size_dict = {k: v for d in dict_list for k, v in d.items()}
        # self.log_dict(all_size_dict)
        return all_size_dict

    def validation_end(self, outputs):
        acc1_list = []
        acc2_list = []
        acc3_list = []
        acc_best_list = []
        for size_i in self.size_list:
            avg_acc1_size_i = sum([output[f"{size_i}_acc1"] for output in outputs]) / len(outputs)
            avg_acc2_size_i = sum([output[f"{size_i}_acc2"] for output in outputs]) / len(outputs)
            avg_acc3_size_i = sum([output[f"{size_i}_acc3"] for output in outputs]) / len(outputs)
            avg_acc_best_size_i = max(avg_acc1_size_i, avg_acc2_size_i, avg_acc3_size_i)
            acc1_list.append(avg_acc1_size_i)
            acc2_list.append(avg_acc2_size_i)
            acc3_list.append(avg_acc3_size_i)
            acc_best_list.append(avg_acc_best_size_i)

        self.log({"acc1": acc1_list, "x_axis": self.size_list}, on_step=False, prog_bar=False)
        self.log({"acc2": acc2_list, "x_axis": self.size_list}, on_step=False, prog_bar=False)
        self.log({"acc3": acc3_list, "x_axis": self.size_list}, on_step=False, prog_bar=False)
        self.log({"acc_best": acc_best_list, "x_axis": self.size_list}, on_step=False, prog_bar=False)


if __name__ == '__main__':
    pl.seed_everything(5)
    args = parse_args()
    val_ckpt_path = args.val_ckpt_path
    model = MSNetValPL(args)
    wandb_logger = WandbLogger(name=args.name, project=args.project, entity=args.entity, offline=args.offline)
    wandb_logger.watch(model, log="gradients", log_freq=100)
    wandb_logger.log_hyperparams(args)
    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir, save_last=True, save_top_k=2, monitor="val_acc3")
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

    trainer.validate(model, datamodule=dali_datamodule)