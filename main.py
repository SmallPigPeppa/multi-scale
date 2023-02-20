from data_modules.dali_dataloader import ClassificationDALIDataModule
from data_modules.torch_dataloader import prepare_data
from models.multi_resnet_l1_v2 import MultiScaleNet
from args import parse_args

from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy


if __name__=='__main__':
    large_imgs = torch.rand(8, 3, 224, 224)
    model = MultiScaleNet()
    out = model(large_imgs)
    print("end")

    args = parse_args()
    callbacks = []
    wandb_logger = WandbLogger(name=args.name, project=args.project, entity=args.entity, offline=args.offline)
    wandb_logger.watch(model, log="gradients", log_freq=100)
    wandb_logger.log_hyperparams(args)
    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir, save_last=True, save_top_k=2, monitor="total_loss")

    trainer = Trainer(
        gradient_clip_val=10.0,
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10), checkpoint_callback],
    )








    if args.data_format == "dali":
        val_data_format = "image_folder"
    else:
        val_data_format = args.data_format
    train_loader, val_loader = prepare_data(
        args.dataset,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        data_format=val_data_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.data_format == "dali":
        dali_datamodule = ClassificationDALIDataModule(
            dataset=args.dataset,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            data_fraction=args.data_fraction,
            dali_device=args.dali_device,
        )

        # use normal torchvision dataloader for validation to save memory
        dali_datamodule.val_dataloader = lambda: val_loader

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        enable_checkpointing=False,
        strategy=DDPStrategy(find_unused_parameters=False)
        if args.strategy == "ddp"
        else args.strategy,
    )


    if args.data_format == "dali":
        trainer.fit(model, datamodule=dali_datamodule)
    else:
        trainer.fit(model, train_loader, val_loader)
