import os
import torch
import argparse
from pathlib import Path
from argparse import Namespace
from contextlib import suppress
from opt_utils.lars import LARS
from data_modules.dali_dataloader import ClassificationDALIDataModule


SUPPORTED_DATASETS = [
        "cifar10",
        "cifar100",
        "stl10",
        "imagenet",
        "imagenet100",
        "imagenet32",
        "custom",
    ]

OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "lars": LARS,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}
SCHEDULERS = [
    "reduce",
    "warmup_cosine",
    "step",
    "exponential",
    "none",
]
N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "imagenet": 1000,
    "imagenet100": 100,
}

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/torch_ds',required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--num_workers", type=int, default=4)





    # wandb
    parser.add_argument("--name")
    parser.add_argument("--project")
    parser.add_argument("--entity", default=None, type=str)
    parser.add_argument("--offline", action="store_true")

    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, type=str, required=True)
    # dataset path
    parser.add_argument("--train_data_path", type=Path, required=True)
    parser.add_argument("--val_data_path", type=Path, default=None)
    parser.add_argument(
        "--data_format",
        default="image_folder",
        choices=["image_folder", "dali"],
    )
    # percentage of data used from training, leave -1.0 to use all data available
    parser.add_argument("--data_fraction", default=-1.0, type=float)
    parser.add_argument("--crop_size", type=int, default=[224], nargs="+")


    # for custom dataset
    parser.add_argument("--mean", type=float, default=[0.485, 0.456, 0.406], nargs="+")
    parser.add_argument("--std", type=float, default=[0.228, 0.224, 0.225], nargs="+")

    # general train


    parser.add_argument(
        "--optimizer", choices=OPTIMIZERS.keys(), type=str, required=True
    )
    parser.add_argument("--grad_clip_lars", action="store_true")
    parser.add_argument("--eta_lars", default=1e-3, type=float)
    parser.add_argument("--exclude_bias_n_norm", action="store_true")

    parser.add_argument(
        "--scheduler", choices=SCHEDULERS, type=str, default="reduce"
    )
    parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--warmup_start_lr", default=0.00003, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument(
        "--scheduler_interval", choices=["step", "epoch"], default="step", type=str
    )

    temp_args, _ = parser.parse_known_args()

    if temp_args.data_format == "dali":
        parser = ClassificationDALIDataModule.add_dali_args(parser)


    args = parser.parse_args()
    additional_setup(args)
    return args



def additional_setup(args: Namespace):
    """Provides final setup for linear evaluation to non-user given parameters by changing args.

    Parsers arguments to extract the number of classes of a dataset, correctly parse gpus, identify
    if a cifar dataset is being used and adjust the lr.

    Args:
        args: Namespace object that needs to contain, at least:
        - dataset: dataset name.
        - optimizer: optimizer name being used.
        - gpus: list of gpus to use.
        - lr: learning rate.
    """

    if args.dataset in N_CLASSES_PER_DATASET:
        args.num_classes = N_CLASSES_PER_DATASET[args.dataset]
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        args.num_classes = max(
            1,
            len([entry.name for entry in os.scandir(args.train_data_path) if entry.is_dir]),
        )

    # create backbone-specific arguments
    if args.data_format == "dali":
        assert args.dataset in ["imagenet100", "imagenet", "custom"]

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9
    if args.optimizer == "lars":
        args.extra_optimizer_args["momentum"] = 0.9
        args.extra_optimizer_args["exclude_bias_n_norm"] = args.exclude_bias_n_norm

    with suppress(AttributeError):
        del args.exclude_bias_n_norm

    if isinstance(args.devices, int):
        args.devices = [args.devices]
    elif isinstance(args.devices, str):
        args.devices = [int(device) for device in args.devices.split(",") if device]


    scale_factor = args.batch_size * len(args.devices) / 256
    args.lr = args.lr * scale_factor