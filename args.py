import os
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./torch_ds', required=True)
    parser.add_argument("--ckpt_dir", type=str, default='./supervised_ckpt')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=90)

    # wandb
    parser.add_argument("--name", type=str, default='multi-scale-net-l1')
    parser.add_argument("--project", type=str, default='Multi-Scale-Net')
    parser.add_argument("--entity", type=str, default='pigpeppa')
    parser.add_argument("--offline", action="store_true")

    args = parser.parse_args()
    return args
