#python3 main_baseline.py \
#    --data_dir /share/wenzhuoliu/torch_ds/imagenet \
#    --name baseline-supervised \
#    --ckpt_dir baseline-supervised




cd /mnt/mmtech01/usr/liuwenzhuo/code/multi-scale
DATA_PATH=/mnt/mmtech01/dataset/lzy/ILSVRC2012
/root/miniconda3/envs/solo-learn-A100/bin/python baseline_val.py \
    --data_dir ${DATA_PATH} \
    --num_gpus 1 \
    --num_workers 8 \
    --batch_size 128 \
    --ckpt_dir supervised-baseline-ckpt \
    --name supervised-baseline \
    --project Multi-Scale-Net-val \
    --val_ckpt_path supervised-ckpt/supervised-baseline.ckpt