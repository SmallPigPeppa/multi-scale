cd /mnt/mmtech01/usr/liuwenzhuo/code/multi-scale
DATA_PATH=/mnt/mmtech01/dataset/lzy/ILSVRC2012
/root/miniconda3/envs/test2/bin/python main_ms.py \
    --data_dir ${DATA_PATH} \
    --num_gpus 2 \
    --num_workers 32 \
    --batch_size 128 \
    --ckpt_dir supervised-l1-ckpt\
    --name supervised-l1