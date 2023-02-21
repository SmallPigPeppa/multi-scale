cd /mnt/mmtech01/usr/liuwenzhuo/code/multi-scale
DATA_PATH=/mnt/mmtech01/dataset/lzy/ILSVRC2012
/root/miniconda3/envs/solo-learn/bin/python debug.py \
    --data_dir ${DATA_PATH} \
    --num_gpus 2 \
    --num_workers 16 \
    --ckpt_dir supervised-l1-ckpt\
    --name supervised-l1 \
    --offline