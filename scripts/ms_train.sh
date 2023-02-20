cd /mnt/mmtech01/usr/liuwenzhuo/code/multi-scale
DATA_PATH=/mnt/mmtech01/dataset/lzy/ILSVRC2012
/root/miniconda3/envs/solo-learn/bin/python main_ms.py \
    --data_dir ${DATA_PATH} \
    --num_gpus 8 \
    --num_workers 4