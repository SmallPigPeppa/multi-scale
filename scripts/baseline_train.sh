cd /mnt/mmtech01/usr/liuwenzhuo/code/multi-scale
DATA_PATH=/mnt/mmtech01/dataset/lzy/ILSVRC2012
/root/miniconda3/envs/solo-learn/bin/python baseline_train.py \
    --data_dir ${DATA_PATH} \
    --num_gpus 4 \
    --num_workers 4 \
    --name baseline-supervised \
    --ckpt_dir baseline-supervised





