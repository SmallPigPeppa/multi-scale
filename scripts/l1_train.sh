cd /mnt/mmtech01/usr/liuwenzhuo/code/multi-scale
DATA_PATH=/mnt/mmtech01/dataset/lzy/ILSVRC2012
/root/miniconda3/envs/solo-learn-A100/bin/python l2_train.py \
    --data_dir ${DATA_PATH} \
    --num_gpus 8 \
    --num_workers 8 \
    --batch_size 128 \
    --ckpt_dir supervised-l2-ckpt\
    --name supervised-l2


export https_proxy=http://10.7.4.2:3128 ;cd /mnt/mmtech01/usr/liuwenzhuo/code/multi-scale ; /root/miniconda3/envs/solo-learn/bin/python l2_train.py --data_dir ${DATA_PATH} --num_gpus 8 --num_workers 8 --batch_size 128 --ckpt_dir supervised-l2-ckpt --name supervised-l2