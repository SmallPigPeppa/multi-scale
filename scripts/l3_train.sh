cd /mnt/mmtech01/usr/liuwenzhuo/code/multi-scale
DATA_PATH=/mnt/mmtech01/dataset/lzy/ILSVRC2012
/root/miniconda3/envs/solo-learn-A100/bin/python main_ms.py \
    --data_dir ${DATA_PATH} \
    --num_gpus 4 \
    --num_workers 8 \
    --batch_size 128 \
    --ckpt_dir supervised-l1-ckpt\
    --name supervised-l1


export https_proxy=http://10.7.4.2:3128 ;cd /mnt/mmtech01/usr/liuwenzhuo/code/multi-scale ; /root/miniconda3/envs/solo-learn-A100/bin/python l3_train.py --data_dir ${DATA_PATH} --num_gpus 8 --num_workers 8 --batch_size 128 --ckpt_dir supervised-l3-ckpt --name supervised-l3