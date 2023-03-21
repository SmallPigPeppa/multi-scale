cd /mnt/mmtech01/usr/liuwenzhuo/code/multi-scale
DATA_PATH=/mnt/mmtech01/dataset/lzy/ILSVRC2012
/root/miniconda3/envs/solo-learn/bin/python baseline_train_32.py \
    --data_dir ${DATA_PATH} \
    --num_gpus 4 \
    --num_workers 8 \
    --batch_size 128 \
    --name baseline-supervised-32 \
    --ckpt_dir baseline-supervised-32

#cd /mnt/mmtech01/usr/liuwenzhuo/code/multi-scale
#DATA_PATH=/mnt/mmtech01/dataset/lzy/ILSVRC2012
#/root/miniconda3/envs/solo-learn-A100/bin/python baseline_train.py \
#    --data_dir ${DATA_PATH} \
#    --num_gpus 8 \
#    --num_workers 8 \
#    --batch_size 128 \
#    --ckpt_dir supervised-baseline-ckpt\
#    --name supervised-baseline



