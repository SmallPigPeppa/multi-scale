python3 main.py \
    --dataset imagenet \
    --backbone resnet50 \
    --train_data_path /datasets/ILSVRC2012/train \
    --val_data_path /datasets/ILSVRC2012/val \
    --max_epochs 100 \
    --devices 0,1 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.45 \
    --accumulate_grad_batches 16 \
    --weight_decay 1e-6 \
    --batch_size 128 \
    --num_workers 4 \
    --data_format dali \
    --name multi-scale-net-supervised \
    --entity unitn-mhug \
    --project solo-learn



