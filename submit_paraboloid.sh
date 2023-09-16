#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J paraboloid
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u s164222@student.dtu.dk
#BSUB -o output/output_%J.out
#BSUB -e error/error_%J.err
#BSUB -B
#BSUB -N

#Load the following in case
#module load python/3.8
module swap cuda/8.0
module swap cudnn/v7.0-prod-cuda8

python3 train_paraboloid.py \
    --data_path ../../Data/paraboloid/paraboloid.csv \
    --save_model_path models/paraboloid/vae_paraboloid \
    --model_number _3
    --save_hours 1 \
    --img_size 64 \
    --num_img 0.8 \
    --device cuda \
    --workers 4 \
    --epochs 50000 \
    --batch_size 100 \
    --lr 0.0002 \
    --con_training 0 \
    --load_model_path trained_models/celeba_epoch_5700.pt
