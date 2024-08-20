#!/usr/bin/env bash
#sh vtcls_script.sh

# P12 dataset
for dataset_prefix in FD002_order_differ_interpolation_-x1_xx2_21x1_1344x1024_
#for dataset_condition in ConID_FD002_order_differ_interpolation_-x1_xx2_21x1_3360x2560_
do
CUDA_VISIBLE_DEVICES=0 /root/.virtualenvs/ViTST_fzy/bin/python run_VisionTextCLS.py \
    --image_model swin \
    --text_model roberta \
    --freeze_vision_model False \
    --freeze_text_model False \
    --dataset FD002 \
    --dataset_prefix $dataset_prefix \
    --dataset_condition ConID_FD002_order_differ_interpolation_-x1_xx2_21x1_3360x2560_\
    --seed 10 \
    --save_total_limit 1 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --logging_steps 3 \
    --save_steps 3 \
    --epochs 20 \
    --learning_rate 2e-5 \
    --n_runs 1 \
    --n_splits 5 \
    --do_train True
done

# P19 dataset
#for dataset_prefix in order_differ_interpolation_-*1_**2_6*6_384*384_
#do
#CUDA_VISIBLE_DEVICES=0 python3 run_VisionVisionCLS.py \
#    --image_model swin \
#    --text_model roberta \
#    --freeze_vision_model False \
#    --freeze_text_model False \
#    --dataset P19 \
#    --dataset_prefix $dataset_prefix \
#    --seed 1799 \
#    --save_total_limit 1 \
#    --train_batch_size 48 \
#    --eval_batch_size 128 \
#    --logging_steps 20 \
#    --save_steps 100 \
#    --epochs 2 \
#    --learning_rate 2e-5 \
#    --n_runs 1 \
#    --n_splits 5 \
#    --do_train
#done
