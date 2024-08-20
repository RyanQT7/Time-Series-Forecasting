@echo off
REM Windows batch script equivalent

set "dataset_prefix=FD002_order_differ_interpolation_-x1_xx2_4x6_256x384_"
set "python_path=D:/Anaconda3/envs/pytorch-gpu/python.exe"
set "script_path=H:\dachang\ViTST-main\code\FD003_train\run_VisionTextCLS.py"

for %%a in (%dataset_prefix%) do (
    set "CUDA_VISIBLE_DEVICES=0"
    %python_path% %script_path% ^
        --image_model swin ^
        --text_model roberta ^
        --freeze_vision_model False ^
        --freeze_text_model False ^
        --dataset FD003 ^
        --dataset_prefix=%%a ^
        --seed 10 ^
        --save_total_limit 1 ^
        --train_batch_size 6 ^
        --eval_batch_size 16 ^
        --logging_steps 4 ^
        --save_steps 100 ^
        --epochs 20 ^
        --learning_rate 2e-5 ^
        --n_runs 1 ^
        --n_splits 5 ^
        --do_train
)