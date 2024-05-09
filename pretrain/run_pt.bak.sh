CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29560 \
    --config_file accelerate_config.yaml \
    pretrain_v1.py \
    --model_name_or_path /data2/model_file/pretrained-checkpoint/Baichuan2-7B-Base \
    --train_file_dir /data2/model_file/dataset/ready-for-train/pretrain/json/train \
    --validation_file_dir /data2/model_file/dataset/ready-for-train/pretrain/json/valid \
    --output_dir /data2/home/yangbt/projects/llm-pretrain/baichuan2/output/pt1 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --tokenizer_name_or_path /data2/model_file/pretrained-checkpoint/Baichuan2-7B-Base \
    --use_fast_tokenizer False \
    --bf16 \
    --do_train \
    --do_eval \
    --seed 42 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --save_steps 10000 \
    --eval_steps 5000 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 20 \
    --block_size 2048 \
    --overwrite_output_dir