# torchrun --master_port=1255 
python train_sqlllama.py \
    --model_name_or_path /c1/kangsan/AI612/SQL-LLaMA2/checkpoints/SQL-LLaMA-7B-small \
    --data_path "datasets/ehrsql24_train_withnull.json" \
    --bf16 True \
    --output_dir ./output/6epoch/ \
    --num_train_epochs 6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --evaluation_strategy "no" \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 0 \
    --tf32 True \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 160 \
    --save_total_limit 2

# --deepspeed ds_config_sql.json \