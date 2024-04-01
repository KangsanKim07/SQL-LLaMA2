#Please put this script in the main SQL-LLaMA folder to run training

torchrun --master_port=1212 train_sqlllama.py \
    --model_name_or_path ./models_hf/13B/ \
    --data_path "./datasets/sql_create_dataset_cleaned.json" \
    --bf16 True \
    --output_dir ./output_sqlAlpaca13B_10ep/ \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing True \
    --evaluation_strategy "no" \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 0 \
    --deepspeed ds_config_sql.json \
    --tf32 True \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
