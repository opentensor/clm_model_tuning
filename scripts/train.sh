python run_clm.py \
    --model_name_or_path facebook/opt-30b \
    --dataset_name bittensor \
    --dataset_config_name bittensor \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm