export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
python scripts/create_train_dataset.py --cache_dir /data/llm_sd/remote_mount/cache/chunk$1/ \
--output_dataset_path /data/llm_sd/remote_mount/output/chunk$1 \
--vae_model_path /data/llm_sd/sdxl-vae-fp16-fix/ \
--image_column URL --caption_column TEXT \
--original_dataset_name_or_path /data/llm_sd/remote_mount/csv/${1}_parquet.csv 