python extract_en.py --cache_dir /data/llm_sd/remote_mount/cache/chunk$1/ \
--output_dataset_path /data/llm_sd/remote_mount/output/chunk$1 \
--vae_model_path /data/llm_sd/sdxl-vae-fp16-fix/ \
--image_column URL --caption_column TEXT \
--original_dataset_name_or_path /data/llm_sd/remote_mount/csv/${1}_parquet.csv 

python extract_en.py --cache_dir ./cache/chunk_0 \
--output_dataset_path ./output/chunk_0 \
--vae_model_path /data/llm_sd/sdxl-vae-fp16-fix/ \
--image_column URL --caption_column TEXT \
--original_dataset_name_or_path ./csv/0_parquet.csv 