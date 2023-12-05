screen -L -mS create_zh_dataset python scripts/create_train_dataset_zh.py \
--cache_dir /data0/zh/$1/laion_coco/cache \
--output_dataset_path /data0/zh/$1/laion_coco/output/chunk_$2 \
--vae_model_path /data/llm_sd/sdxl-vae-fp16-fix/ \
--image_column URL --caption_column TEXT \
--original_dataset_name_or_path /data0/zh/$1/laion_coco/images/chunk_$2 \
--num_proc 8
