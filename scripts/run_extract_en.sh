python extract_en.py --cache_dir /data/llm_sd/remote_mount/cache/chunk$1/ \
--output_dataset_path /data/llm_sd/remote_mount/output/chunk$1 \
--csv_path /data/llm_sd/remote_mount/csv/${1}_parquet.csv 

sudo chmod 777 /data0/en/01/laion_coco 
python extract_en.py --work_dir /data0/en/01/laion_coco \
--csv_path /data0/en/csv/2_parquet.csv \
--images_dir /data0/en/01/laion_coco/images/chunk_2 