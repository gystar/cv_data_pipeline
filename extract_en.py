import sys

sys.path.append(".")  # 这里将scripts目录的父目录添加到路径中

import os

import pandas as pd

# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)
# 获得当前脚本所在目录的父目录（即train目录所在的位置）
parent_dir = os.path.dirname(os.path.dirname(script_path))
# 将父目录添加到Python的搜索路径中
sys.path.append(parent_dir)
import argparse
from datasets import Dataset, load_dataset, load_from_disk

import torch
from diffusers import AutoencoderKL
from pathlib import Path

from train.utils import preprocess_train_image_fn
from PIL import Image


def get_image_name(id, url):  # id为csv中编号
    subfix = url.split(".")[-1]
    if len(subfix) > 4:  # 常见图片文件格式后缀长度不会超过4
        subfix = "uk"
    return f"{id}.{subfix}"


def get_new_dataset_generator(
    dataset: pd.DataFrame, images_dir: Path, vae_transform_fn, vae, args
):
    def sample_generator():        
        for i in range(0, len(dataset), args.batch_size):
            last = min(i + args.batch_size, len(dataset))
            urls = dataset.iloc[i:last]["URL"]           
            caps,features, original_sizes, crop_top_lefts, target_sizes = [],[], [], [], []            
            for j,url in  enumerate(urls):                
                filename = get_image_name(i, url)
                file_path = images_dir / filename
                if not file_path.exists():
                    continue
                try:
                    image = Image.open(file_path)
                    with torch.no_grad():
                        image, original_size, crop_top_left, target_size = vae_transform_fn(image)
                    #feature = image.unsqueeze(0)
                    caps.append(dataset.iloc[i:last]["TEXT"].to_list()[j])
                    features.append(image)
                    original_sizes.append(original_size)
                    crop_top_lefts.append(crop_top_left)
                    target_sizes.append(target_size)
                except Exception as e:
                    print(e)
                    continue
            if len(features) > 0:
                features = torch.tensor( [f.numpy() for f in features], device="cuda",dtype=torch.float16)
                features= list(vae.encode(features).latent_dist.parameters.detach().to("cpu"))           
                       
                for k in range(len(features)):
                    yield {
                    "caption": caps[k],
                    "image_feature": features[k],
                    "original_size": original_sizes[k],
                    "crop_top_left": crop_top_lefts[k],
                    "target_size": target_sizes[k],
                }  

    return sample_generator


def convert(args):
    # if os.path.exists(args.original_dataset_name_or_path):
    #     dataset = load_from_disk(args.original_dataset_name_or_path)
    # else:
    #     dataset = load_dataset(args.original_dataset_name_or_path)
    df_info = pd.read_csv(args.csv_path, usecols=[0, 1])

    if args.split is not None:
        df_info = df_info[args.split]
        df_info = df_info["train"]

    vae_transform_fn = preprocess_train_image_fn(args)
    vae = (
        AutoencoderKL.from_pretrained(
            args.vae_model_path,
            subfolder="unet",
            cache_dir=args.cache_dir,
        )
        .to(args.dtype)
        .cuda()
    )  
    ds = Dataset.from_generator( get_new_dataset_generator(df_info, Path(args.images_dir), vae_transform_fn, vae, args), cache_dir=args.cache_dir,  num_proc=args.num_proc )

    ds.save_to_disk(args.output_dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dataset_path", default="./output/chunk_0")
    parser.add_argument("--vae_model_path", default="./sdxl_vae_fp16fix")
    parser.add_argument("--images_dir", default="./images/chunk_0")
    parser.add_argument("--csv_path", default="./csv/0_parquet.csv")
    parser.add_argument("--split", default=None)
    parser.add_argument("--image_resolution", default=512, type=int)
    parser.add_argument("--image_center_crop", default=True)
    parser.add_argument("--image_random_flip", default=True)
    parser.add_argument("--cache_dir", default="./cache/chunk_0")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--num_proc", default=None, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    args = parser.parse_args()

    if args.dtype.lower() in ["fp16", "float16"]:
        args.dtype = torch.float16
    elif args.dtype.lower() in ["fp32", "float32"]:
        args.dtype = torch.float32
    else:
        raise RuntimeError(f"Unsupported dtype {args.dtype}")

    convert(args)
