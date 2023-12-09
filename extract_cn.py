import sys

sys.path.append(".")  # 这里将scripts目录的父目录添加到路径中
import os

import pandas as pd
import numpy as np

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

from concurrent.futures import ThreadPoolExecutor, as_completed


def get_new_dataset_generator(
    images_dir: Path, vae_transform_fn, vae, args
):
    def sample_generator():   
        pool = ThreadPoolExecutor(max_workers=32)
        files = os.listdir(images_dir)
        items = []        
        for _, f in enumerate(files):
            suffix = f.split("_")[-1]
            pos = suffix.rfind(".")    
            items.append([f, float(suffix[:pos])])           

        items = sorted(items, key=lambda x: x[1],reverse=True) 
        items = items[:int(args.threshold*len(items))]       
        for batch_start in range(0, len(items), args.batch_size):
            last = min(batch_start + args.batch_size, len(files))
            batch_files = items[batch_start:last]  
            def open_and_transfrom(path, cap):
                try:
                    image = Image.open(path)
                    with torch.no_grad():
                        return cap, vae_transform_fn(image)
                except Exception as e:
                    print(e)
                    return None,(None,None,None,None)
            futures = []  #小文件太多如果一个个打开会浪费时间,使用线程池  
            caps,features, original_sizes, crop_top_lefts, target_sizes = [],torch.zeros((1,3,512,512)), [], [], []            
            for _,(f,_) in  enumerate(batch_files):                
                file_path = images_dir / f
                if not file_path.exists():
                    continue  
                pos = f.rfind("_")
                futures.append(pool.submit(open_and_transfrom, file_path, f[:pos]))                     
                    
            for future in as_completed(futures):
                cap, (image, original_size, crop_top_left, target_size) = future.result() 
                if cap is None:
                    continue
                caps.append(cap)
                features = torch.concat([features, image.unsqueeze(0)],dim=0)   
                original_sizes.append(original_size)
                crop_top_lefts.append(crop_top_left)
                target_sizes.append(target_size)

            if len(caps) > 0: 
                features = features[1:].to(device=args.device,dtype=torch.float16)
                print(f"features size: {len(features)}, {features.numel()}")
                with torch.no_grad():
                    features = vae.encode(features).latent_dist.parameters.detach().to("cpu")         
                       
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

    
    # paths
    work_dir = Path(args.work_dir)
    input_dir = Path(args.images_dir)
    output_dir = work_dir/f"output/{str(input_dir.name)}"
    cache_dir = work_dir/f"cache/{str(input_dir.name)}"
    output_dir.mkdir(parents=True,exist_ok=True)
    cache_dir.mkdir(parents=True,exist_ok=True)


    vae_transform_fn = preprocess_train_image_fn(args)
    vae = (
        AutoencoderKL.from_pretrained(
            args.vae_model_path,
            subfolder="unet",
            cache_dir=cache_dir,
        )
        .to(device = args.device, dtype= args.dtype)
    )  
    ds = Dataset.from_generator( get_new_dataset_generator(input_dir, vae_transform_fn, vae, args), cache_dir=cache_dir,  num_proc=args.num_proc )

    ds.save_to_disk(output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_model_path", default="./sdxl_vae_fp16fix")
    parser.add_argument("--images_dir", default="./scored")
    parser.add_argument("--work_dir", default="./")
    parser.add_argument("--split", default=None)
    parser.add_argument("--image_resolution", default=512, type=int)
    parser.add_argument("--image_center_crop", default=True)
    parser.add_argument("--image_random_flip", default=True)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--num_proc", default=16, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()

    if args.dtype.lower() in ["fp16", "float16"]:
        args.dtype = torch.float16
    elif args.dtype.lower() in ["fp32", "float32"]:
        args.dtype = torch.float32
    else:
        raise RuntimeError(f"Unsupported dtype {args.dtype}")

    convert(args)
