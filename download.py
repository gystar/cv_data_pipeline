# %%
import pandas as pd
import json
import os,sys
import requests
import multiprocessing
from pathlib import Path
import argparse
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import logging
LOGGER = logging.getLogger(__name__)

def get_image_name(id, url):
    subfix = url.split('.')[-1]
    if len(subfix) > 4: #常见图片文件格式后缀长度不会超过4
        subfix = "uk"   
    return f"{id}.{subfix}"

def setup_log(log_file:Path):
    LOGGER.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    # 再创建一个handler，用于输出到控制台
    #console_handler = logging.StreamHandler()
    #console_handler.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    #console_handler.setFormatter(formatter)

    # 给logger添加handler
    LOGGER.addHandler(handler)
    #LOGGER.addHandler(console_handler)  


class Downloader:
    def __init__(self, urls, ids:List, save_dir:Path, retry_times=3, nworks=64):
        self.urls = urls
        self.retry_times = retry_times
        self.save_dir=save_dir
        self.ids=ids #下载的文件以dataframe中的id来区分
        self.executor = ThreadPoolExecutor(max_workers=nworks)
        self.max_submit = 1000 # 一次最多submit的任务数量，太多可能会内存溢出
        
        #统计
        self.success_count = 0
        self.fail_count = 0
        self.begin_count = 0
        self.start_time = time.time()
        self.last_time = time.time()
        self.last_success = 0
   

    def download(self, url, id, retry_times, timeout=5):
        try:
            response = requests.get(url, timeout=timeout, proxies={"http":"http://127.0.0.1:7890/", "https":"http://127.0.0.1:7890/"})
            if response.status_code == 200:                             
                return id,response.content
            else:
                if retry_times > 0:
                    logging.debug(f"下载失败，状态码：{response.status_code}，正在重试：{id}:{url}")
                    return self.download(url, id,retry_times-1)
                else:
                    LOGGER.error(f"重试结束，下载失败：{id}:{url}")                                        
                    return id,None
        except Exception as e:
            if retry_times > 0:
                LOGGER.debug(f"下载异常，原因：{str(e)}，正在重试：{id}:{url}")
                return self.download(url, id, retry_times-1)
            else:
                LOGGER.error(f"重试结束，下载异常：{id}:{url}，原因：{str(e)}")                                
                return id,None
            
    def stat(self):                    
        duration_total = time.time()-self.start_time
        duration_current = time.time()-self.last_time
        success_current = self.success_count - self.last_success
        speed_total = duration_total/self.success_count if self.success_count>0 else float("nan")
        speed_current = duration_current/success_current if success_current>0 else float("nan")
        LOGGER.info(f"\n[===Stat Info===]\n已经尝试下载{self.begin_count}个图片，耗时{duration_total}s，\n其中成功下载{self.success_count}个, \
            失败{self.fail_count}个，平均速度{speed_total}s/张，实时速度{speed_current}s/张。")
        self.last_time = time.time()
        self.last_success = self.success_count
        

    def save_image(self, id, url, filename,image_content):
        try:           
            with open(self.save_dir / filename, 'wb') as f:
                f.write(image_content)
            LOGGER.debug(f"下载成功：{id}:{url}") 
            return True                       
        except Exception as e:
            LOGGER.error(f"下载失败：{id}:{url}, 原因：{str(e)}")
            return False

    def run(self):
        futures = []
        for id, url in  zip(self.ids, self.urls):            
            futures.append(self.executor.submit(self.download, url, id, self.retry_times))
            self.begin_count += 1
            
            # 每self.max_submit条等待执行完成，检查一下进度，不要等全部sumbit，那样会导致内存溢出
            if len(futures) == self.max_submit or self.begin_count ==len(self.ids):                    
                for future in as_completed(futures):                    
                    url = self.urls[futures.index(future)]
                    id, content = future.result()
                    if content is not None:             
                        if self.save_image(url, id, get_image_name(id, url), content):
                            self.success_count += 1
                        else:
                            self.fail_count += 1                        
                    else:
                        self.fail_count += 1
                self.stat()
                futures = []
            
if __name__ =="__main__":               
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_csv_dir", type=str, default="/data0/en/csv")
    parser.add_argument("-o", "--output_dir", type=str, required=True)    
    parser.add_argument("-b", "--block_index", type=str, required=True)
    # 支持断线重新下载，会在目标文件夹中找到序号最大的文件作为起始点继续下载
    parser.add_argument("-r", "--resume", type=bool, default=True) 
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    
    image_dir=output_dir / f"images/chunk_{args.block_index}"
    image_dir.mkdir(parents=True, exist_ok=True)

    resume_from = 0
    if args.resume:
        try:
            files=os.listdir(image_dir)
            ids = [int(f.split(".")[0]) for f in files]
            resume_from = max(ids)
        except:
            resume_from = 0

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_log(log_dir/f"{args.block_index}.log")
    
    df = pd.read_csv(Path(args.input_csv_dir) / f"{args.block_index}_parquet.csv")  
    LOGGER.info(f"dataset size: {len(df)-resume_from}")


    # %%  
 
    urls = df.URL.tolist()[resume_from:]
    ids = df.index.tolist()[resume_from:]
    # 删除df节约内存
    del df
    df = None    
    downloader = Downloader(urls, ids, image_dir,1)
    downloader.run()




