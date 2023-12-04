from pathlib import Path
import os,sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import os,sys
import pandas as pd
import argparse
import tqdm

parser = argparse.ArgumentParser()
THRESHOLD_PERCENT=0.5

parser.add_argument("-s", "--source_dir", type=str, default="/home/starmage/data_pipeline/scored")
parser.add_argument("-d", "--dest_dir", type=str, default="/home/starmage/data_pipeline/filtered")
args = parser.parse_args()
scored_images_dir = Path(args.source_dir) 
filtered_images_dir = Path(args.dest_dir)
filtered_images_dir.mkdir(exist_ok=True,parents=True)
scores = []
full_paths = []
for f in os.listdir(scored_images_dir):
    suffix = f.split("_")[-1]
    pos = suffix.rfind(".")    
    scores.append(float(suffix[:pos]))
    full_paths.append(str(scored_images_dir/f))

df = pd.DataFrame({"path":full_paths, "score":scores})

df.sort_values(by=["score"], ascending=False,inplace=True)
sample_count = int(THRESHOLD_PERCENT*len(df))
df_sampled = df.iloc[:sample_count]

thread_pool = ThreadPoolExecutor(16)
futures=[]
for i,(_,row) in tqdm.tqdm(enumerate(df_sampled.iterrows())):  
    from_path = Path(row["path"])  
    to_path = filtered_images_dir/ from_path.name    
    futures.append(thread_pool.submit(os.system, f"cp '{str(from_path)}' '{str(to_path)}'") )
    if len(futures) == 1000 or i+1==len(df_sampled):
        completed = as_completed(futures)        
        futures=[] 
        
        

    
    
