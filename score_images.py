from components import Score, Gan
import time
import argparse
from pathlib import Path
import os



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--source_dir", type=str, default="/home/starmage/data_pipeline/images")
    parser.add_argument("-d", "--dest_dir", type = str, default="/home/starmage/data_pipeline/scored")
    args = parser.parse_args()
    t1 = time.time()
    score = Score()
    dest_dir = Path(args.dest_dir)    
    dest_dir.mkdir(parents=False, exist_ok=True)
    score.predict(args.source_dir,args.dest_dir) 
    print(f"duration {time.time()-t1}")
