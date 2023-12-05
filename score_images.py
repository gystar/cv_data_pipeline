from components import Score, Gan
import time
import argparse
from pathlib import Path
import os



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_dir", type=str, default="/home/starmage/data_pipeline/images")
    parser.add_argument("-o", "--output_dir", type = str, default="/home/starmage/data_pipeline/scored")
    parser.add_argument("-d", "--device", type = str, default="cuda")
    args = parser.parse_args()
    t1 = time.time()
    score = Score()
    dest_dir = Path(args.output_dir)    
    dest_dir.mkdir(parents=False, exist_ok=True)
    score.predict(args.input_dir,args.output_dir) 
    print(f"duration {time.time()-t1}")
