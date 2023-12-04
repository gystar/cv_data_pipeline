import os,sys
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("-d","--dir", type=str,default="/mnt/data1/laion_coco/images/chunk_0")
args= parser.parse_args()

files=os.listdir(args.dir)

#counts = [int(f.split(".")[0]) for f in files]
print(len(files))#, max(counts))
