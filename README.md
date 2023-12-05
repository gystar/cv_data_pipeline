# data_pipeline
This is a pipeline for processing image data

# 下载
python==3.8.18

screen -mS down_n python download.py -b n -s dir_path
# 其中n表示csv的编号，比方说下载csv/0_parquet中的数据则写0，千万不要写错了，否则会覆盖其它csv的下载图片
# r表示断点恢复下载位置，默认0表示从新下载，否则从r那一行开始下载
# dir_path是数据路径，可以不改直接使用默认的/mnt/data1/laion_coco （当然其它服务器也要把硬盘挂载到/mnt/data1）
