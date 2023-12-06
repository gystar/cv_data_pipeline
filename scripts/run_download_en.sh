# $1:output_dir是图片下载存储路径，会在目录下创建 images/chunk_n的目录存放
# $2:csv_dir为csv所在路劲，此目录中存放csv，例如0_parquet.csv，我已经把所有英文csv下载到 /data0/en/csv中
# $3:其中n表示csv的编号，比方说下载csv/0_parquet中的数据则写0，千万不要写错了，否则会覆盖其它csv的下载图片
# 此脚本可以多次运行，会自动从上次没有下载完成的继续
# 示例 sh scripts/run_download_en.sh ./csv ./ 0
screen -mS down_$3  python download.py -i $1 -o $2 -b $3