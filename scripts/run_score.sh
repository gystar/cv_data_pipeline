sudo chmod 777 $(dirname $3)
screen -mS score_$1 python score_images.py -s $2 -d $3