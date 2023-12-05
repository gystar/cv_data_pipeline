sudo chmod 777 $(dirname $3)
screen -mS score_$1 python score_images.py -i $2 -o $3 -d $4