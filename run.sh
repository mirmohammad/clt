#!/bin/bash

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /data/Databases/s4/ -b 64 -e 60 -g 0 --step_lr --skd_step 30

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /data/Databases/s4/ -b 64 -e 60 -g 0 --step_lr --skd_step 30 --vflip

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /data/Databases/s4/ -b 64 -e 60 -g 0 --step_lr --skd_step 30 --hflip

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /data/Databases/s4/ -b 64 -e 60 -g 0 --step_lr --skd_step 30 --vflip --hflip
