#!/bin/bash

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /data/Databases/mir/s4/ -b 32 -e 50 -g 1 --step_lr --skd_step 25

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /data/Databases/mir/s4/ -b 32 -e 50 -g 1 --step_lr --skd_step 25 --vflip

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /data/Databases/mir/s4/ -b 32 -e 50 -g 1 --step_lr --skd_step 25 --hflip

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /data/Databases/mir/s4/ -b 32 -e 50 -g 1 --step_lr --skd_step 25 --vflip --hflip
