#!/bin/bash

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /export/livia/data/CommonData/cows_data/clt/final/s4/ -b 32 -e 50 -g 0 --step_lr --skd_step 25

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /export/livia/data/CommonData/cows_data/clt/final/s4/ -b 32 -e 50 -g 0 --step_lr --skd_step 25 --vflip

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /export/livia/data/CommonData/cows_data/clt/final/s4/ -b 32 -e 50 -g 0 --step_lr --skd_step 25 --hflip

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /export/livia/data/CommonData/cows_data/clt/final/s4/ -b 32 -e 50 -g 0 --step_lr --skd_step 25 --vflip --hflip
