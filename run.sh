#!/bin/bash

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /data/Databases/s4/ -b 64 -e 50 --step_lr --skd_step 25

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /data/Databases/s4/ -b 64 -e 30 --step_lr --skd_step 15

echo
echo ">>>>>>>>>>>>>>>>>>> START"
python main.py /data/Databases/s4/ -b 64 -e 30 --step_lr --skd_step 15
