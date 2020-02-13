# generate dataset
python3 Generate_TFRecord.py

# 
python3 Train.py --use_gpu 0
python3 Train.py --use_gpu 0,1,2,3,4,5,6,7
