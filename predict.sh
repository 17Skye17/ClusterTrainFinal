#!/bin/bash
set -e
cfg=trainer_config.py
model=ClusterTrain-4-26-g2305-4/pass-00092
paddle train \
    --config=$cfg \
    --use_gpu=true \
    --job=test \
    --init_model_path=$model \
    --config_args=is_predict=1 \
    --predict_output_dir=. 

python gen_result.py > result.csv

rm -rf rank-00000
