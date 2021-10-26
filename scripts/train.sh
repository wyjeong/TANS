python3 ../main.py --gpu $1 \
                   --mode train \
                   --batch-size 140 \
                   --n-epochs 10000 \
                   --base-path /v8/wyjeong/research/tans_test\
                   --data-path /v8/wyjeong/research/tans_test/data\
                   --model-zoo /v8/wyjeong/research/tans_test/data/model_zoo.pt\
                   --model-zoo-raw /v14/geon/samsung_model_zoo/ofa_models\
                   --seed 777 