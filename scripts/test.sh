python3 ../main.py --gpu $1 \
                   --mode test \
                   --n-retrievals 10\
                   --n-eps-finetuning 50\
                   --batch-size 32\
                   --load-path /v8/wyjeong/research/tans_test/20211027_0147\
                   --data-path /v8/wyjeong/research/tans_test/data\
                   --model-zoo /v8/wyjeong/research/tans_test/data/model_zoo.pt\
                   --model-zoo-raw /v14/geon/samsung_model_zoo/ofa_models\
                   --seed 777