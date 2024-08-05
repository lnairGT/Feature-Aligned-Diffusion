CUDA_VISIBLE_DEVICES=1 python train_sd.py \
    --config config.yml \
    --num-img-gens 5 \
    --do-train \
    --use-feature-alignment
