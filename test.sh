#!/bin/bash
DEVICE=0
PORT=29507
seeds=(35)
# seeds=(55)
for seed in ${seeds[*]}
do
    echo 'seed: ' $seed
    CUDA_VISIBLE_DEVICES=$DEVICE \
        python -m torch.distributed.launch \
            --nproc_per_node=1 \
            --use-env \
            --master_port $PORT test_BLEEP_her2_main.py \
            --model stnet \
            --exp_name gene_query_name_her2_518 \
            --image_embedding_dim 2048 \
            --spot_embedding_dim 3468 \
            --max_length 785 \
            --batch_size 100 \
            --max_epochs 100 \
            --lr 0.0003 \
            --seed $seed
            # --num_projection_layers 2 \                  
done