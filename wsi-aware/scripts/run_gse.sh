#!/bin/bash
#SBATCH --job-name=gse_chatgpt_add
#SBATCH --mem=230G
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --cpus-per-task=64 # Number of CPU cores
#SBATCH -p cscc-gpu-p # Use the gpu partition
#SBATCH --time=72:00:00 # Specify the time needed for you job
#SBATCH -q cscc-gpu-qos # To enable the use of up to 8 GPUs

DEVICE=1
NUM_DEVICE=1
PROJECT_ROOT=/path
DATA_DIR=$PROJECT_ROOT/../data
# DATA_DIR=$PROJECT_ROOT/data/GSE
IMAGE_ENCODER=resnet50
MODEL_TYPE=gene_query_des
GENE_ENCODER_TYPE=mlp
NUM_PROJECTION_LAYERS=2
GENE_PROJECTION_DIM=256
EPOCHS=100
LR=0.0001
BATCH_SIZE=1
echo $PROJECT_ROOT
FOLDS=(0 1 2 3)
for fold in ${FOLDS[*]}
do
    echo 'FOLD:'$fold
    OUTPUT_DIR=$PROJECT_ROOT/checkpoints_wsi/$DATA_NAME-$MODEL_TYPE-$IMAGE_ENCODER-$GENE_ENCODER_TYPE-dim$GENE_PROJECTION_DIM-mlp$NUM_PROJECTION_LAYERS-epoch$EPOCHS-lr$LR-crossvalidation$fold-gpt
    CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
            --nnodes=1 \
            --nproc-per-node=$NUM_DEVICE \
            --master_port 12344 \
            $PROJECT_ROOT/wsi-aware/src/main.py \
            --model_type gene_query_des \
            --gene_dim $GENE_PROJECTION_DIM \
            --gene_num 54683 \
            --gene_total 54683 \
            --image_encoder_name 'resnet50' \
            --num_classes 0 \
            --dataset gse \
            --global_pool avg \
            --pretrained \
            --trainable \
            --train_ratio 0.9 \
            --image_embedding_dim 2048 \
            --num_projection_layers 2 \
            --projection_dim 256 \
            --n_layers 2 \
            --gene_embedding_dim 768 \
            --wsi_max_length 2300 \
            --dropout 0.1 \
            --dataset_fold $fold \
            --fuse_method add \
            --data_dir $DATA_DIR \
            --gene_name_file $PROJECT_ROOT/../data/bleep_description.npy \
            --gene_des_file $PROJECT_ROOT/../data/gse_gpt_description_emb.npy \
            --include_inputs_for_metrics \
            --do_eval \
            --do_train \
            --output_dir $OUTPUT_DIR \
            --per_device_train_batch_size $BATCH_SIZE \
            --per_device_eval_batch_size $BATCH_SIZE \
            --num_train_epochs $EPOCHS \
            --learning_rate $LR \
            --gradient_accumulation_steps 1 \
            --save_strategy epoch \
            --eval_strategy epoch \
            --logging_strategy epoch \
            --seed 2025
done


# --model_name_or_path /l/users/ying.xiong/projects/GeneQuery-upload/GSE_genename_vit_35/checkpoint-1274 \
