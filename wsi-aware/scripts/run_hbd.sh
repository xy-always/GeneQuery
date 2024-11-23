#!/bin/bash
#SBATCH --job-name=hbd_chatgpt_add
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
DATA_DIR=$PROJECT_ROOT/../HBD
# DATA_DIR=$PROJECT_ROOT/data/GSE
LOG_DIR=$PROJECT_ROOT/logs
IMAGE_ENCODER=resnet50
MODEL_TYPE=gene_query_des
GENE_ENCODER_TYPE=mlp
NUM_PROJECTION_LAYERS=2
GENE_PROJECTION_DIM=256
EPOCHS=100
LR=0.0001
BATCH_SIZE=30
datafolds=(0 1 2 3 4)
# seeds=(35)
DATA_NAME=`basename $DATA_DIR`
if [ ! -d $LOG_DIR ]; then
    mkdir $LOG_DIR
fi
for fold in ${datafolds[*]}
do
    echo 'FOLD:'$fold
    OUTPUT_DIR=$PROJECT_ROOT/checkpoints_wsi/$DATA_NAME-$MODEL_TYPE-$IMAGE_ENCODER-$GENE_ENCODER_TYPE-dim$GENE_PROJECTION_DIM-mlp$NUM_PROJECTION_LAYERS-epoch$EPOCHS-lr$LR-crossvalidation$fold
    RESUME_DIR=$OUTPUT_DIR/checkpoint-1950
    CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
            --nnodes=1 \
            --nproc-per-node=$NUM_DEVICE \
            --master_port=12345 \
            $PROJECT_ROOT/wsi-aware/src/main.py \
            --model_type $MODEL_TYPE \
            --gene_dim $GENE_PROJECTION_DIM \
            --gene_num 54683 \
            --gene_total 54683 \
            --image_encoder_name $IMAGE_ENCODER \
            --num_classes 0 \
            --dataset_fold $fold \
            --dataset hbd \
            --global_pool avg \
            --pretrained \
            --trainable \
            --train_ratio 0.9 \
            --image_embedding_dim 2048 \
            --num_projection_layers $NUM_PROJECTION_LAYERS \
            --projection_dim $GENE_PROJECTION_DIM \
            --n_layers 2 \
            --gene_embedding_dim 768 \
            --wsi_max_length 600 \
            --dropout 0.1 \
            --fuse_method add \
            --data_dir $DATA_DIR \
            --do_eval \
            --do_train \
            --include_inputs_for_metrics \
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
            # > $LOG_DIR/$DATA_NAME-$MODEL_TYPE-$IMAGE_ENCODER-$GENE_ENCODER_TYPE-dim$GENE_PROJECTION_DIM-mlp$NUM_PROJECTION_LAYERS-epoch$EPOCHS-lr$LR-$seed.log 2>&1