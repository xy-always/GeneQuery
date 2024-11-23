#!/bin/bash
#SBATCH --job-name=gse_genename
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Ying.Xiong@mbzuai.ac.ae
#SBATCH --mem=230G
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --cpus-per-task=64 # Number of CPU cores
#SBATCH -p cscc-gpu-p # Use the gpu partition
#SBATCH --time=12:00:00 # Specify the time needed for you job
#SBATCH -q cscc-gpu-qos # To enable the use of up to 8 GPUs

DEVICE=1
NUM_DEVICES=1
PROJECT_ROOT=/path
DATA_DIR=$PROJECT_ROOT/data/HBD
# DATA_DIR=$PROJECT_ROOT/data/GSE
LOG_DIR=$PROJECT_ROOT/logs
IMAGE_ENCODER=resnet50
MODEL_TYPE=gene_des_transformer
GENE_ENCODER_TYPE=mlp
NUM_PROJECTION_LAYERS=2
GENE_PROJECTION_DIM=256
EPOCHS=100
LR=0.0001
N_LAYER=2
BATCH_SIZE=30
FOLDS=(0 1 2 3 4)
DATASET=hbd
seed=2024
DATA_NAME=`basename $DATA_DIR`
if [ ! -d $LOG_DIR ]; then
    mkdir $LOG_DIR
fi
for fold in ${FOLDS[*]}
do
    echo 'SEED:'$seed
    OUTPUT_DIR=$PROJECT_ROOT/checkpoints_gene_wise/$DATA_NAME-$MODEL_TYPE-$IMAGE_ENCODER-$GENE_ENCODER_TYPE-dim$GENE_PROJECTION_DIM-mlp$NUM_PROJECTION_LAYERS-nlayer$N_LAYER-epoch$EPOCHS-lr$LR-dataset$DATASET-fold$fold-$seed
    RESUME_DIR=$OUTPUT_DIR/checkpoint-1950
    # MODEL_EVAL=/home/xy/GenePro/GeneQuery_new/checkpoints/her2st-gene_des_transformer-resnet50-mlp-dim256-mlp1-nlayer2-epoch100-lr0.0001-datasether2-fold0-35/checkpoint-231100
    CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
            --nnodes=1 \
            --nproc-per-node=$NUM_DEVICES \
            $PROJECT_ROOT/src/main.py \
            --model_type $MODEL_TYPE \
            --gene_dim 256 \
            --gene_num 54683 \
            --gene_total 54683 \
            --image_encoder_name $IMAGE_ENCODER \
            --num_classes 0 \
            --dataset_fold $fold \
            --dataset $DATASET \
            --global_pool avg \
            --pretrained_image_encoder \
            --trainable_image_encoder \
            --image_embedding_dim 2048 \
            --gene_encoder_type $GENE_ENCODER_TYPE \
            --gene_embedding_dim 768 \
            --gene_max_length 723 \
            --n_layers $N_LAYER \
            --train_ratio 0.9 \
            --num_projection_layers $NUM_PROJECTION_LAYERS \
            --gene_projection_dim $GENE_PROJECTION_DIM \
            --gene_dropout 0.3 \
            --fuse_method add \
            --data_dir $DATA_DIR \
            --do_eval \
            --do_train \
            --output_dir $OUTPUT_DIR \
            --per_device_train_batch_size $BATCH_SIZE \
            --per_device_eval_batch_size $BATCH_SIZE \
            --lr_scheduler_type constant_with_warmup \
            --num_train_epochs $EPOCHS \
            --learning_rate $LR \
            --gradient_accumulation_steps 1 \
            --save_strategy epoch \
            --eval_strategy epoch \
            --logging_strategy epoch \
            --save_steps 500 \
            --seed $seed 
done

# --model_name_or_path $MODEL_EVAL \
