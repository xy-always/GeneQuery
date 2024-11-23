import logging

from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_type: str = field(
        default="gene_query_name",
        # choices=["gene_query_id", "gene_query_id_aug", "gene_query_name_aug", 
        #          "gene_query_name", "gene_query_name_des", "gene_query_transformer", 
        #          "gene_query_name_transformer", "stnet", "stnet_name", "stnet_des"],
    )
    model_name_or_path: str = field(
        default=None,
    )
    gene_dim: int = field(
        default=256,
    )
    gene_num: int = field(
        default=3467,
    )
    gene_total: int = field(
        default=54683,
    )
    image_encoder_name: str = field(
        default="resnet50",
        # choices=["resnet50", "resnet101"],
    )
    num_classes: int = field(
        default=0,
    )
    global_pool: str = field(
        default="avg",
    )
    pretrained: bool = field(
        default=True,
    )
    trainable: bool = field(
        default=True,
    )
    image_embedding_dim: int = field(
        default=512,
    )
    num_projection_layers: int = field(
        default=1,
    )
    projection_dim: int = field(
        default=256,
    )
    gene_embedding_dim: int = field(
        default=768,
    )
    wsi_max_length: int = field(
        default=2500,
    )
    dropout: float = field(
        default=0.1,
    )
    fuse_method: str = field(
        default="add",
    )

@dataclass
class GeneDataTrainingArguments:
    data_dir: str = field(
        default="",
    )
    image_path: str = field(
        default="",
    )
    matrix_path: str = field(
        default="",
    )
    dataset: str =  field(
        default='gse'
    )
    matrix_path: str = field(
        default="",
    )
    image_path: str = field(
        default="",
    )
    spot_path: str = field(
        default="",
    )
    retriever_dir: str = field(
        default="",
    )
    nprobe: int = field(
        default=512,
    )
    topk: int = field(
        default=5,
    )
    device_id: int = field(
        default=-1,
    )
    gene_name_file: str = field(
        default="",
    )
    gene_des_file: str = field(
        default="",
    )
    dataset_fold: str = field(
        default=0
    )
    train_ratio: float = field(
        default=0.9,
    )

@dataclass
class GeneTrainingArguments(TrainingArguments):
    max_epochs: int = field(
        default=20,
    )
    seed: int = field(
        default=100,
    )
    weight_decay: float = field(
        default=1e-3,
    )
    patience: int = field(
        default=2,
    )
    factor: float = field(
        default=0.5,
    )
    n_layers: int = field(
        default=2,
    )
    dim_head: int = field(
        default=64,
    )
    num_workers: int = field(
        default=0,
    )
    save_strategy: str = field(
        default="epoch",
    )
    logging_strategy: str = field(
        default="epoch",
    )
    eval_strategy: str = field(
        default="epoch",
    )
    metric_for_best_model: str = field(
        default="loss",
    )
    load_best_model_at_end: bool = field(
        default=True,
    )
    save_total_limit: int = field(
        default=10,
    )
    save_at_last: bool = field(
        default=False,
    )
    lam: float = field(
        default=0.1,
    )
    
