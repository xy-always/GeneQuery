import logging

from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_type: str = field(
        default="gene_query_name",
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
    pretrained_image_encoder: bool = field(
        default=True,
    )
    trainable_image_encoder: bool = field(
        default=True,
    )
    image_embedding_dim: int = field(
        default=512,
    )
    gene_encoder_type: str = field(
        default="mlp",   
    )
    gene_embedding_dim: int = field(
        default=768,
    )
    gene_max_length: int = field(
        default=3467,
    )
    num_gene_attention_heads: int = field(
        default=12,
    )
    num_projection_layers: int = field(
        default=1,
    )
    gene_projection_dim: int = field(
        default=256,
    )
    gene_intermediate_size: int = field(
        default=1024,
    )
    gene_dropout: float = field(
        default=0.1,
    )
    gene_layer_norm_eps: float = field(
        default=1e-12,
    )
    fuse_method: str = field(
        default="add",
    )
    retriever_dir: str = field(
        default=None,
    )
    nprobe: int = field(
        default=None,
    )
    topk: int = field(
        default=None,
    )
    retriever_device: int = field(
        default=-1,
    )
    lam: float = field(
        default=0.1,
    )
    use_gpt_description: bool = field(
        default=False,
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
    train_ratio: float = field(
        default=0.8,
    )
    dataset_fold: float = field(
        default=0,
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
    resume_epochs: int = field(
        default=0,
    )
    n_layers: int = field(
        default=2,
    )
    dim_head: int = field(
        default=64,
    )
