import os
import sys
import torch
import random
import logging
import transformers
import torch.utils.data.distributed
import numpy as np

from dataset_gse import GSEDataset
from dataset_her2 import load_her2_data
from dataset_hbd import load_hbd_data
from models import (
    GeneConfig,
    AugGeneQueryID, 
    GeneQueryID, 
    GeneQueryName, 
    GeneQueryNameDes,
    GeneQueryDesMLP,
    GeneFuse,
)
from transformers import HfArgumentParser, Trainer
from args import (
    ModelArguments,
    GeneTrainingArguments, 
    GeneDataTrainingArguments,
)
from typing import Any, Dict, List, Mapping
import scanpy as sc
from utils import (
    compute_metrics_gse,
    compute_metrics_her2,
    compute_metrics_hbd,
    fix_sc_normalize_truncate_padding,
    norm_total,
    normalization,
)

logger = logging.getLogger(__name__)

torch.distributed.init_process_group(backend='nccl', init_method='env://')


def load_data(model_args, data_args):
    dataset = GSEDataset(data_dir = data_args.data_dir,
                image_path = os.path.join(data_args.data_dir, "A1_Merged.tiff"),
                spatial_pos_path = os.path.join(data_args.data_dir, "update_data/A1_tissue_positions_list.csv"),
                h5ad_path = os.path.join(data_args.data_dir, "adata_update/A1_adata.h5ad1000.h5ad"),
                barcode_path = os.path.join(data_args.data_dir, "update_data/A1_barcodes.tsv"),
                max_length=model_args.wsi_max_length,
                gene_name_file = data_args.gene_name_file,
                gene_des_file = data_args.gene_des_file,
                )
    dataset2 = GSEDataset(data_dir = data_args.data_dir,
                image_path = os.path.join(data_args.data_dir, "B1_Merged.tiff"),
                spatial_pos_path = os.path.join(data_args.data_dir, "update_data/B1_tissue_positions_list.csv"),
                h5ad_path = os.path.join(data_args.data_dir, "adata_update/B1_adata.h5ad1000.h5ad"),
                barcode_path = os.path.join(data_args.data_dir, "update_data/B1_barcodes.tsv"),
                max_length=model_args.wsi_max_length,
                gene_name_file = data_args.gene_name_file,
                gene_des_file = data_args.gene_des_file,
                )
    dataset4 = GSEDataset(data_dir=data_args.data_dir,
                image_path = os.path.join(data_args.data_dir, "D1_Merged.tiff"),
                spatial_pos_path = os.path.join(data_args.data_dir, "update_data/D1_tissue_positions_list.csv"),
                h5ad_path = os.path.join(data_args.data_dir, "adata_update/D1_adata.h5ad1000.h5ad"),
                barcode_path = os.path.join(data_args.data_dir, "update_data/D1_barcodes.tsv"),
                max_length=model_args.wsi_max_length,
                gene_name_file = data_args.gene_name_file,
                gene_des_file = data_args.gene_des_file,
                )
    
    dataset = torch.utils.data.ConcatDataset([dataset, dataset2, dataset4])
    test_dataset = GSEDataset(data_dir = data_args.data_dir,
                image_path = os.path.join(data_args.data_dir, "C1_Merged.tiff"),
                spatial_pos_path = os.path.join(data_args.data_dir, "update_data/C1_tissue_positions_list.csv"),
                h5ad_path = os.path.join(data_args.data_dir, "adata_update/C1_adata.h5ad1000.h5ad"),
                barcode_path = os.path.join(data_args.data_dir, "update_data/C1_barcodes.tsv"),
                max_length=model_args.wsi_max_length,
                gene_name_file = data_args.gene_name_file,
                gene_des_file = data_args.gene_des_file,
                )
    print("Finished loading all files")
    

    train_size = int(data_args.train_ratio * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42))
    return train_dataset, test_dataset, test_dataset
    # return dataset, dataset, dataset

def gene_data_collator(features):
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]

    batch = {}

    if "id" in first and first["id"] is not None:
        batch["id"] = [f["id"] for f in features]

    for k, v in first.items():
        if k not in ("id") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    return batch

def setup_parallel():
    rank = torch.distributed.get_rank()
    local_rank = os.environ['LOCAL_RANK']
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    logger.info(f"Rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
    torch.cuda.set_device(rank)
    return local_rank

def setup_seed(seed):
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = HfArgumentParser((ModelArguments, GeneDataTrainingArguments, GeneTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    transformers.logging.set_verbosity_info()
    logger.warning(
        "Process rank: %s, world size: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.world_size,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    
    # Set seed and parallel
    setup_seed(training_args.seed)
    setup_parallel()

    logger.info(f"Load model: {model_args.model_name_or_path}")
    config = GeneConfig(
        model_type=model_args.model_type,
        gene_dim=model_args.gene_dim,
        gene_num=model_args.gene_num,
        gene_total=model_args.gene_total,
        image_encoder_name=model_args.image_encoder_name,
        num_classes=model_args.num_classes,
        global_pool=model_args.global_pool,
        pretrained=model_args.pretrained,
        trainable=model_args.trainable,
        image_embedding_dim=model_args.image_embedding_dim,
        num_projection_layers=model_args.num_projection_layers,
        projection_dim=model_args.projection_dim,
        gene_embedding_dim=768,
        gene_max_length=3467,
        dropout=0.1,
        fuse_method=model_args.fuse_method,
        topk=data_args.topk,
        lam=training_args.lam,
        n_layers=training_args.n_layers,
        dim_head=training_args.dim_head,
    )

    logger.info(f"Initialize model: {model_args.model_type}")
    if config.model_type == "gene_query_id":
        # TODO: debugging
        model_fn = GeneQueryID
    elif config.model_type == "gene_query_id_aug":
        # TODO: debugging
        model_fn = AugGeneQueryID
    elif config.model_type == "gene_query_name":
        model_fn = GeneQueryName
    elif config.model_type == "gene_query_des":
        model_fn = GeneQueryNameDes
    elif config.model_type == "gene_cross_fuse":
        model_fn = GeneFuse
    
    
    if model_args.model_name_or_path is not None:
        logger.info(f"Load model from {model_args.model_name_or_path}")
        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
    else:
        model = model_fn(config)

    if data_args.dataset == 'gse':
        train_dataset, eval_dataset, test_dataset = load_data(model_args, data_args)
    elif data_args.dataset == 'her2':
        train_dataset, eval_dataset, test_dataset = load_her2_data(
                                                        data_dir=data_args.data_dir, 
                                                        image_max_length=model_args.wsi_max_length, 
                                                        gene_total=model_args.gene_total, 
                                                        dataset_fold=data_args.dataset_fold,
                                                        load_test=False, 
                                                        train_ratio=data_args.train_ratio,
                                                    )
    elif data_args.dataset == 'hbd':
        train_dataset, eval_dataset, test_dataset = load_hbd_data(
                                                        data_dir=data_args.data_dir, 
                                                        image_max_length=model_args.wsi_max_length, 
                                                        gene_total=model_args.gene_total, 
                                                        dataset_fold=data_args.dataset_fold,
                                                        load_test=False, 
                                                        train_ratio=data_args.train_ratio,
                                                        )

    
    compute_metrics = {'her2':compute_metrics_her2, 
                            'gse':compute_metrics_gse,
                            'hbd':compute_metrics_hbd}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=gene_data_collator,
        compute_metrics=compute_metrics[data_args.dataset],
    )

    if training_args.do_train:
        trainer.train()
        if training_args.save_at_last:
            logger.info("save model at last")
            final_checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-final")
            trainer.save_model(final_checkpoint_dir)
        
            model = model_fn.from_pretrained(
                final_checkpoint_dir,
                config=config,
            )
            model = model.to(training_args.device)
            trainer.model = model
    if training_args.do_eval:
        results = trainer.evaluate(eval_dataset=test_dataset)
        for k, v in results.items():
            logger.info(f"{k}: {v}")

if __name__ == "__main__":
    main()