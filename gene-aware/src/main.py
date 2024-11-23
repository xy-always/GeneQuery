import os
import sys
import torch
import random
import logging
import transformers
import torch.utils.data.distributed
import numpy as np
import scanpy as sc

from dataset_gse import load_gse_data
from dataset_her2 import load_her2_data
from dataset_hbd import load_hbd_data
from models import (
    GeneConfig,
    AugGeneQueryID, 
    GeneQueryID, 
    GeneQueryName, 
    GeneQueryNameDes,
    GeneDesTransformer,
    STNet,
    CLIP,
)
from transformers import HfArgumentParser, Trainer
from args import (
    ModelArguments,
    GeneTrainingArguments, 
    GeneDataTrainingArguments,
)
from typing import Any, Dict, List, Mapping

logger = logging.getLogger(__name__)

torch.distributed.init_process_group(backend='nccl', init_method='env://')

def gene_data_collator(features):
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    if "barcode" in first and first["barcode"] is not None:
        batch["barcode"] = [f["barcode"] for f in features]

    for k, v in first.items():
        if k not in ("barcode") and v is not None and not isinstance(v, str):
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
    # training_args.num_train_epochs -= training_args.resume_epochs

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
        pretrained_image_encoder=model_args.pretrained_image_encoder,
        trainable_image_encoder=model_args.trainable_image_encoder,
        image_embedding_dim=model_args.image_embedding_dim,
        gene_encoder_type=model_args.gene_encoder_type,
        gene_embedding_dim=model_args.gene_embedding_dim,
        gene_max_length=model_args.gene_max_length,
        num_gene_attention_heads=model_args.num_gene_attention_heads,
        num_projection_layers=model_args.num_projection_layers,
        gene_projection_dim=model_args.gene_projection_dim,
        gene_intermediate_size=model_args.gene_intermediate_size,
        gene_dropout=model_args.gene_dropout,
        gene_layer_norm_eps=model_args.gene_layer_norm_eps,
        fuse_method=model_args.fuse_method,
        use_retriever=model_args.retriever_dir is not None,
        nprobe=model_args.nprobe,
        topk=model_args.topk,
        retriever_device=model_args.retriever_device,
        lam=model_args.lam,
        n_layers=training_args.n_layers,
        dim_head=training_args.dim_head,
        use_gpt_description=model_args.use_gpt_description,
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
    elif config.model_type == "gene_des_transformer":
        model_fn = GeneDesTransformer
    elif config.model_type == 'stnet':
        model_fn = STNet
    elif config.model_type == 'clip':
        model_fn = CLIP
    else:
        raise ValueError(f"Invalid model type: {config.model_type}")
    
    if model_args.model_name_or_path is not None:
        logger.info(f"Load model from {model_args.model_name_or_path}")
        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
    else:
        model = model_fn(config)

    if data_args.dataset == 'gse':
        train_dataset, eval_dataset, test_dataset = load_gse_data(
                                                        data_dir=data_args.data_dir, 
                                                        gene_max_length=model_args.gene_max_length, 
                                                        gene_total=model_args.gene_total, 
                                                        retriever_dir=model_args.retriever_dir, 
                                                        nprobe=model_args.nprobe, 
                                                        topk=model_args.topk, 
                                                        retriever_device=model_args.retriever_device, 
                                                        dataset_fold=data_args.dataset_fold,
                                                        train_ratio=data_args.train_ratio,
                                                    )
    elif data_args.dataset == 'hbd':
        train_dataset, eval_dataset, test_dataset = load_hbd_data(
                                                        data_dir=data_args.data_dir, 
                                                        gene_max_length=model_args.gene_max_length, 
                                                        gene_total=model_args.gene_total, 
                                                        dataset_fold=data_args.dataset_fold,
                                                        retriever_dir=model_args.retriever_dir, 
                                                        nprobe=model_args.nprobe, 
                                                        topk=model_args.topk, 
                                                        retriever_device=model_args.retriever_device, 
                                                        load_test=False, 
                                                        train_ratio=data_args.train_ratio,
                                                    )
    elif data_args.dataset == 'her2':
        train_dataset, eval_dataset, test_dataset = load_her2_data(
                                                        data_dir=data_args.data_dir, 
                                                        gene_max_length=model_args.gene_max_length, 
                                                        gene_total=model_args.gene_total, 
                                                        dataset_fold=data_args.dataset_fold,
                                                        retriever_dir=model_args.retriever_dir, 
                                                        nprobe=model_args.nprobe, 
                                                        topk=model_args.topk, 
                                                        retriever_device=model_args.retriever_device, 
                                                        load_test=False, 
                                                        train_ratio=data_args.train_ratio,
                                                    )

    def compute_metrics_gse(eval_preds):
        adata = sc.read_h5ad(os.path.join(data_args.data_dir, "adata_update/C1_adata.h5ad1000.h5ad"))
        preds, gold = eval_preds.predictions.squeeze(), eval_preds.label_ids

        results = {}
        # corr_spot = np.zeros(preds.shape[0])
        # for i in range(preds.shape[0]):
        #     corr_spot[i] = np.corrcoef(preds[i,:], gold[i,:],)[0,1]
        # corr_spot = corr_spot[~np.isnan(corr_spot)]
        # logger.info(f"spot correlation shape: {corr_spot.shape}")
        # mean_corr = np.mean(corr_spot)
        # results["mean_spot_corr"] = mean_corr

        corr = np.zeros(preds.shape[1])
        for i in range(preds.shape[1]):
            corr[i] = np.corrcoef(preds[:,i], gold[:,i],)[0,1]
      
        corr1 = corr[~np.isnan(corr)]
        ind = np.argsort(np.sum(gold, axis=0))[-50:]
        corr_all = np.mean(corr1)
        heg_top50 = np.mean(corr[ind])
        results["mean_gene_corr_all"] = corr_all
        results["mean_gene_corr_heg_top50"] = heg_top50
        ind1 = np.argsort(np.var(gold, axis=0))[-50:]
        hvg_top50 = np.mean(corr[ind1])
        results["mean_gene_corr_hvg_top50"] = hvg_top50
        return results
    
    def compute_metrics_her2(eval_preds):
        preds, gold = eval_preds.predictions.squeeze(), eval_preds.label_ids
        logger.info(f'preds shape: {preds.shape}')
        logger.info(f'gold shape: {gold.shape}')
        results = {}
        corr = np.zeros(preds.shape[1])
        for i in range(preds.shape[1]):
            corr[i] = np.corrcoef(preds[:,i], gold[:,i],)[0,1]

        corr1 = corr[~np.isnan(corr)]
        ind = np.argsort(np.sum(gold, axis=0))[-50:]
        corr_all = np.mean(corr1)
        heg_top50 = np.mean(corr[ind])
        results["mean_gene_corr_all"] = corr_all
        results["mean_gene_corr_heg_top50"] = heg_top50
        ind1 = np.argsort(np.var(gold, axis=0))[-50:]
        hvg_top50 = np.mean(corr[ind1])
        results["mean_gene_corr_hvg_top50"] = hvg_top50
        return results


    compute_metrics = {'her2': compute_metrics_her2, 'gse': compute_metrics_gse, 'hbd': compute_metrics_her2}
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=gene_data_collator,
        compute_metrics=compute_metrics[data_args.dataset],
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
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
        if trainer.is_world_process_zero():
            for k, v in results.items():
                logger.info(f"{k}: {v}")
            with open(os.path.join(training_args.output_dir, "eval_results.txt"), "w") as f:
                for k, v in results.items():
                    f.write(f"{k}: {v}\n")
                f.write(f"Training Arguments:\n")
                for k, v in training_args.__dict__.items():
                    f.write(f"{k}: {v}\n")
                f.write(f"Model Arguments:\n")
                for k, v in model_args.__dict__.items():
                    f.write(f"{k}: {v}\n")
                f.write(f"Data Arguments:\n")
                for k, v in data_args.__dict__.items():
                    f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()