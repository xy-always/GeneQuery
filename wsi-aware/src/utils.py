import numpy as np
import logging
logger = logging.getLogger(__name__)
import torch

def normalization(x, low=1e-8, high=1):
    MIN = min(x)
    MAX = max(x)
    if MAX == MIN:
        return np.zeros_like(x)

    x = (x-MIN)/(MAX-MIN+low) # zoom to (low, high)
    return x

def normalization_logp(x):
    x = np.log1p(x)
    x = normalization(x)
    return x


def compute_metrics_gse(eval_preds):
    preds, gold = eval_preds.predictions.squeeze(), eval_preds.label_ids
    mask = eval_preds.inputs
    mask = np.array(mask)
    results = {}
    gene_num = 3467
    if preds.shape[0] % gene_num != 0:
        return results
    n_slides = preds.shape[0] // gene_num
    logger.info(f"n_slides: {n_slides}")
    preds = preds.reshape(n_slides, gene_num, -1)
    preds = np.transpose(preds, (1, 0, 2))
    preds = preds.reshape(gene_num, -1)
    preds = np.transpose(preds, (1, 0))
    gold = gold.reshape(n_slides, gene_num, -1)
    gold = np.transpose(gold, (1, 0, 2))
    gold = gold.reshape(gene_num, -1)
    gold = np.transpose(gold, (1, 0))
    mask = mask.reshape(n_slides, gene_num, -1)
    mask = np.transpose(mask, (1, 0, 2))
    mask = mask.reshape(gene_num, -1)
    mask = np.transpose(mask)
    preds = list(filter(lambda x: mask[x[0]].sum() > 0, enumerate(preds)))
    preds = np.stack([x[1] for x in preds])

    gold = list(filter(lambda x: mask[x[0]].sum() > 0, enumerate(gold)))
    gold = np.stack([x[1] for x in gold])

    logger.info(f"preds: {preds.shape}")
    logger.info(f"gold: {gold.shape}")

    
    results = {}
    corr_spot = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        corr_spot[i] = np.corrcoef(preds[i,:], gold[i,:],)[0,1]
    corr_spot = corr_spot[~np.isnan(corr_spot)]
    logger.info(f"spot correlation shape: {corr_spot.shape}")
    mean_corr = np.mean(corr_spot)
    results["mean_spot_corr"] = mean_corr

    corr = np.zeros(preds.shape[1])
    for i in range(preds.shape[1]):
        corr[i] = np.corrcoef(preds[:,i], gold[:,i],)[0,1]


    max_corr = np.max(corr[~np.isnan(corr)])
    results["max_gene_corr"] = max_corr
    # corr1 = corr[~np.isnan(corr)]
    ind = np.argsort(np.sum(gold, axis=0))[-50:]
    corr_all = np.mean(corr[~np.isnan(corr)])
    heg_top50 = np.mean(corr[ind])
    results["mean_gene_corr_all"] = corr_all
    results["mean_gene_corr_heg_top50"] = heg_top50
    ind1 = np.argsort(np.var(gold, axis=0))[-50:]
    hvg_top50 = np.mean(corr[ind1])
    results["mean_gene_corr_hvg_top50"] = hvg_top50

    return results

def compute_metrics_her2(eval_preds):
    preds, gold = eval_preds.predictions.squeeze(), eval_preds.label_ids
    mask = eval_preds.inputs
    mask = np.array(mask)
   
    gene_num = 314
    results = {}
    if preds.shape[0] % gene_num != 0:
        return results
    n_slides = preds.shape[0] // gene_num
    logger.info(f"n_slides: {n_slides}")
    preds = preds.reshape(n_slides, gene_num, -1)
    preds = np.transpose(preds, (1, 0, 2))
    preds = preds.reshape(gene_num, -1)
    preds = np.transpose(preds, (1, 0))
    gold = gold.reshape(n_slides, gene_num, -1)
    gold = np.transpose(gold, (1, 0, 2))
    gold = gold.reshape(gene_num, -1)
    gold = np.transpose(gold, (1, 0))
    mask = mask.reshape(n_slides, gene_num, -1)
    mask = np.transpose(mask, (1, 0, 2))
    mask = mask.reshape(gene_num, -1)
    mask = np.transpose(mask)
    for i in range(mask.shape[0]):
        assert mask[i].sum() == mask.shape[1] or mask[i].sum() == 0
    preds = list(filter(lambda x: mask[x[0]].sum() > 0, enumerate(preds)))
    preds = np.stack([x[1] for x in preds])

    gold = list(filter(lambda x: mask[x[0]].sum() > 0, enumerate(gold)))
    gold = np.stack([x[1] for x in gold])

    logger.info(f"preds: {preds.shape}")
    logger.info(f"gold: {gold.shape}")
    
    corr_spot = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        corr_spot[i] = np.corrcoef(preds[i,:], gold[i,:],)[0,1]
    corr_spot = corr_spot[~np.isnan(corr_spot)]
    logger.info(f"spot correlation shape: {corr_spot.shape}")
    mean_corr = np.mean(corr_spot)
    results["mean_spot_corr"] = mean_corr

    corr = np.zeros(preds.shape[1])
    for i in range(preds.shape[1]):
        corr[i] = np.corrcoef(preds[:,i], gold[:,i],)[0,1]

    # corr = np.zeros(preds.shape[0])
    # for i in range(preds.shape[0]):
    #     corr[i] = np.corrcoef(preds[i,:], gold[i,:],)[0,1]

    # print(corr)
    max_corr = np.max(corr[~np.isnan(corr)])
    results["max_gene_corr"] = max_corr


    ind = np.argsort(np.sum(gold, axis=0))[-50:]
    corr_all = np.mean(corr[~np.isnan(corr)])
    heg_top50 = np.mean(corr[ind])
    results["mean_gene_corr_all"] = corr_all
    results["mean_gene_corr_heg_top50"] = heg_top50
    ind1 = np.argsort(np.var(gold, axis=0))[-50:]
    hvg_top50 = np.mean(corr[ind1])
    results["mean_gene_corr_hvg_top50"] = hvg_top50

    return results


def compute_metrics_hbd(eval_preds):
    preds, gold = eval_preds.predictions.squeeze(), eval_preds.label_ids
    mask = eval_preds.inputs
    mask = np.array(mask)
    print('preds shape: ', preds.shape)
    print('gold shape: ', gold.shape)
    print('mask shape: ', mask.shape)
    # adata = sc.read_h5ad(os.path.join(data_args.data_dir, "adata_update/A1_adata.h5ad1000.h5ad"))
    # marker_gene_list = [ "GNAS", "FASN"]
    gene_num = 723
    results = {}
    if preds.shape[0] % gene_num != 0:
        print('cannot divided by 723!')
        return results
    
    n_slides = preds.shape[0] // gene_num
    logger.info(f"n_slides: {n_slides}")
    preds = preds.reshape(n_slides, gene_num, -1)
    preds = np.transpose(preds, (1, 0, 2))
    preds = preds.reshape(gene_num, -1)
    preds = np.transpose(preds, (1, 0))
    gold = gold.reshape(n_slides, gene_num, -1)
    gold = np.transpose(gold, (1, 0, 2))
    gold = gold.reshape(gene_num, -1)
    gold = np.transpose(gold, (1, 0))
    mask = mask.reshape(n_slides, gene_num, -1)
    mask = np.transpose(mask, (1, 0, 2))
    mask = mask.reshape(gene_num, -1)
    mask = np.transpose(mask)
    preds = list(filter(lambda x: mask[x[0]].sum() > 0, enumerate(preds)))
    preds = np.stack([x[1] for x in preds])

    gold = list(filter(lambda x: mask[x[0]].sum() > 0, enumerate(gold)))
    gold = np.stack([x[1] for x in gold])

    logger.info(f"preds: {preds.shape}")
    logger.info(f"gold: {gold.shape}")
    
    corr_spot = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        corr_spot[i] = np.corrcoef(preds[i,:], gold[i,:],)[0,1]
    corr_spot = corr_spot[~np.isnan(corr_spot)]
    logger.info(f"spot correlation shape: {corr_spot.shape}")
    mean_corr = np.mean(corr_spot)
    results["mean_spot_corr"] = mean_corr

    corr = np.zeros(preds.shape[1])
    for i in range(preds.shape[1]):
        corr[i] = np.corrcoef(preds[:,i], gold[:,i],)[0,1]


    max_corr = np.max(corr[~np.isnan(corr)])
    results["max_gene_corr"] = max_corr

   
    ind = np.argsort(np.sum(gold, axis=0))[-50:]
    corr_all = np.mean(corr[~np.isnan(corr)])
    heg_top50 = np.mean(corr[ind])
    results["mean_gene_corr_all"] = corr_all
    results["mean_gene_corr_heg_top50"] = heg_top50
    ind1 = np.argsort(np.var(gold, axis=0))[-50:]
    hvg_top50 = np.mean(corr[ind1])
    results["mean_gene_corr_hvg_top50"] = hvg_top50
    return results


def compute_mean_std(x):
    return np.mean(x), np.std(x)
