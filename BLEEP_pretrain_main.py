import os
import numpy as np
import torch.distributed
from tqdm import tqdm
import scipy.io as sio

import torch
from torch import nn
import torch.distributed as dist
import torch.utils.data.distributed

from dataset_pretrain import CLIPDataset
from models import CLIPModel, AugGeneQueryID, AugGeneQueryName, GeneQueryID, GeneQueryName,GeneQueryNameDes,STNet, GeneQueryTansformer, GeneQueryNameTansformer, CLIPModel_CLIP, CLIPModel_ViT,CLIPModel_ViT_L,CLIPModel_resnet101,CLIPModel_resnet152,CLIPModel_CLIP_Pretrain,CLIPModel_CLIP_CrossAtt, CLIPModel_CrossAtt_Multi
from utils import AvgMeter
from torch.utils.data import DataLoader

import scanpy as sc
import argparse
import sys
import scipy.stats as stats
import random

torch.distributed.init_process_group(backend='nccl', init_method='env://')

parser = argparse.ArgumentParser(description='DDP for CLIP')

parser.add_argument('--exp_name', type=str, default='clip_crossatt', help='')
parser.add_argument('--batch_size', type=int, default=20, help='')
parser.add_argument('--max_length', type=int, default=3468, help='')
parser.add_argument('--gene_total', type=int, default=54683, help='')
parser.add_argument('--max_epochs', type=int, default=20, help='')
parser.add_argument('--seed', type=int, default=100, help='')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--patience', type=int, default=2, help='patience for lr scheduler')
parser.add_argument('--temperature', type=int, default=1, help='')
parser.add_argument('--factor', type=float, default=0.5, help='factor for lr scheduler')
parser.add_argument('--image_embedding_dim', type=int, default=512, help='image embedding dimension')
parser.add_argument('--spot_embedding_dim', type=int, default=512, help='spot embedding dimension')
parser.add_argument('--num_projection_layers', type=int, default=1, help='number of projection layers')
parser.add_argument('--projection_dim', type=int, default=256, help='projection dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--pretrained', type=bool, default=True, help='use pretrained model')
parser.add_argument('--trainable', type=bool, default=True, help='image model trainable')
parser.add_argument('--model', type=str, default='clip_crossatt', help='')


parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--init_method', default='tcp://127.0.0.1:1111', type=str, help='')
# parser.add_argument('--dist-backend', default='gloo', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')

def normalization(x, low=1e-8, high=1):
    MIN = min(x)
    MAX = max(x)
    if MAX == MIN:
        return np.zeros_like(x)
    x = low + (x-MIN)/(MAX-MIN)*(high-low) # zoom to (low, high)
    return x

def build_loaders(args):
    # slice 3 randomly chosen to be test and will be left out during training
    print("Building loaders")
    
    dataset = CLIPDataset(image_path = "data/A1_Merged.tiff",
                spatial_pos_path = "data/update_data/A1_tissue_positions_list.csv",
                h5ad_path = "data/adata_update/A1_adata.h5ad1000.h5ad",
                barcode_path = "data/update_data/A1_barcodes.tsv",
                max_length=args.max_length,
                cls_token=args.gene_total-2)
    dataset2 = CLIPDataset(image_path = "data/B1_Merged.tiff",
                spatial_pos_path = "data/update_data/B1_tissue_positions_list.csv",
                h5ad_path = "data/adata_update/B1_adata.h5ad1000.h5ad",
                barcode_path = "data/update_data/B1_barcodes.tsv",
                max_length=args.max_length,
                cls_token=args.gene_total-2)
    dataset4 = CLIPDataset(image_path = "data/D1_Merged.tiff",
                spatial_pos_path = "data/update_data/D1_tissue_positions_list.csv",
                h5ad_path = "data/adata_update/D1_adata.h5ad1000.h5ad",
                barcode_path = "data/update_data/D1_barcodes.tsv",
                max_length=args.max_length,
                cls_token=args.gene_total-2)
    
    dataset = torch.utils.data.ConcatDataset([dataset, dataset2, dataset4])
    # dataset = torch.utils.data.ConcatDataset([dataset])
    print('dataset_length:', len(dataset))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    print(len(train_dataset), len(test_dataset))
    print("train/test split completed")

    # Set up distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) #by default, rank and world sizes are retrieved from env variables
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print("Finished building loaders")
    return train_loader, test_loader

def cleanup():
    dist.destroy_process_group()


def train_epoch(model, train_loader, optimizer, args, lr_scheduler=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    # print('data:', len(train_loader))

    for batch in tqdm_object:
        # batch = {k: v.to() for k, v in batch.items() if k == "image" or k == "rna_list" or k == ""}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()

        # for name, param in model.named_parameters():
        #     if not param.requires_grad or param.grad is None:
        #         print(f"detected unused parameter: {param}")
        #         print(f"detected unused parameter: {name}")
            # torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            # param.grad.data /= args.world_size

        optimizer.step()
        # if step == "batch":
        #   lr_scheduler.step()
        # print("loss:", loss)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=args.lr)

    return loss_meter


def test_epoch(model, test_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        # batch = {k: v.to() for k, v in batch.items() if k == "image" or k == "reduced_expression"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def train(model, train_loader, optimizer, args, lr_scheduler=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    # print('data:', len(train_loader))

    for batch in tqdm_object:
        # batch = {k: v.to() for k, v in batch.items() if k == "image" or k == "rna_list" or k == ""}
        loss, _ = model(batch)
        optimizer.zero_grad()
        loss.backward()

        # for name, param in model.named_parameters():
        #     if not param.requires_grad or param.grad is None:
        #         print(f"detected unused parameter: {param}")
        #         print(f"detected unused parameter: {name}")
            # torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            # param.grad.data /= args.world_size

        optimizer.step()
        # if step == "batch":
        #   lr_scheduler.step()
        # print("loss:", loss)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=args.lr)

    return loss_meter

def test(model, test_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    gold = []
    preds = []
    for batch in tqdm_object:
        # batch = {k: v.to() for k, v in batch.items() if k == "image" or k == "reduced_expression"}
        loss, pred = model(batch)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    #     x = batch["rna_list"]
    #     rna_value = x[:,0,:][:,1:]
    #     pred = pred.cpu().detach().numpy()
    #     pred[pred < 0] = 0
    #     print(rna_value.shape)
    #     print(pred.shape)
    #     preds.extend(pred.squeeze())
    #     gold.extend(rna_value.squeeze().cpu().detach().numpy())
    # preds = np.array(preds)
    # gold = np.array(gold)
    # # print('pred:', preds.shape)
    # # print('gold:',gold.shape)
    # pearson = np.corrcoef(preds, gold)[0,1]
    # print('evaluation pearson:', pearson)
    return loss_meter, 0

def evaluate(model, args):
    dataset3 = CLIPDataset(image_path = "data/C1_Merged.tiff",
                spatial_pos_path = "data/update_data/C1_tissue_positions_list.csv",
                h5ad_path = "data/adata_update/C1_adata.h5ad1000.h5ad",
                barcode_path = "data/update_data/C1_barcodes.tsv",
                max_length=args.max_length,
                cls_token=args.gene_total-2)
    test_loader = DataLoader(dataset3, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    preds = []
    adata = sc.read_h5ad("data/adata_update/C1_adata.h5ad1000.h5ad")
    # adata_all = sc.read_h5ad("data/adata_update/C1_adata.h5ad")
    # gold_all = adata_all.X
    # gold = adata.X[:, 1000:]
    gold = adata.X
    marker_gene_list = [ "HAL", "CYP3A4", "VWF", "SOX9", "KRT7", "ANXA4", "ACTA2", "DCN"] #markers from macparland paper
    # marker_gene_list = ["AQP1", "KRT7", "CD24", "PIGR", "ANXA4"]
    genes = adata.var['hugo symbol'].values
    print(len(test_loader))
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            _, pred = model(batch)
            pred = pred.cpu().detach().numpy()
            pred = np.array(pred)
            pred[pred < 0] = 0
            preds.extend(pred.squeeze())
        preds = np.array(preds)
    # gold = [np.log1p(normalization(g)) for g in gold]
    # gold = np.array(gold)
    # gold = np.normalize(gold, axis=1)
    # gold = np.log1p(gold)
    # print(preds)
    # print(gold)
    
    # np.save('pred_bleep_geneid_1000.npy', preds)
    # preds = np.load('pred_bleep_geneid.npy')
    corr = np.zeros(preds.shape[0])
    gold = gold[:preds.shape[0], :]
    for i in range(preds.shape[0]):
        corr[i] = np.corrcoef(preds[i,:], gold[i,:],)[0,1]
    corr = corr[~np.isnan(corr)]
    print("Mean correlation across cells: ", np.mean(corr))

    corr = np.zeros(preds.shape[1])
    for i in range(preds.shape[1]):
        corr[i] = np.corrcoef(preds[:,i], gold[:,i],)[0,1]
    print("max correlation: ", np.max(corr))

    # marker_gene_ind = np.zeros(len(marker_gene_list))
    # for i, gene in enumerate(marker_gene_list):
    #     marker_gene_ind[i] = np.where(genes == gene)[0]
    #     print(corr[marker_gene_ind[i].astype(int)])
    #     # marker_gene_ind[i] = np.where(genes == marker_gene_list[i])[0]
    # print("mean correlation marker genes: ", np.mean(corr[marker_gene_ind.astype(int)]))

    corr2 = corr[~np.isnan(corr)]
    top_corr_indices = np.argsort(corr2)[-5:]
    top_corr_values = corr2[top_corr_indices]
    print("Top 5 max correlation values:", top_corr_values)
    print("Indices of top 5 max correlation values:", top_corr_indices)
    print('top 5 genes:', genes[top_corr_indices])

    corr1 = corr[~np.isnan(corr)]
    ind = np.argsort(np.sum(gold, axis=0))[-50:]
    print("mean correlation all expressed genes: ", np.mean(corr1))
    print("mean correlation highly expressed genes: ", np.mean(corr[ind]))
    ind1 = np.argsort(np.var(gold, axis=0))[-50:]
    print("mean correlation highly variable genes: ", np.mean(corr[ind1]))


def evaluate_unseen(model, args):
    dataset3 = CLIPDataset(image_path = "data/C1_Merged.tiff",
                spatial_pos_path = "data/update_data/C1_tissue_positions_list.csv",
                h5ad_path = "data/adata_update/C1_adata.h5ad",
                barcode_path = "data/update_data/C1_barcodes.tsv",
                max_length=args.max_length,
                cls_token=args.gene_total-2)
    test_loader = DataLoader(dataset3, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    preds = []
    adata = sc.read_h5ad("data/adata_update/C1_adata.h5ad1000.h5ad")
    adata_all = sc.read_h5ad("data/adata_update/C1_adata.h5ad")
    gold_all = adata_all.X
    gold = adata.X
   
    genes = adata.var['hugo symbol'].values

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            _, pred = model(batch)
            pred = pred.cpu().detach().numpy()
            pred = np.array(pred)
            pred[pred < 0] = 0
            preds.extend(pred.squeeze())
        preds = np.array(preds)
    # gold = [np.log1p(normalization(g)) for g in gold]
    # gold = np.array(gold)
    # gold = np.normalize(gold, axis=1)
    # gold = np.log1p(gold)
    # print(preds)
    # print(gold)
    
    np.save('pred_bleep_geneid_unseen.npy', preds)
    # preds = np.load('pred_bleep_geneid.npy')
    corr = np.zeros(preds.shape[0])
    gold = gold[:preds.shape[0], :]
    for i in range(preds.shape[0]):
        corr[i] = np.corrcoef(preds[i,:], gold[i,:],)[0,1]
    corr = corr[~np.isnan(corr)]
    print("Mean correlation across cells: ", np.mean(corr))

    corr = np.zeros(preds.shape[1])
    for i in range(preds.shape[1]):
        corr[i] = np.corrcoef(preds[:,i], gold[:,i],)[0,1]
    corr = corr[~np.isnan(corr)]
    print("max correlation: ", np.max(corr))

    # marker_gene_ind = np.zeros(len(marker_gene_list))
    # for i, gene in enumerate(marker_gene_list):
    #     marker_gene_ind[i] = np.where(genes == gene)[0]
    #     print(corr[marker_gene_ind[i].astype(int)])
    #     # marker_gene_ind[i] = np.where(genes == marker_gene_list[i])[0]
    # print("mean correlation marker genes: ", np.mean(corr[marker_gene_ind.astype(int)]))

    corr2 = corr[~np.isnan(corr)]
    top_corr_indices = np.argsort(corr2)[-5:]
    top_corr_values = corr2[top_corr_indices]
    print("Top 5 max correlation values:", top_corr_values)
    print("Indices of top 5 max correlation values:", top_corr_indices)
    print('top 5 genes:', genes[top_corr_indices])

    corr1 = corr[~np.isnan(corr)]
    ind = np.argsort(np.sum(gold, axis=0))[-50:]
    print("mean correlation all expressed genes: ", np.mean(corr1))
    print("mean correlation highly expressed genes: ", np.mean(corr[ind]))
    ind1 = np.argsort(np.var(gold, axis=0))[-50:]
    print("mean correlation highly variable genes: ", np.mean(corr[ind1]))

def setup_parallel():
    rank = torch.distributed.get_rank()
    local_rank = os.environ['LOCAL_RANK']
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.cuda.set_device(rank)
    return local_rank

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def main():
    print("Starting...")
    
    args = parser.parse_args()
    setup_seed(args.seed)
    # ngpus_per_node = torch.cuda.device_count()
    # local_rank = 0
    # rank = 0
    setup_parallel()
    print('start get model: ', args.model)
    current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if current_device.type == 'cuda':
        torch.cuda.set_device(current_device)
    args.current_device = current_device
  

    """ this block initializes a process group and initiate communications
		between all processes running on all nodes """

#     print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    #init the process group
    # dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
    # print("process group ready!")

    train_loader, test_loader = build_loaders(args)

    #make the model
#     print('From Rank: {}, ==> Making model..'.format(rank))
    # args.model = 'pretrained_clip'
    if args.model == "clip":
        model = CLIPModel_CLIP(args).to(current_device)
        print("Image encoder is CLIP")
    elif args.model == "pretrained_clip":
        spot_gene_model = torch.load('/root/autodl-tmp/xy/scTranslator/checkpoint/stage2_single-cell_scTranslator.pt')
        # spot_gene_model = Spot_model(dim=128,translator_depth=2,initial_dropout=0,enc_depth=2,enc_heads=8,enc_max_seq_len=20000,dec_depth=2,dec_heads=8,dec_max_seq_len=1000).to('cuda')
        # for n, p in spot_gene_model.named_parameters():
        #     print(n)
        # encoder_weights = {k: v for k, v in spot_pretrained_weight.named_parameters() if k.startswith('enc.')}
        # # print(encoder_weights)
        # spot_gene_model.load_state_dict(encoder_weights)
        for name, param in spot_gene_model.named_parameters():
            # print(name)
            param.requires_grad = False
        model = CLIPModel_CLIP_Pretrain(spot_pretrained_model=spot_gene_model).to(current_device)
        print("Image encoder is CLIP pretrained")
    elif args.model == "clip_crossatt":
        # model = CLIPModel_CLIP_CrossAtt(args).to(current_device)
        model = CLIPModel_CrossAtt_Multi(args).to(current_device)
        print("Image encoder is CLIP with Cross Attention")
    elif args.model == "gene_query_id":
        model = GeneQueryID(args).to(current_device)
        print("Image encoder with Gene Query")
    elif args.model == "gene_query_id_aug":
        model = AugGeneQueryID(args).to(current_device)
        print("Image encoder with Augmented Gene Query")
    elif args.model == "gene_query_name_aug":
        model = AugGeneQueryName(args).to(current_device)
        print("Image encoder with Augmented Gene Query Name")
    elif args.model == "gene_query_name":
        model = GeneQueryName(args).to(current_device)
        print("Image encoder with Gene Query")
    elif args.model == "gene_query_name_des":
        model = GeneQueryNameDes(args).to(current_device)
        print("Image encoder with Gene Name Description")
    elif args.model == 'gene_query_transformer':
        model = GeneQueryTansformer(args).to(current_device)
        print("Image encoder with Gene Query Transformer")
    elif args.model == 'gene_query_name_transformer':
        model = GeneQueryNameTansformer(args).to(current_device)
        print("Image encoder with Gene Name Transformer")
    elif args.model == "vit":
        model = CLIPModel_ViT(args).to(current_device)
        print("Image encoder is ViT")
    elif args.model == "vit_l":
        model = CLIPModel_ViT_L(args).to(current_device)
        print("Image encoder is ViT_L")
    elif args.model == "resnet101":
        model = CLIPModel_resnet101(args).to(current_device)
        print("Image encoder is ResNet101")
    elif args.model == "resnet152":
        model = CLIPModel_resnet152(args).to(current_device)
        print("Image encoder is ResNet152")
    elif args.model == 'stnet':
        model = STNet(args).to(current_device)
        print("Image encoder is STNet")
    else:
        model = CLIPModel(args).to(current_device)
        print("Image encoder is ResNet50")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[current_device])

    #load the data
    # print('From Rank: {}, ==> Preparing data..'.format(rank))
    
    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    # )
    
    
    # Train the model for a fixed number of epochs
    best_loss = float('inf')
    best_p = -100
    best_epoch = 0
    for epoch in range(args.max_epochs):
        print(f"Epoch: {epoch + 1}")
        # step = "epoch"

        # Train the model
        model.train()
        if args.model.startswith('gene_query') or args.model.startswith('stnet'):
            train_loss = train(model, train_loader, optimizer, args)
        else:
            train_loss = train_epoch(model, train_loader, optimizer, args)
        # Evaluate the model
        model.eval()
        with torch.no_grad():
            if args.model.startswith('gene_query') or args.model.startswith('stnet'):
                test_loss, pearson = test(model, test_loader)
            else:
                test_loss = test_epoch(model, test_loader)
        
        if test_loss.avg < best_loss:
        # if best_p < pearson:
            if not os.path.exists(str(args.exp_name)):
                os.mkdir(str(args.exp_name))
            best_loss = test_loss.avg
            # best_p = pearson
            best_epoch = epoch
            if args.model.startswith('gene_query') or args.model.startswith('stnet'):
                evaluate(model, args)
            torch.save(model.state_dict(), str(args.exp_name) + "/" + str(args.seed) + '_' + str(args.model) + ".pt")
            print("Saved Best Model! Loss: {}".format(best_loss))

    print("Done!, final loss: {}".format(best_loss))
    print("Best epoch: {}".format(best_epoch))
    model = torch.load(str(args.exp_name) + "/" + str(args.seed) + '_' + str(args.model) + ".pt")
    evaluate(model, args)
    cleanup()

if __name__ == "__main__":
    main()
    # args = parser.parse_args()
    # setup_seed(args.seed)
    # args.current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = GeneQueryName(args).to('cuda')
    # state_dict = torch.load('/root/autodl-tmp/xy/BLEEP/gene_query_bleep_name/'+str(args.seed)+ '_gene_query_name.pt')
    # # state_dict = torch.load("/root/autodl-tmp/xy/BLEEP/gene_query_bleep_ID_unseen/45_gene_query_id.pt")
    # # state_dict = torch.load("/root/autodl-tmp/xy/BLEEP/resnet50_bleep_update/stnet.pt")
    # new_state_dict = {}
    # for key in state_dict.keys():
    #     new_key = key.replace('module.', '')  # remove the prefix 'module.'
    #     new_key = new_key.replace('well', 'spot') # for compatibility with prior naming
    #     new_state_dict[new_key] = state_dict[key]

    # model.load_state_dict(state_dict)
    # model.eval()
    # evaluate(model, args)


