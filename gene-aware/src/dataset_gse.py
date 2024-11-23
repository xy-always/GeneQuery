import os
import sys
import cv2
import torch
import random
import logging
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import scanpy as sc
from tqdm import tqdm
from PIL import Image
from utils import fix_sc_normalize_truncate_padding, normalization_logp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_gse_data(
    data_dir, 
    gene_max_length, 
    gene_total, 
    retriever_dir=None, 
    nprobe=None, 
    topk=None, 
    retriever_device=None, 
    dataset_fold=0, 
    train_ratio=0.8
):
    dataset = []
    test_dataset = []

    files = os.listdir(os.path.join(data_dir))
    for file in files:
        if file.endswith('.tiff'):
            name = file.split('.')[0][:2]
            image_path = os.path.join(data_dir, file)
            spatial_pos_path = os.path.join(data_dir, "update_data/"+name+"_tissue_positions_list.csv")
            barcode_path = os.path.join(data_dir, "update_data/"+name+"_barcodes.tsv")
            h5ad_path = os.path.join(data_dir, "adata_update/"+name+"_adata.h5ad1000.h5ad")
            gene_name_emb_path = os.path.join(data_dir, "bleep_description.npy")
            gene_name_desc_path = os.path.join(data_dir, "gse_gpt_description_emb.npy")

            d = GSEDataset(
                image_path=image_path,
                spatial_pos_path=spatial_pos_path,
                barcode_path=barcode_path,
                h5ad_path=h5ad_path,
                gene_name_emb_path=gene_name_emb_path,
                gene_name_desc_path=gene_name_desc_path,
                max_length=gene_max_length,
                cls_token=gene_total-2,
                retriever_dir=retriever_dir,
                nprobe=nprobe,
                topk=topk,
                retriever_device=retriever_device,
            )
            
            test_name = chr(ord('A') + int(dataset_fold)) + '1'
            if name == test_name:
                logger.info('dataset_fold: {}'.format(dataset_fold))
                logger.info(f'test_dataset name: {test_name}')
                test_dataset.append(d)
            else:
                dataset.append(d)
    dataset = torch.utils.data.ConcatDataset(dataset)
    test_dataset = torch.utils.data.ConcatDataset(test_dataset)
    train_size = int(train_ratio * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42))
    return train_dataset, eval_dataset, test_dataset

class GSEDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        image_path, 
        spatial_pos_path, 
        barcode_path, 
        h5ad_path, 
        gene_name_emb_path,
        gene_name_desc_path,
        max_length, 
        cls_token, 
        retriever_dir=None, 
        nprobe=None, 
        topk=None, 
        retriever_device=None
    ):
        # Load the data
        whole_image = cv2.imread(image_path)
        spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header = None)
        barcode_tsv = pd.read_csv(barcode_path, sep="\t", header = None)
        self.reduced_matrix = sc.read_h5ad(h5ad_path)
        self.gene_name_emb = np.load(gene_name_emb_path)
        self.gene_name_desc = np.load(gene_name_desc_path)
        self.max_length = max_length
        self.cls_token = cls_token
        # Preprocess the data
        self.num_spots = len(barcode_tsv)
        self.position_list = spatial_pos_csv.loc[:,[2,3]].values
        self.postion_pixel = spatial_pos_csv.loc[:,[4,5]].values
        self.image_features = []
        self.spatial_coords_list = []
        self.barcode_list = []
        self.rna_list = []
        self.input_list = []
        for idx in tqdm(range(self.num_spots), total=self.num_spots):
            barcode = barcode_tsv.values[idx, 0]
            v1 = spatial_pos_csv.loc[spatial_pos_csv[0]==barcode, 4].values[0]
            v2 = spatial_pos_csv.loc[spatial_pos_csv[0]==barcode, 5].values[0]
            p_v1 = spatial_pos_csv.loc[spatial_pos_csv[0]==barcode, 2].values[0]
            p_v2 = spatial_pos_csv.loc[spatial_pos_csv[0]==barcode, 3].values[0]
            image = whole_image[(v1-112):(v1+112),(v2-112):(v2+112)]
            image = self.transform(image)
            input = self.reduced_matrix[idx]
            rna_value, rna_gene, rna_mask = fix_sc_normalize_truncate_padding(input, self.max_length, self.cls_token)
            self.image_features.append(image)
            self.rna_list.append([rna_value, rna_gene, rna_mask])
            self.spatial_coords_list.append([v1, v2])
            self.barcode_list.append(barcode)
            self.input_list.append(input)
        # Get the neighbors
        if retriever_dir is None:
            retriever = None
            self.neighbors = None

    def __getitem__(self, idx):
        item = {}
        item['image'] = torch.from_numpy(self.image_features[idx]).permute(2, 0, 1).float() #color channel first, then XY
        item['barcode'] = self.barcode_list[idx]
        item['spatial_coords'] = self.spatial_coords_list[idx]
        item['rna_list'] = np.array(self.rna_list[idx])
        item['normalization'] = normalization_logp(np.array(self.input_list[idx].X[0]))
        item['gene_name']  = self.gene_name_emb
        item['gene_name_des']  = self.gene_name_desc
        item['random_name'] = np.random.standard_normal(size=(3467, 768))
        item['random_name'] = item['random_name'].astype(float)
        item['pro_list'] = item['rna_list']
        item['label'] = np.array(self.input_list[idx].X[0])
        if self.neighbors is not None:
            item['img_retrieved_neighbor'] = self.neighbors[self.barcode_list[idx]]
        return item

    def __len__(self):
        return self.num_spots
    
    def transform(self, image):
        image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.array(image)