import os
import cv2
import pandas as pd
import torch
import logging
from sklearn.decomposition import TruncatedSVD
# from scipy.sparse import csr_matrix
import numpy as np
import torchvision.transforms.functional as TF
import random
from tqdm import tqdm
from PIL import Image
import scanpy as sc
import math
from torch.utils.data import DataLoader
from utils import fix_sc_normalize_truncate_padding, normalization_logp

from modules import ImageEncoder_resnet50
import sys


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Her2STDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, 
        mtx_path, 
        gene_name_emb_path,
        gene_name_desc_path, 
        max_length, 
        gene_total,
        retriever_dir=None, 
        nprobe=None, 
        topk=None, 
        retriever_device=None):
        whole_image = cv2.imread(image_path)
        # print("==image===:", self.whole_image)
        # self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep="\t").astype(str)
        # self.barcode = pd.read_csv(barcode_path,header=None)
        # self.reduced_matrix = sc.read(reduced_mtx_path).X.T  #cell x features
        self.reduced_matrix = sc.read_h5ad(mtx_path)
        # print(self.reduced_matrix)
        barcode_all = self.reduced_matrix.obs['x'].astype(str) + 'x' + self.reduced_matrix.obs['y'].astype(str)
        assert len(barcode_all) == self.reduced_matrix.shape[0]

        spatial_pos_x_all = self.reduced_matrix.obs['pixel_x']
        spatial_pos_y_all = self.reduced_matrix.obs['pixel_y']
        self.max_length = max_length
        self.cls_token = gene_total
        self.gene_name_emb = np.load(gene_name_emb_path)
        self.gene_name_desc = np.load(gene_name_desc_path)
        self.num_spots = len(barcode_all)
        self.image_features = []
        self.spatial_coords_list = []
        self.barcode_list = []
        self.rna_list = []
        self.input_list = []
        for idx in tqdm(range(self.num_spots), total=self.num_spots):
            barcode = barcode_all.values[idx]
            v1 = math.floor(float(spatial_pos_x_all.iloc[idx]))
            v2 = math.floor(float(spatial_pos_y_all.iloc[idx]))
            image = whole_image[(v1-112):(v1+112),(v2-112):(v2+112)]
            image = self.transform(self.pad_image(image))
            input = self.reduced_matrix[idx]
            rna_value, rna_gene, rna_mask = fix_sc_normalize_truncate_padding(input, self.max_length, self.cls_token, norm=False)
            self.image_features.append(image)
            self.rna_list.append([rna_value, rna_gene, rna_mask])
            self.spatial_coords_list.append([v1, v2])
            self.barcode_list.append(barcode)
            self.input_list.append(input)

        # self.sentence_model = SentenceTransformer('/root/autodl-tmp/wsy/models/bert-base-uncased/')
        # for para in self.sentence_model.parameters():
        #     para.requires_grad = False
        if retriever_dir is None:
            retriever = None
            self.neighbors = None

    def transform(self, image):
        # image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)
    
    def pad_image(self, image):
        original_size = image.shape
        patch_size = (224, 224)  

  
        padding_width = (patch_size[0] - original_size[0]) // 2
        padding_height = (patch_size[1] - original_size[1]) // 2

        if padding_width > 0 or padding_height > 0:
 
            padded_image = Image.new('RGB', (patch_size[0], patch_size[1]), (255, 255, 255))
         
            padded_image.paste(Image.fromarray(image), (padding_height, padding_width))
        else:
    
            padded_image = Image.fromarray(image)
        return padded_image

    def __getitem__(self, idx):
        item = {}
        item['image'] = torch.from_numpy(self.image_features[idx]).permute(2, 0, 1).float() #color channel first, then XY
        item['barcode'] = self.barcode_list[idx]
        item['spatial_coords'] = self.spatial_coords_list[idx]
        item['rna_list'] = np.array(self.rna_list[idx])
        item['normalization'] = normalization_logp(np.array(self.input_list[idx].X[0]))
        item['gene_name']  = self.gene_name_emb
        item['gene_name_des']  = self.gene_name_desc
        item['random_name'] = np.random.standard_normal(size=(785, 768))
        item['random_name'] = item['random_name'].astype(float)
        item['pro_list'] = item['rna_list']
        item['label'] = np.array(self.input_list[idx].X[0])
        if self.neighbors is not None:
            item['img_retrieved_neighbor'] = self.neighbors[self.barcode_list[idx]]
        return item

    def __len__(self):
        return self.num_spots


def load_her2_data(
    data_dir,
    gene_max_length,
    gene_total,
    dataset_fold=0,
    retriever_dir=None, 
    nprobe=None, 
    topk=None, 
    retriever_device=None, 
    load_test=False,
    train_ratio=0.8, 
):
    # slice 3 randomly chosen to be test and will be left out during training
    logger.info("Building Her2+ loaders")
    datasets = []
    test_dataset = []
    files = os.listdir(os.path.join(data_dir, 'images/HE'))
    random.shuffle(files, random.seed(42))
    logger.info(f"files after shuffle: {files}")
    # Split files into 5 folds
    total_files = len(files)
    fold_size = total_files // 5
    fold_start = int(dataset_fold) * fold_size
    fold_end = int(fold_start) + fold_size if int(dataset_fold) < 4 else total_files
    test_files = files[int(fold_start):int(fold_end)]
    logger.info(f"test_files: {test_files}")
    logger.info(f"test files number: {len(test_files)}")
    for _, f in enumerate(files):
        name = f.split('.')[0]
        patient_no = name[0]
        image = os.path.join(data_dir, 'images/HE/'+f)
        # pos = os.path.join(data_dir, 'spot-selectionsname/'+name+'_selection.tsv')
        # barcode = os.path.join(data_dir, 'spot-selectionsname/'+name+'_selection_barcode.csv' )
        matrix = os.path.join(data_dir, 'adata/' + name+'_adata.h5ad1000.h5ad')
        gene_name_emb_path=os.path.join(data_dir, 'her2_description.npy')
        gene_name_desc_path=os.path.join(data_dir, 'her2_gpt_description_emb.npy')
        d = Her2STDataset(image_path = image,
            mtx_path = matrix,
            gene_name_emb_path=gene_name_emb_path,
            gene_name_desc_path=gene_name_desc_path, 
            max_length=gene_max_length,
            gene_total=gene_total,
        )
        if f in test_files:
            test_dataset.append(d)
        else:
            datasets.append(d)

    datasets = torch.utils.data.ConcatDataset(datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_dataset)
    train_size = int(train_ratio * len(datasets))
    eval_size = len(datasets) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(datasets, [train_size, eval_size], generator=torch.Generator().manual_seed(42))
    logger.info(f'all train dataset_length: {len(datasets)}')
    logger.info(f'eval_dataset size: {len(eval_dataset)}')
    logger.info(f'test_dataset size: {len(test_dataset)}')
    logger.info("Finished loading Her2+ data")
    return train_dataset, eval_dataset, test_dataset
