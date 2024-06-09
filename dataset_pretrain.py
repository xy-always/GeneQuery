import os
import cv2
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
# from scipy.sparse import csr_matrix
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
import scanpy as sc
from modules import ImageEncoder_resnet50

def normalization(x, low=1e-8, high=1):
    MIN = min(x)
    MAX = max(x)
    if MAX == MIN:
        return np.zeros_like(x)
    x = low + (x-MIN)/(MAX-MIN)*(high-low) # zoom to (low, high)
    return x

def fix_sc_normalize_truncate_padding(x, length, cls_token):
    '''
    x = (1, num_gene)

    '''
    len_x = len(x.X[0])
    # len_x = 2467
    # print("len_x:", len_x)
    tmp = [i for i in x.X[0]]
    # tmp = [i for i in x.X[0]]
    # tmp = normalization(tmp)
    # tmp = np.log1p(tmp)
    # print(tmp)
    if len_x >= length: # truncate
        x_value = [1e-8] + list(tmp[:length])
        # print("x:", x_value)
        gene = [cls_token] + list(x.var.iloc[:length]['uid'].astype(int).values)
        # gene = [cls_token] + list(x.var.iloc[1000:]['uid'].astype(int).values)
        mask = np.full(length+1, False)
        # print('gene len:', gene)
        # print(x_value)
    else: # padding
        x_value =[1e-8] + list(tmp)
        x_value.extend([1e-8 for i in range(length-len_x)])
        gene =[cls_token] + list(x.var['uid'].astype(int).values)
        # pad token is cls_token+1
        gene.extend([(cls_token+1) for i in range(length-len_x)])
        mask = np.concatenate((np.full(len_x+1,False), np.full(length-len_x,True)))
    # print(type(x_value), type(gene), type(mask))
    return np.array(x_value), np.array(gene), np.array(mask)

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_pos_path, barcode_path, h5ad_path, max_length, cls_token):
        #image_path is the path of an entire slice of visium h&e stained image (~2.5GB)
        
        #spatial_pos_csv
            #barcode name
            #detected tissue boolean
            #x spot index
            #y spot index
            #x spot position (px)
            #y spot position (px)

        #barcode_tsv
            #spot barcodes - alphabetical order

        self.whole_image = cv2.imread(image_path)
        # print("==image===:", self.whole_image)
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header = None) 
        # self.expression_mtx = csr_matrix(sio.mmread(expression_mtx_path)).toarray()
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header = None)
        self.reduced_matrix = sc.read_h5ad(h5ad_path)
        self.max_length = max_length
        self.cls_token = cls_token
        self.position_list = self.spatial_pos_csv.loc[:,[2,3]].values
        self.postion_pixel = self.spatial_pos_csv.loc[:,[4,5]].values
        # print(self.position_list)
        # print(self.postion_pixel)
        self.ImageEncoder_resnet50=ImageEncoder_resnet50()
        print("Finished loading all files")

    def transform(self, image):
        image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)
    
    # 得到position的邻居图片坐标
    def get_neighbor(self, position, position_list, pix_list):
        neighbors = []
        for i, p in enumerate(position_list):
            if abs(p[0] - position[0]) <= 2 and abs(p[1] - position[1]) <= 2:
                neighbors.append(pix_list[i])
        return neighbors

    def __getitem__(self, idx):
        item = {}
        # print('=='*4, idx)
        barcode = self.barcode_tsv.values[idx,0]
        v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,4].values[0]
        v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,5].values[0]
        p_v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,2].values[0]
        p_v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,3].values[0]
        # print(v1)
        # print(v2)
        # print("***:", self.whole_image.shape)
        image = self.whole_image[(v1-112):(v1+112),(v2-112):(v2+112)]
        # print(image.shape)
        image = self.transform(image)
        # neighbor = self.get_neighbor([p_v1,p_v2], self.position_list, self.postion_pixel)
        # # print("neighbor:", p_v1, p_v2, neighbor)
        # ne_imgs = []
        # for ne in neighbor:
        #     image = self.whole_image[(v1-112):(v1+112),(v2-112):(v2+112)]
        #     image = self.transform(image)
        #     ne_imgs.append(image)
        # ne_imgs = np.array(ne_imgs)
        # ne_imgs = torch.tensor(ne_imgs).permute(0,3,1,2).float()
        # print(self.reduced_matrix[idx])
        input = self.reduced_matrix[idx]
        rna_value, rna_gene, rna_mask = fix_sc_normalize_truncate_padding(input, self.max_length, self.cls_token)
        # print(rna_gene)
        # pro_value, pro_gene, pro_mask = fix_sc_normalize_truncate_padding(self.reduced_matrix[idx], 1000)
        item['image'] = torch.tensor(image).permute(2, 0, 1).float() #color channel first, then XY
        item['barcode'] = barcode
        item['spatial_coords'] = [v1,v2]
        # print('rna_value:', len(rna_gene))
        item['rna_list'] = np.array([rna_value, rna_gene, rna_mask])
        # print("rna shape:", item['rna_list'].shape)
        item['gene_name']  = np.array(np.load('/root/autodl-tmp/xy/BLEEP/data/gene_name_emb.npy'))
        item['gene_name_des']  = np.array(np.load("/root/autodl-tmp/xy/BLEEP/data/bleep_description.npy"))
        # item['gene_name_all'] = np.array(np.load('/root/autodl-tmp/xy/BLEEP/data/gene_name_all.npy'))
        item['random_name'] = np.random.standard_normal(size=(3467, 768))
        item['random_name'] = item['random_name'].astype(float)
        # item['neighbor'] = torch.mean(ne_imgs_encode, dim=0)
        # item['neighbor_raw'] = ne_imgs[:8]
        # print('ne_imgs_encode:', item['neighbor_raw'].shape)
        
        # print('rns_list:', len(rna_value))
        item['pro_list'] = item['rna_list']
        
        return item


    def __len__(self):
        return len(self.barcode_tsv)