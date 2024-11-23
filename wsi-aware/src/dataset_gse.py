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
import math
from modules import ImageEncoder_resnet50
from transformers import AutoTokenizer, AutoModel
from utils import normalization_logp


from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




class GSEDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_path, spatial_pos_path, barcode_path, h5ad_path, max_length, gene_name_file, gene_des_file,):
        self.data_dir = data_dir
        self.whole_image = cv2.imread(image_path)
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header = None) 
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header = None)
        self.reduced_matrix = sc.read_h5ad(h5ad_path)
        # print(self.reduced_matrix)
        self.all_spots = self.get_all_spots()
        self.all_genename = self.get_all_genename(gene_name_file)
        self.all_gene_des = self.get_all_gene_des(gene_des_file)
        self.all_random = self.get_all_random()
        self.gene_ids = self.get_gene_ids()
        self.gene_exp = self.reduced_matrix.X
        self.max_length = max_length



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

    def get_all_spots(self):
        img_encoder = ImageEncoder_resnet50()
        batch_size = 150
        barcodes = self.barcode_tsv.values
        num_batches = int(np.ceil(len(barcodes) / batch_size))
        images = []

        def get_img_emb(barcode):
            v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,4].values[0]
            v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,5].values[0]
            image = self.whole_image[(v1-112):(v1+112),(v2-112):(v2+112)]
            image = self.transform(image)
            image = torch.tensor(image).permute(2, 0, 1).float()
            return image

        img_encoder = img_encoder.to('cuda')
        for batch_idx in tqdm(range(num_batches), total=num_batches):
            batch_barcodes = barcodes[batch_idx*batch_size:(batch_idx+1)*batch_size, 0]
            img_emb = []
            for barcode in batch_barcodes:
                image = get_img_emb(barcode)
                img_emb.append(image)
            img_emb = torch.stack(img_emb, dim=0).to('cuda')
            img_emb = img_encoder(img_emb)
            img_emb = img_emb.detach().cpu()
            images.extend(img_emb)
        
        images = torch.stack(images)

        # image_encode = ImageEncoder_resnet50()
        # images = []
        # for i, barcode in enumerate(self.barcode_tsv.values):
        #     barcode = barcode[0]
        #     v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,4].values[0]
        #     v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,5].values[0]
        #     image = self.whole_image[(v1-112):(v1+112),(v2-112):(v2+112)]
        #     image = self.transform(image)
        #     image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
        #     images.append(image_encode(image))
        # images = torch.stack(images)
        return images

    def get_all_genename(self, gene_name_file):
        gene_name = np.array(np.load(gene_name_file))
        return gene_name
    def get_gene_ids(self):
        print('use gpt description...')
        # gene_ids = self.reduced_matrix.var['hugo symbol'].astype(str).values
        lines = open(os.path.join(self.data_dir, 'GSE_chatgpt_definition.txt'), 'r').readlines()
        gene_ids = [line.split('####')[1] for line in lines]
        tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        gene_ids = tokenizer(gene_ids, max_length=512, padding=True, truncation=True, return_tensors="pt").input_ids
        return gene_ids
    def get_all_gene_des(self, gene_des_file):
        gene_name_des = np.array(np.load(gene_des_file))
        return gene_name_des

    def get_all_random(self):
        random_name = np.random.standard_normal(size=(3467, 768))
        random_name = random_name.astype(float)
        return random_name

    def pad_image(self, image, gene_exp, length):
        mask = np.full(len(image), True)
        if len(image) >= length:
            return image[:length], gene_exp[:length], mask[:length]
        else:
            pad_len = length - len(image)
            for i in range(pad_len):
                gene_exp = np.append(gene_exp, 0)
            gene_exp = np.array(gene_exp)
            pad_image = [torch.zeros_like(image[0]) for i in range(length-len(image))]
            image = torch.cat((image, torch.stack(pad_image)), dim=0)
            mask = np.concatenate((mask, np.full(pad_len, False)), axis=0)
            return image, gene_exp, mask

    
    def __getitem__(self, idx):
        item = {}
        item['gene_ids'] = self.gene_ids[idx]
        item['id'] = idx
        item['image'], item['label'], item['mask'] = self.pad_image(self.all_spots, self.gene_exp[:, idx], self.max_length)
        item['normalization'] = normalization_logp(item['label'])
        item['mask'] = item['mask'].astype(int)
        item['gene_name'] = self.all_genename[idx]
        item['gene_name_des'] = self.all_gene_des[idx]
        item['random_name'] = self.all_random[idx]
        return item

    def __len__(self):
        return len(self.reduced_matrix.X[1])


# def main():
#     data_dir = '/l/users/ying.xiong/projects/data'
#     dataset = GSEDataset(image_path = os.path.join(data_dir, "A1_Merged.tiff"),
#                 spatial_pos_path = os.path.join(data_dir, "update_data/A1_tissue_positions_list.csv"),
#                 h5ad_path = os.path.join(data_dir, "adata_update/A1_adata.h5ad1000.h5ad"),
#                 barcode_path = os.path.join(data_dir, "update_data/A1_barcodes.tsv")
#                 )

#     for i in range(len(dataset)):
#         item = dataset[i]
#         print(len(item['image']))
#         print(item['image'][0].shape)
#         print(item['spots_gene'].shape)
#         print(item['gene_name'].shape)
#         print(item['gene_des'].shape)
#         print(item['random'].shape)
#         break

# if __name__ == '__main__':
#     main()