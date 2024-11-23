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
import sys
from modules import ImageEncoder_resnet50
from transformers import AutoTokenizer, AutoModel
from utils import normalization_logp
from tqdm import tqdm
import math

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Her2STDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_path, 
        mtx_path, 
        gene_name_emb_path,
        gene_name_desc_path, 
        max_length, 
        gene_total):
        self.data_dir = data_dir
        self.whole_image = cv2.imread(image_path)
        # print("==image===:", self.whole_image)
        # self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep="\t").astype(str)
        # self.barcode = pd.read_csv(barcode_path,header=None)
        # self.reduced_matrix = sc.read(reduced_mtx_path).X.T  #cell x features
        self.reduced_matrix = sc.read_h5ad(mtx_path)
        # print(self.reduced_matrix)
        self.barcode_all = self.reduced_matrix.obs['x'].astype(str) + 'x' + self.reduced_matrix.obs['y'].astype(str)
        assert len(self.barcode_all) == self.reduced_matrix.shape[0]
        self.spatial_pos_x_all = self.reduced_matrix.obs['pixel_x']
        self.spatial_pos_y_all = self.reduced_matrix.obs['pixel_y']
        self.pos_x, self.pos_y = self.get_pos()
        self.max_length = max_length
        self.cls_token = gene_total
        self.num_spots = len(self.barcode_all)
        self.all_spots = self.get_all_spots()
        self.all_genename = self.get_all_genename(gene_name_emb_path)
        self.all_gene_des = self.get_all_gene_des(gene_name_desc_path)
        self.all_random = self.get_all_random()
        self.gene_ids = self.get_gene_ids()
        # print(self.gene_ids.shape)
        self.gene_ids = self.gene_ids[:314]
        # self.gene_ids = self.gene_ids
        self.gene_exp = self.reduced_matrix.X[:, :314]
        # self.gene_exp = self.reduced_matrix.X


    def get_pos(self):
        pos_x = []
        pos_y = []
        for i in range(len(self.barcode_all)):
            pos_x.append(math.floor(float(self.reduced_matrix.obs['x'].iloc[i])))
            pos_y.append(math.floor(float(self.reduced_matrix.obs['y'].iloc[i])))
        return pos_x, pos_y

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
    
    def enlarge_image(self, image):
        original_size = image.shape
        # 目标patch尺寸
        patch_size = (224, 224)  # 例如，100x100像素

        # 计算需要填充的宽度
        padding_width = (patch_size[0] - original_size[0]) // 2
        padding_height = (patch_size[1] - original_size[1]) // 2
        # print(padding_height, padding_width)
        # 如果宽度或高度小于0，则不需要填充
        if padding_width > 0 or padding_height > 0:
            # 创建一个填充后的图像
            padded_image = Image.new('RGB', (patch_size[0], patch_size[1]), (255, 255, 255))
            # print(type(image))
            # print('padd:', padded_image.size)
            # 将原始图像粘贴到新图像的中心
            padded_image.paste(Image.fromarray(image), (padding_height, padding_width))
        else:
            # 如果不需要填充，直接使用原始图像
            padded_image = Image.fromarray(image)
        return padded_image

    def get_all_spots(self):
        img_encoder = ImageEncoder_resnet50()
        batch_size = 100
        barcodes = self.barcode_all.values
        num_batches = int(np.ceil(len(barcodes) / batch_size))
        images = []

        def get_img_emb(idx):
            v1 = math.floor(float(self.spatial_pos_x_all.iloc[idx]))
            v2 = math.floor(float(self.spatial_pos_y_all.iloc[idx]))
            image = self.whole_image[(v1-112):(v1+112),(v2-112):(v2+112)]
            image = self.transform(self.enlarge_image(image))
            image = torch.tensor(image).permute(2, 0, 1).float()
            return image
        img_encoder = img_encoder.to('cuda')
        for batch_idx in tqdm(range(num_batches), total=num_batches):
            img_emb = []
            for idx in range(batch_idx*batch_size, min((batch_idx+1)*batch_size, len(barcodes))):
                image = get_img_emb(idx)
                img_emb.append(image)
            img_emb = torch.stack(img_emb, dim=0).to('cuda')
            img_emb = img_encoder(img_emb)
            img_emb = img_emb.detach().cpu()
            images.extend(img_emb)
        
        images = torch.stack(images)
        return images
    
    def get_all_genename(self, gene_name_file):
        gene_name = np.array(np.load(gene_name_file))
        return gene_name

    def get_gene_ids(self):
        print('use gpt description...')
        # gene_ids = self.reduced_matrix.var['hugo symbol'].astype(str).values
        # print(gene_ids)
        lines = open(os.path.join(self.data_dir, 'her2_chatgpt_definition.txt'), 'r').readlines()
        gene_ids = [line.split('####')[1] for line in lines]
        tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        gene_ids = tokenizer(gene_ids, max_length=512, padding=True, truncation=True, return_tensors="pt").input_ids
        return gene_ids

        # gene_ids = self.reduced_matrix.var['hugo symbol'].astype(str).values
        # tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        # gene_ids = tokenizer(gene_ids.tolist(), max_length=20, padding=True, truncation=True, return_tensors="pt").input_ids
        # return gene_ids

    def get_all_gene_des(self, gene_des_file):
        gene_name_des = np.array(np.load(gene_des_file))
        return gene_name_des

    def get_all_random(self):
        random_name = np.random.standard_normal(size=(785, 768))
        random_name = random_name.astype(float)
        return random_name

    def pad_image(self, image, gene_exp, length):
        assert len(image.shape) == 2
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
        item['pos_x'] = self.pos_x
        item['pos_y'] = self.pos_y
        item['mask'] = item['mask'].astype(int)
        item['gene_name'] = self.all_genename[idx]
        item['gene_name_des'] = self.all_gene_des[idx]
        item['random_name'] = self.all_random[idx]
        return item
    
    def __len__(self):
        # return len(self.reduced_matrix.X[1])
        return len(self.reduced_matrix.X[1][:314])
    

def load_her2_data(
    data_dir,
    image_max_length,
    gene_total,
    dataset_fold=8, 
    load_test=False,
    train_ratio=0.8, 
):
    logger.info("Building Her2+ loaders")
    datasets = []
    test_dataset = []
    files = os.listdir(os.path.join(data_dir, 'images/HE'))
    logger.info(f"files: {files}")
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
        
        d = Her2STDataset(
            data_dir = data_dir,
            image_path = image,
            mtx_path = matrix,
            gene_name_emb_path=gene_name_emb_path,
            gene_name_desc_path=gene_name_desc_path, 
            max_length=image_max_length,
            gene_total=gene_total,
            )
        if f in test_files:
            test_dataset.append(d)
        else:
            datasets.append(d)

        # if patient_no == chr(ord('A') + int(dataset_fold)):
        #     logger.info(patient_no)
        #     test_dataset.append(d)
        # else:
        #     datasets.append(d)
    
    datasets = torch.utils.data.ConcatDataset(datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_dataset)
    train_size = int(train_ratio * len(datasets))
    eval_size = len(datasets) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(datasets, [train_size, eval_size], generator=torch.Generator().manual_seed(42))
    logger.info(f'all train dataset_length: {len(datasets)}')
    logger.info(f'eval_dataset size: {len(eval_dataset)}')
    logger.info(f'test_dataset size: {len(test_dataset)}')
    logger.info("Finished loading Her2+ data")
    return train_dataset, test_dataset, test_dataset

# def main():
#     data_dir = '/home/xy/GenePro/her2st'
#     dataset, val, test = load_her2_data(
#         data_dir=data_dir,
#         image_max_length=1000,
#         gene_total=50000,
#     )
    
#     for i in range(len(dataset)):
#         item = dataset[i]
#         print(len(item['image']))
#         print(item['image'][0].shape)
#         print(item['label'].shape)
#         print(item['gene_name'].shape)
#         print(item['gene_name_des'].shape)
#         print(item['random'].shape)
#         break

# if __name__ == '__main__':
#     main()