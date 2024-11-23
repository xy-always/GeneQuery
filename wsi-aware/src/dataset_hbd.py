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
from tqdm import tqdm
import math
from utils import normalization_logp


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HBDataset(torch.utils.data.Dataset):
    def __init__(self, 
            data_dir,
            image_path,
            mtx_path,
            gene_name_emb_path,
            gene_name_desc_path, 
            max_length,
            gene_total,
    ):
        self.data_dir = data_dir
        self.whole_image = cv2.imread(image_path)
        # print("==image===:", self.whole_image)
     
        self.reduced_matrix = sc.read_h5ad(mtx_path)
        # print(self.reduced_matrix)
        self.barcode_all = self.reduced_matrix.obs['xy'].astype(str)
        self.spatial_pos_x_all = self.reduced_matrix.obs['pixel_x']
        self.spatial_pos_y_all = self.reduced_matrix.obs['pixel_y']
        logger.info(f'spots number {len(self.barcode_all.values)}')
        self.max_length = max_length
        self.cls_token = gene_total
        self.num_spots = len(self.barcode_all)
        self.all_spots = self.get_all_spots()
        self.all_genename = self.get_all_genename(gene_name_emb_path)
        self.all_gene_des = self.get_all_gene_des(gene_name_desc_path)
        self.all_random = self.get_all_random()
        self.gene_ids = self.get_gene_ids()
        self.gene_exp = self.reduced_matrix.X
        self.pos_x, self.pos_y = self.get_pos()

    def get_pos(self):
        pos_x = []
        pos_y = []
        for i, barcode in enumerate(self.barcode_all):
            pos_x.append(math.floor(float(barcode.split('x')[0])))
            pos_y.append(math.floor(float(barcode.split('x')[1])))
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
        # print('origin size:', original_size)
        patch_size = (224, 224)


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
        print('number_batchs:', num_batches)
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
        lines = open(os.path.join(self.data_dir, 'HBD_chatgpt_definition.txt'), 'r').readlines()
        gene_ids = [line.split('####')[1] for line in lines]
        tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        gene_ids = tokenizer(gene_ids, max_length=512, padding=True, truncation=True, return_tensors="pt").input_ids
        return gene_ids

    def get_all_gene_des(self, gene_des_file):
        gene_name_des = np.array(np.load(gene_des_file))
        return gene_name_des

    def get_all_random(self):
        random_name = np.random.standard_normal(size=(785, 768))
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
        item['x'] = self.pos_x
        item['y'] = self.pos_y
        return item

    def __len__(self):
        return len(self.reduced_matrix.X[1])

def load_hbd_data(
        data_dir,
        image_max_length,
        gene_total,
        dataset_fold=0, 
        load_test=False,
        train_ratio=0.8,
):
    # slice 3 randomly chosen to be test and will be left out during training
    print("Building loaders")
    datasets = []
    test_datasets = []
    files = os.listdir(os.path.join(data_dir, 'image'))
    logger.info(f"files: {files}")
    random.shuffle(files, random.seed(42))
    logger.info(f"files after shuffle: {files}")
    total_files = len(files)
    # 5 fold cross validation
    fold_size = total_files // 5
    fold_start = int(dataset_fold) * fold_size
    fold_end = fold_start + fold_size if int(dataset_fold) < 4 else total_files
    test_files = files[fold_start:fold_end]
    logger.info(f"test_files: {test_files}")
    logger.info(f"test files number: {len(test_files)}")
    patient_group = []
    for _, f in enumerate(files):
        name = f.split('.')[0]
        mark = name[3:]
        patient_no = name[3:10]
        if patient_no not in patient_group:
            patient_group.append(patient_no)
    logger.info(f'patient group number: {len(patient_group)}')
    for i, f in enumerate(files):
        name = f.split('.')[0]
        mark = name[3:]
        patient_no = name[3:10]
        image = os.path.join(data_dir, 'image/'+f)
        matrix_path1 = os.path.join(data_dir, 'adata/' + mark + '_adata.h5ad1000.h5ad')
        matrix_path2 = os.path.join(data_dir, 'adata/' + mark.replace('BT', 'BC') + '_adata.h5ad1000.h5ad')
        if os.path.exists(matrix_path1):
            matrix = matrix_path1
        elif os.path.exists(matrix_path2):
            matrix = matrix_path2
        gene_name_emb_path=os.path.join(data_dir, 'gene_description_emb.npy')
        gene_name_desc_path=os.path.join(data_dir, 'hbd_gpt_description_emb.npy')
        d = HBDataset(
            data_dir = data_dir,
            image_path = image,
            mtx_path = matrix,
            gene_name_emb_path=gene_name_emb_path,
            gene_name_desc_path=gene_name_desc_path, 
            max_length=image_max_length,
            gene_total=gene_total,
        )
        if f in test_files:
            test_datasets.append(d)
        else:
            datasets.append(d)
        # if patient_no == patient_group[int(dataset_fold)]:
        #     logger.info(patient_no)
        #     test_datasets.append(d)
        # else:
        #     datasets.append(d)
    
    datasets = torch.utils.data.ConcatDataset(datasets)
    test_datasets = torch.utils.data.ConcatDataset(test_datasets)
    train_size = int(train_ratio * len(datasets))
    eval_size = len(datasets) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(datasets, [train_size, eval_size], generator=torch.Generator().manual_seed(42))
    logger.info("train/test split completed")
    logger.info("Finished building loaders")
    return train_dataset, test_datasets, test_datasets
