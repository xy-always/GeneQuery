import os
import torch
import random
import argparse
import transformers
import numpy as np
from tqdm import tqdm
from modules import ImageEncoder_resnet50
from dataset_gse import load_gse_data

def extract_feature(dataset, output_dir, batch_size=1024):
    img_encoder = ImageEncoder_resnet50()
    img_encoder.eval()
    img_encoder.cuda()
    img_features = []
    gene_values = []
    num_batches = int(np.ceil(len(dataset) / batch_size))
    for batch_idx in tqdm(range(num_batches), total=num_batches):
        image = []
        barcodes = []
        for idx in range(batch_idx*batch_size, np.min(((batch_idx+1)*batch_size, len(dataset)))):
            barcodes.append(dataset[idx]['barcode'])
            image.append(dataset[idx]['image'])
            gene_values.append(dataset[idx]['label'])
        image = torch.stack(image, dim=0)
        image = image.cuda()
        if batch_idx == 0:
            print(f'Sample of barcodes\n{barcodes[:3]}')
            print(f'Image shape: {image[0].shape}\nSample of image\n{image[0]}')
        feature = img_encoder(image).detach().cpu().numpy()
        img_features.append(feature)
    img_features = np.concatenate(img_features, axis=0)
    gene_values = np.array(gene_values)
    
    # Save the features
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file_img = os.path.join(output_dir, "image_features.npy")
    np.save(output_file_img, img_features)
    
    output_file_value = os.path.join(output_dir, "gene_values.npy")
    np.save(output_file_value, gene_values)
    
def load_feature(output_dir):
    img_path = os.path.join(output_dir, "image_features.npy")
    value_path = os.path.join(output_dir, "gene_values.npy")

    img_features = np.load(img_path)
    gene_values = np.load(value_path)
    print(img_features.shape)
    print(gene_values.shape)
    print(f'Image features\n{img_features[:3]}')
    print(f'Gene values\n{gene_values[:3]}')
    return img_features, gene_values

def setup_seed(seed):
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--gene_max_length', type=int, default=3467)
    parser.add_argument('--gene_total', type=int, default=54683)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    setup_seed(args.seed)
    extract_feature(load_gse_data(args.data_dir, args.gene_max_length, args.gene_total), args.output_dir)
    load_feature(args.output_dir)