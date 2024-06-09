import sio
import scanpy as sc
import numpy as np
import os
import pandas as pd

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
# 读取文件中HVGs，并保存到新的文件中
def extract_hvg_data(exp_paths, out_dir):
    #only need to run once to save hvg_matrix.npy
    #filter expression matrices to only include HVGs shared across all datasets

    def hvg_selection_and_pooling(exp_paths, n_top_genes = 1000):
        #input n expression matrices paths, output n expression matrices with only the union of the HVGs

        #read adata and find hvgs
        hvg_bools = []
        for d in exp_paths:
            adata = sio.mmread(d)
            adata = adata.toarray()
            print(adata.shape)
            adata = sc.AnnData(X=adata.T, dtype=adata.dtype)

            # Preprocess the data
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            
            #save hvgs
            hvg = adata.var['highly_variable']
            
            hvg_bools.append(hvg)
        
        #find union of hvgs
        hvg_union = hvg_bools[0]
        for i in range(1, len(hvg_bools)):
            print(sum(hvg_union), sum(hvg_bools[i]))
            hvg_union = hvg_union | hvg_bools[i]

        print("Number of HVGs: ", hvg_union.sum())
        
        #filter expression matrices
        filtered_exp_mtxs = []
        for d in exp_paths:
            adata = sio.mmread(d)
            adata = adata.toarray()
            adata = adata[hvg_union]
            filtered_exp_mtxs.append(adata)

        return filtered_exp_mtxs

    # exp_paths = ["data/A1_matrix.mtx",
    #             "data/B1_matrix.mtx",
    #             "data/C1_matrix.mtx",
    #             "data/D1_matrix.mtx"]

    filtered_mtx = hvg_selection_and_pooling(exp_paths)
    
    for i in range(len(filtered_mtx)):
        np.save(os.path.join(out_dir, "hvg_" + str(i) + "_matrix.npy"), filtered_mtx[i])

# 读取文件中HVGs，并保存到新的文件中
def extract_her2_hvg(exp_paths, out_dir):
    #only need to run once to save hvg_matrix.npy
    #filter expression matrices to only include HVGs shared across all datasets

    def hvg_selection_and_pooling(exp_paths, n_top_genes = 50):
        #input n expression matrices paths, output n expression matrices with only the union of the HVGs

        #read adata and find hvgs
        hvg_bools = []
        for d in os.listdir(exp_paths):
            path = os.path.join(exp_paths, d)
            df = pd.read_csv(path)
           
            # 确保CSV文件至少有两列
            if len(df.columns) < 2:
                print("CSV文件中至少需要有两列，请检查文件。")
                exit()

            # 将除去第一列的所有列转换为NumPy数组
            # 这里我们假设第一列是索引列，所以我们从第二列开始选择
            array_2d = df.iloc[:, 1:].values
            # print(array_2d)
            adata = sc.AnnData(X=array_2d)
            print(adata)
            # Preprocess the data
            # sc.pp.normalize_total (adata)
            # sc.pp.log1p(adata)
            # sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            
            #save hvgs
            hvg = adata.var['highly_variable']
            
            hvg_bools.append(hvg)
        
        #find union of hvgs
        hvg_union = hvg_bools[0]
        for i in range(1, len(hvg_bools)):
            print(sum(hvg_union), sum(hvg_bools[i]))
            hvg_union = hvg_union | hvg_bools[i]

        print("Number of HVGs: ", hvg_union.sum())
        
        #filter expression matrices
        filtered_exp_mtxs = []
        filenames = []
        for d in os.listdir(exp_paths):
            filenames.append(d.split('.')[0])
            path = os.path.join(exp_paths, d)
            df = pd.read_csv(path)
            # print(path)
            # 确保CSV文件至少有两列
            if len(df.columns) < 2:
                print("CSV文件中至少需要有两列，请检查文件。")
                exit()

            # 将除去第一列的所有列转换为NumPy数组
            # 这里我们假设第一列是索引列，所以我们从第二列开始选择
            array_2d = df.iloc[:, 1:].values
            # print(array_2d)
            adata = sc.AnnData(X=array_2d)
            # print(adata)
            adata = adata[:, hvg_union].copy()
            filtered_exp_mtxs.append(adata.X)

        return filtered_exp_mtxs, filenames

    # exp_paths = ["data/A1_matrix.mtx",
    #             "data/B1_matrix.mtx",
    #             "data/C1_matrix.mtx",
    #             "data/D1_matrix.mtx"]

    filtered_mtx, filenames = hvg_selection_and_pooling(exp_paths)
    
    for i in range(len(filtered_mtx)):
        np.save(os.path.join(out_dir, filenames[i] + "_matrix.npy"), filtered_mtx[i])

def extract_barcode(dir_name, out_dir):
    for file in os.listdir(dir_name):
        path = os.path.join(dir_name, file)
        print(path)
        df = pd.read_csv(path, sep='\t')
        # print(df)
        # 确保CSV文件至少有一列
        if len(df.columns) < 2:
            print("CSV文件中没有列，请检查文件。")
            exit()

        # 获取第一列的数据
        x = df.iloc[:, 0].astype(str)
        y = df.iloc[:, 1].astype(str)
        concatenated_data = x + 'x' + y

        # 将第一列的数据写入到新的CSV文件
        new_csv_file_path = os.path.join(out_dir, file.split('.')[0] + '_barcode.csv')  # 新CSV文件的路径
        concatenated_data.to_csv(new_csv_file_path, index=False, header=False)

        print(f"数据已写入到 {new_csv_file_path}")


def uniform_gene(dir_name, out_dir):

    # Read all files and combine unique genes
    all_genes = set()
    for p in os.listdir(dir_name):
        print(p)
        file_path = os.path.join(dir_name, p)
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        all_genes.update(df.columns)
    print(len(all_genes))
    # Sort genes in the order they appear in the first file
    file_paths=os.listdir(dir_name)
    first_file_path = file_paths[0]
    first_df = pd.read_csv(os.path.join(dir_name,first_file_path), sep='\t', index_col=0)
    sorted_genes = [gene for gene in first_df.columns if gene in all_genes]
    print(len(sorted_genes))
    # Update each file with missing genes
    for p in os.listdir(dir_name):
        file_path = os.path.join(dir_name, p)
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        missing_genes = [gene for gene in sorted_genes if gene not in df.columns]
        for gene in missing_genes:
            df[gene] = 0
        df = df[sorted_genes]  # Reorder columns
        df.to_csv(os.path.join(out_dir, 'new_'+ p), index=True)


## 构造bleep数据成h5ad格式，把数据中的top1000 hvg保留下来，并且加上id
class BleepData:
    def __init__(self, data_dir, new_data_dir, hvg_path):
        self.data_dir = data_dir
        self.new_data_dir = new_data_dir
        self.hvg_path = hvg_path

    def process(self):
        # list all files in dir
        for p in os.listdir(self.data_dir):
            if p.endswith('h5ad'):
                path = os.path.join(self.data_dir, p)
                new_path = os.path.join(self.new_data_dir, 'test_'+p)
                adata = sc.read_h5ad(path)
                adata.var['Hugo_symbol'] = deepcopy(adata.var['feature'])
                adata.var['NCBI_Gene_ID'] = deepcopy(adata.var['Hugo_symbol'])
                adata.var['my_Id'] = deepcopy(adata.var['Hugo_symbol'])
                for i in adata.var.index:
                    # print(i)
                    # print(adata.var.loc[i,'Hugo_symbol'])
                    # print(hgs2hgid.get(adata.var.loc[i,'Hugo_symbol'], np.nan))
                    # print(adata.var.loc[i,'NCBI_Gene_ID'])
                    # adata.var.loc[i,'NCBI_Gene_ID'] = hgs2hgid.get(adata.var.loc[i,'gene_name'], np.nan)
                    adata.var.loc[i,'NCBI_Gene_ID'] = hgs2hgid.get(adata.var.loc[i,'gene_symbol'], np.nan)
                    adata.var.loc[i,'my_Id'] = hgid2myid.get(str(adata.var.loc[i,'NCBI_Gene_ID']), np.nan)

                # print(adata.var)

                nan_count = adata.var['NCBI_Gene_ID'].isna().sum()
                print('number of no mapping:', nan_count)
                
                # 从 adata.var 中删除要删除的基因
                flag = adata.var.index[(~adata.var['NCBI_Gene_ID'].isna()) & (~adata.var['my_Id'].isna())]
                new_var = adata.var.loc[flag,['gene_name', 'gene_symbol', 'NCBI_Gene_ID', 'my_Id']]
                # 从 adata.X 中删除相应的表达数据
                new_X = adata[:, flag].X
                # 创建一个新的 AnnData 对象
                filtered_adata = sc.AnnData