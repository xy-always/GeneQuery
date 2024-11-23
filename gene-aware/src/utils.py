import numpy as np
from sentence_transformers import SentenceTransformer
import scanpy as sc

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


def fix_sc_normalize_truncate_padding(x, length, cls_token, norm=False):
    len_x = len(x.X[0])
    tmp = [i for i in x.X[0]]
    if len_x >= length: # truncate
        x_value = list(tmp[:length])
        # gene = list(x.var.iloc[:length]['uid'].astype(int).values)
        mask = np.full(length, True)
    else: # padding
        x_value = tmp
        x_value.extend([1e-8 for i in range(length-len_x)])
        # gene =list(x.var['uid'].astype(int).values)
        # gene.extend([(cls_token+1) for i in range(length-len_x)])
        mask = np.concatenate((np.full(len_x,True), np.full(length-len_x,False)))
    return np.array(x_value), np.array(x_value), np.array(mask)

def get_genename_emb(gene_name_file, gpt_description_file, saved_path):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    gene_name = sc.read_h5ad(gene_name_file).var['hugo symbol'].values
    print(len(gene_name))
    gpt = open(gpt_description_file, 'r').readlines()
    gpt_description = {}
    for g in gpt:
        name, des = g.split('####')
        gpt_description[name] = des
    outputs = []
    for gene in gene_name:
        inputs = tokenizer(gpt_description[gene], return_tensors="pt", padding=True, truncation=True)
        output = model(**inputs)
        output = output.last_hidden_state[:,0,:].detach().squeeze().numpy()
        # print(output.shape)
        outputs.append(output)
    np.save(saved_path, np.array(outputs))
