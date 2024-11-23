# GeneQuery: A General QA-based Framework for Spatial Gene Expression Predictions from Histology Images

## dataset
We have processed the data into h5ad data, which includes:
```
AnnData object with n_obs × n_vars = spots_num × gene_num
    obs: 'x', 'y', 'new_x', 'new_y', 'pixel_x', 'pixel_y', 'selected', 'barcode'
    var: 'hugo symbol', 'uid'
```

## wsi-aware code
wsi-aware/

### run code, take GSE dataset for example
bash scripts/run_gse.sh

## gene-aware coe
gene-aware

### run code, take GSE dataset for example
bash scripts/run_gse.sh


