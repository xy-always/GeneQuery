

### 数据准备
把GSE_chatgpt_definition.txt 移到GSE目录下，为GSE/GSE_chatgpt_definition.txt
把gse_gpt_description_emb.npy 移到数据GSE目录下为GSE/gse_gpt_description_emb.npy

把HBD_chatgpt_definition.txt 移到HBD目录下，为HBD/HBD_chatgpt_definition.txt
把hbd_gpt_description_emb.npy 移到数据HBD目录下为HBD/hbd_gpt_description_emb.npy

把her2_chatgpt_definition.txt 移到her2st目录下，为her2st/her2_chatgpt_definition.txt
把her2_gpt_description_emb.npy 移到数据her2st目录下为her2st/her2_gpt_description_emb.npy


### 训练
需要修改的东西，PROJECT_ROOT、脚本GPU的标号DEVICE和使用的GPU的个数NUM_DEVICES

### For GSE dataset
cd wsi-aware
sbatch scripts/run_gse_genefuse.sh

#### For hbd dataset
cd wsi-aware
sbatch scripts/run_hbd_nfold.sh

### For her2+ dataset
cd wsi-aware
sbatch scripts/run_her2_nfold.sh


