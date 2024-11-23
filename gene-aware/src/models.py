import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from modules import (
    ProjectionHead, 
    ImageEncoder,
)
from transformers.utils import ModelOutput
from typing import Optional
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig
from cross_att import ViT, ViT_fuse
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeneModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    pred: Optional[torch.FloatTensor] = None

class GeneConfig(PretrainedConfig):
    def __init__(
        self,
        model_type="gene_query",
        gene_dim=256,
        gene_num=75500,
        gene_total=54683,
        image_encoder_name='resnet50',
        num_classes=0,
        global_pool='avg',
        pretrained_image_encoder=True,
        trainable_image_encoder=True,
        image_embedding_dim=2048,
        gene_encoder_type='mlp',
        gene_embedding_dim=768,
        gene_max_length=3467,
        num_gene_attention_heads=4,
        num_projection_layers=1,
        gene_projection_dim=256,
        gene_intermediate_size=1024,
        gene_dropout=0.1,
        gene_layer_norm_eps=1e-12,
        fuse_method='add',
        use_retrieval=False,
        nprobe=None,
        topk=None,
        retriever_device=-1,
        lam=0.5,
        n_layers=2,
        dim_head=64,
        use_gpt_description=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.gene_dim = gene_dim
        self.gene_num = gene_num
        self.gene_total = gene_total
        self.image_encoder_name = image_encoder_name
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.pretrained_image_encoder = pretrained_image_encoder
        self.trainable_image_encoder = trainable_image_encoder
        self.image_embedding_dim = image_embedding_dim
        self.gene_encoder_type = gene_encoder_type
        self.gene_embedding_dim = gene_embedding_dim
        self.gene_max_length = gene_max_length
        self.num_gene_attention_heads = num_gene_attention_heads
        self.num_projection_layers = num_projection_layers
        self.gene_projection_dim = gene_projection_dim
        self.gene_intermediate_size = gene_intermediate_size
        self.gene_dropout = gene_dropout
        self.gene_layer_norm_eps = gene_layer_norm_eps
        self.fuse_method = fuse_method
        self.use_retrieval = use_retrieval
        self.nprobe = nprobe
        self.topk = topk
        self.retriever_device = retriever_device
        self.n_layers = n_layers
        self.dim_head = dim_head
        self.lam = lam
        self.use_gpt_description = use_gpt_description

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class GeneQueryNameDes(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.patch_embed = ImageEncoder(config)
        self.image_projection = ProjectionHead(config)
        self.dropout = torch.nn.Dropout(config.gene_dropout)
        self.gene_trans = torch.nn.Linear(config.gene_embedding_dim, config.gene_dim)
        if config.fuse_method == 'add':
            self.regression = torch.nn.Linear(config.gene_dim, 1)
        elif config.fuse_method == 'concat':
            self.regression = torch.nn.Linear(config.gene_dim*2, 1)
        self.gelu = torch.nn.GELU()
        # self.retr_weight = torch.Tensor(config.topk, 1)
        # torch.nn.init.xavier_uniform_(self.retr_weight)
        # self.retr_weight = torch.nn.Parameter(self.retr_weight, requires_grad=True)

    def fuse(self, image_embeds, gene_embs):
        if self.config.fuse_method == 'add':
            return image_embeds + gene_embs
        elif self.config.fuse_method == 'concat':
            return torch.cat((image_embeds.expand(-1, gene_embs.size(1), -1), gene_embs), dim=2)

    def forward(
        self, 
        image, 
        barcode, 
        spatial_coords, 
        rna_list, 
        gene_name,
        gene_name_des, 
        random_name, 
        pro_list,
        label,
        normalization,
    ):
        # Get Gene Values
        rna_value = rna_list[:, 0, :].float().requires_grad_(False)
        # Encode Image
        image_patch_features = self.patch_embed(image)
        image_embeds = self.image_projection(image_patch_features)
        image_embeds = image_embeds.unsqueeze(1)
        # Encode Gene
        gene_embeds = self.gene_trans(gene_name_des)
        # Fuse Image and Gene
        joint_embeds = self.fuse(image_embeds, gene_embeds)
        joint_embeds = self.dropout(joint_embeds)
        joint_embeds = self.gelu(joint_embeds)
     
        pred = self.regression(joint_embeds).squeeze(2)
      
        loss = torch.nn.functional.mse_loss(pred.squeeze(-1), normalization.to(torch.float32))
        return GeneModelOutput(
            loss=loss.mean(dim=0),
            pred=pred
        )

class GeneDesTransformer(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.patch_embed = ImageEncoder(config)
        self.image_projection = ProjectionHead(config, config.image_embedding_dim, config.gene_projection_dim)
        self.dp = config.gene_dropout
        self.dropout = torch.nn.Dropout(config.gene_dropout)
        self.gene_trans = torch.nn.Linear(config.gene_embedding_dim, config.gene_dim)
        if config.fuse_method == 'add':
            self.regression = torch.nn.Linear(config.gene_dim, 1)
            self.cross = ViT(dim=config.gene_dim, depth=config.n_layers, heads=config.dim_head, mlp_dim=2*config.gene_dim, dropout = self.dp, emb_dropout = self.dp)
        elif config.fuse_method == 'concat':
            self.regression = torch.nn.Linear(config.gene_dim*2, 1)
            self.cross = ViT(dim=config.gene_dim*2, depth=config.n_layers, heads=config.dim_head, mlp_dim=2*config.gene_dim, dropout = self.dp, emb_dropout = self.dp)
        self.gelu = torch.nn.GELU()
        # self.add_weight = torch.Tensor(config.gene_max_length, 1)
        # torch.nn.init.xavier_uniform_(self.add_weight)
        # self.add_weight = torch.nn.Parameter(self.add_weight, requires_grad=True)

    def fuse(self, image_embeds, gene_embs):
        if self.config.fuse_method == 'add':
            return image_embeds + gene_embs
        
        elif self.config.fuse_method == 'concat':
            return torch.cat((image_embeds.expand(-1, gene_embs.size(1), -1), gene_embs), dim=2)

    def forward(
        self, 
        image, 
        barcode, 
        spatial_coords, 
        rna_list, 
        gene_name,
        gene_name_des, 
        random_name, 
        pro_list,
        label,
        normalization,
    ):
        # Get Gene Values
        rna_value = rna_list[:, 0, :].float().requires_grad_(False)
        mask = rna_list[:,-1,:].int().requires_grad_(False)
        # Encode Image
        image_patch_features = self.patch_embed(image)
        image_embeds = self.image_projection(image_patch_features)
        image_embeds = image_embeds.unsqueeze(1)
        # Encode Gene
        if self.config.use_gpt_description:
            gene_embeds = self.gene_trans(gene_name_des)
        else:
            gene_embeds = self.gene_trans(gene_name)
        # Fuse Image and Gene
        joint_embeds = self.fuse(image_embeds, gene_embeds)
        joint_embeds = self.dropout(joint_embeds)
        joint_embeds = self.gelu(joint_embeds)
        gene_embeds = self.cross(joint_embeds, mask)
        # Make Predictions
        pred = self.regression(gene_embeds)
        # Compute Loss
        loss = torch.nn.functional.mse_loss(pred.squeeze(-1), normalization.to(torch.float32))
        return GeneModelOutput(
            loss=loss.mean(dim=0),
            pred=pred
        )
