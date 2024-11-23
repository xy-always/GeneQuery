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
from transformers import PreTrainedModel, PretrainedConfig, AutoModel
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
        pretrained=True,
        trainable=True,
        image_embedding_dim=2048,
        num_projection_layers=1,
        projection_dim=256,
        gene_text_embedding_dim=768,
        wsi_max_length=3467,
        n_layers=2,
        dim_head=64,
        fuse_method='add',
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
        self.pretrained = pretrained
        self.trainable = trainable
        self.image_embedding_dim = image_embedding_dim
        self.num_projection_layers = num_projection_layers
        self.projection_dim = projection_dim
        self.gene_text_embedding_dim = gene_text_embedding_dim
        self.wsi_max_length = wsi_max_length
        self.n_layers = n_layers
        self.dim_head = dim_head
        self.fuse_method = fuse_method

class GeneQueryName(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.patch_embed = ImageEncoder(config)
        self.image_projection = ProjectionHead(config)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.dp = 0.1
        self.gene_model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        self.gene_trans = torch.nn.Linear(config.gene_embedding_dim, config.gene_dim)
        if config.fuse_method == 'add':
            self.regression = torch.nn.Linear(config.gene_dim, 1)
            self.cross = ViT(dim=config.gene_dim, depth=config.n_layers, heads=config.dim_head, mlp_dim=2*config.gene_dim, dropout = self.dp, emb_dropout = self.dp)
            self.regression = torch.nn.Linear(config.gene_dim, 1)
        elif config.fuse_method == 'concat':
            self.regression = torch.nn.Linear(config.gene_dim*2, 1)
            self.cross = ViT(dim=config.gene_dim*2, depth=config.n_layers, heads=config.dim_head, mlp_dim=2*config.gene_dim, dropout = self.dp, emb_dropout = self.dp)
            self.regression = torch.nn.Linear(config.gene_dim*2, 1)

        self.gelu = torch.nn.GELU()
        self.relu = torch.nn.ReLU()

    def fuse(self, image_embeds, gene_embeds):
        if self.config.fuse_method == 'add':
            return image_embeds + gene_embeds
        elif self.config.fuse_method == 'concat':
            gene_embed_expanded = gene_embeds.unsqueeze(1).expand(-1, image_embeds.size()[1], -1)
            return torch.cat((gene_embed_expanded, image_embeds), dim=-1)

    def forward(
        self, 
        id,
        gene_ids,
        image, 
        gene_name,
        gene_name_des, 
        random_name, 
        mask,
        label,
        normalization,
    ):
        
        # Encode Image
        # image = image.reshape(-1, 3, 224, 224)
        # image_patch_features = self.patch_embed(image)
        # image_patch_features = image_patch_features.reshape(label.shape[0], -1, image_patch_features.size(-1))
        # print('image_patch_features:', image.shape)
        image_embeds = self.image_projection(image)
        # Encode Gene
        gene_embeds = self.gene_model(gene_ids).last_hidden_state[:,0,:]
        gene_embeds = self.gene_trans(gene_embeds)
    
        # Fuse Image and Gene
        joint_embeds = self.fuse(image_embeds, gene_embeds)
        joint_embeds = self.relu(joint_embeds)
        mask = mask.to(torch.float32)
        cross_embeds = self.cross(joint_embeds, mask=mask)
        # Make Predictions
        pred = self.regression(cross_embeds)
        
        # Compute Loss
        loss = torch.nn.functional.mse_loss(pred.squeeze(-1), normalization.to(torch.float32))
        return GeneModelOutput(
            loss=loss.mean(),
            pred=pred
        )

class GeneQueryDesMLP(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.patch_embed = ImageEncoder(config)
        self.image_projection = ProjectionHead(config)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.dp = 0.1
        self.gene_model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        self.gene_trans = torch.nn.Linear(config.gene_embedding_dim, config.gene_dim)
        if config.fuse_method == 'add':
            self.cross = ViT(dim=config.gene_dim, depth=config.n_layers, heads=config.dim_head, mlp_dim=2*config.gene_dim, dropout = self.dp, emb_dropout = self.dp)
            self.regression = torch.nn.Linear(config.gene_dim, 1)
        elif config.fuse_method == 'concat':
            self.cross = ViT(dim=config.gene_dim*2, depth=config.n_layers, heads=config.dim_head, mlp_dim=2*config.gene_dim, dropout = self.dp, emb_dropout = self.dp)
            self.regression = torch.nn.Linear(config.gene_dim*2, 1)

        self.gelu = torch.nn.GELU()
        self.relu = torch.nn.ReLU()

    def fuse(self, image_embeds, gene_embeds):
        if self.config.fuse_method == 'add':
            gene_embeds = gene_embeds.unsqueeze(1)
            return image_embeds + gene_embeds
        elif self.config.fuse_method == 'concat':
            gene_embed_expanded = gene_embeds.unsqueeze(1).expand(-1, image_embeds.size()[1], -1)
            print('gene_embed_expanded shape:', gene_embed_expanded.shape)
            return torch.cat((gene_embed_expanded, image_embeds), dim=-1)

    def forward(
        self, 
        id,
        gene_ids,
        image, 
        gene_name,
        gene_name_des, 
        random_name, 
        mask,
        label,
        normalization,
    ):
    
        # Encode Image
        # image = image.reshape(-1, 3, 224, 224)
        # image_patch_features = self.patch_embed(image)
        # image_patch_features = image_patch_features.reshape(label.shape[0], -1, image_patch_features.size(-1))
        # print('image_patch_features:', image.shape)
        image_embeds = self.image_projection(image)
        # print('image_embeds:', image_embeds.shape)
        # Encode Gene
        gene_embeds = self.gene_model(gene_ids).last_hidden_state[:,0,:]
        gene_embeds = self.gene_trans(gene_embeds)
        mask = mask.to(torch.float32)
        # Fuse Image and Gene
        joint_embeds = self.fuse(image_embeds, gene_embeds)
        joint_embeds = self.relu(joint_embeds)
        pred = self.regression(joint_embeds)
    

        # Compute Loss
        loss = torch.nn.functional.mse_loss(pred.squeeze(-1), label.to(torch.float32))
        return GeneModelOutput(
            loss=loss.mean(),
            pred=pred
        )

class GeneQueryNameDes(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.patch_embed = ImageEncoder(config)
        self.image_projection = ProjectionHead(config)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.dp = 0.1
        self.gene_model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        self.gene_trans = torch.nn.Linear(config.gene_embedding_dim, config.gene_dim)
        if config.fuse_method == 'add':
            self.regression = torch.nn.Linear(config.gene_dim, 1)
            self.cross = ViT(dim=config.gene_dim, depth=config.n_layers, heads=config.dim_head, mlp_dim=2*config.gene_dim, dropout = self.dp, emb_dropout = self.dp)
        elif config.fuse_method == 'concat':
            self.regression = torch.nn.Linear(config.gene_dim*2, 1)
            self.cross = ViT(dim=config.gene_dim*2, depth=config.n_layers, heads=config.dim_head, mlp_dim=2*config.gene_dim, dropout = self.dp, emb_dropout = self.dp)

        self.gelu = torch.nn.GELU()
        self.relu = torch.nn.ReLU()
        self.main_input_name = 'mask'

    def fuse(self, image_embeds, gene_embeds):
        if self.config.fuse_method == 'add':
            gene_embs = gene_embeds.unsqueeze(1)
            return image_embeds + gene_embs
        elif self.config.fuse_method == 'concat':
            gene_embed_expanded = gene_embeds.unsqueeze(1).expand(-1, image_embeds.size()[1], -1)
            return torch.cat((gene_embed_expanded, image_embeds), dim=-1)

    def forward(
        self, 
        id,
        gene_ids,
        image, 
        gene_name,
        gene_name_des, 
        random_name, 
        mask,
        label,
        normalization,
    ):
    
        # Encode Image
        # image = image.reshape(-1, 3, 224, 224)
        # image_patch_features = self.patch_embed(image)
        # image_patch_features = image_patch_features.reshape(label.shape[0], -1, image_patch_features.size(-1))
        # print('image_patch_features:', image.shape)
        image_embeds = self.image_projection(image)
        # print('image_embeds:', image_embeds.shape)
        # Encode Gene
        gene_embeds = self.gene_model(gene_ids).last_hidden_state[:,0,:]
        gene_embeds = self.gene_trans(gene_embeds)
        mask = mask.to(torch.float32)
        # Fuse Image and Gene
        joint_embeds = self.fuse(image_embeds, gene_embeds)
        joint_embeds = self.relu(joint_embeds)
        cross_embeds = self.cross(joint_embeds, mask)
        # Make Predictions
        pred = self.regression(cross_embeds)
        
        # Compute Loss
        loss = torch.nn.functional.mse_loss(pred.squeeze(-1), normalization.to(torch.float32))
        return GeneModelOutput(
            loss=loss.mean(),
            pred=pred
        )
