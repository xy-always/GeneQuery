import torch
from torch import nn
import torch.nn.functional as F

from modules import ImageEncoder_resnet50, ProjectionHead, ImageEncoder_ViT, ImageEncoder_ViT_L, ImageEncoder_CLIP, ImageEncoder_resnet101, ImageEncoder_resnet152, ImageEncoder_Mae_ViT
from timm.models.vision_transformer import PatchEmbed
import timm
# from TSFM.performer import PerformerModule
# from TSFM.transformer import pytorchTransformerModule
import sys
from sentence_transformers import SentenceTransformer
sys.path.append('/home/xiangruike/XY_project/BLEEP/performer-pytorch/performer_pytorch') 
from performer_pytorch import PerformerGene
import numpy as np
from pytorch_forecasting.metrics.point import PoissonLoss, TweedieLoss

def poisson_loss(y_true, y_pred, temperature=1):
    """
    计算泊松损失的函数。
    
    参数:
    y_true: 真实计数数据，形状为 (batch_size,)
    y_pred: 预测计数数据，形状为 (batch_size,)
    temperature: 控制泊松分布的“温度”，影响预测的平滑程度。
    
    返回:
    负对数似然损失值。
    """
    # 计算负对数似然损失
    loss = -torch.sum(y_true * torch.log(y_pred + 1e-10) + (1 - y_true) * torch.log(1 - y_pred + 1e-10))
    return loss

class CLIPModel(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet50(args)
#         self.spot_encoder = SpotEncoder()
        self.temperature=args.temperature
        self.image_embedding=args.image_embedding_dim
        self.spot_embedding=args.spot_embedding_dim
        self.image_projection = ProjectionHead(args=args,embedding_dim=self.image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(args=args,embedding_dim=self.spot_embedding) #3467 shared hvgs
        self.device = args.current_device

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"].to(self.device))
        # spot_features = batch["reduced_expression"]
        x = batch['rna_list']
        spot_features = x[:,0,:].float().to(self.device)

        # print('image feature:', image_features.shape)
        # print('spot feature:', spot_features.shape)
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    
class CLIPModel_ViT(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.temperature=args.temperature,
        self.image_embedding=args.image_embedding_dim,
        self.spot_embedding=args.spot_embedding_dim,
        self.image_encoder = ImageEncoder_ViT(args)
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(args=args,embedding_dim=self.image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(args=args,embedding_dim=self.spot_embedding) #3467 shared hvgs

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

class CLIPModel_CLIP(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.temperature=args.temperature,
        self.image_embedding=args.image_embedding_dim,
        self.spot_embedding=args.spot_embedding_dim,
        self.image_encoder = ImageEncoder_CLIP(args)
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(args=args, embedding_dim=self.image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(args=args, embedding_dim=self.spot_embedding) #3467 shared hvgs
    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


class CLIPModel_ViT_L(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.temperature=args.temperature,
        self.image_embedding=args.image_embedding_dim,
        self.spot_embedding=args.spot_embedding_dim,
        self.image_encoder = ImageEncoder_ViT_L(args)
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(args=args,embedding_dim=self.image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(args=args,embedding_dim=self.spot_embedding) #3467 shared hvgs

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

class CLIPModel_resnet101(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.image_embedding=args.image_embedding_dim,
        self.spot_embedding=args.spot_embedding_dim,
        self.image_encoder = ImageEncoder_resnet101(args)
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(args=args,embedding_dim=self.image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(args=args,embedding_dim=self.spot_embedding) #3467 shared hvgs
        self.temperature = args.temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

class CLIPModel_resnet152(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.image_embedding=args.image_embedding_dim,
        self.spot_embedding=args.spot_embedding_dim,
        self.image_encoder = ImageEncoder_resnet152(args)
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=self.image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=self.spot_embedding) #3467 shared hvgs
        self.temperature = args.temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

class CLIPModel_CLIP_Pretrain(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.image_embedding=args.image_embedding_dim,
        self.spot_embedding=args.spot_embedding_dim,
        self.spot_pretrained_model = None
        # self.image_encoder = ImageEncoder_resnet50()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(args=args,embedding_dim=self.image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(args=args,embedding_dim=self.spot_embedding) #3467 shared hvgs
        self.temperature = args.temperature
        # self.spot_pretrained_model = scPerformerEncDec(enc_max_seq_len=20000,dec_max_seq_len=1000)
        # self.spot_pretrained_model.load_state_dict(spot_states)
        # for name, param in self.spot_pretrained_model.named_parameters():
        #     param.requires_grad = False
                # self.spot_trans = nn.Linear(20000,1)
        self.patch_embed = PatchEmbed(224, 16, 3, 128)

    def forward(self, batch):
        # Getting Image and spot Features
        # image_features = self.image_encoder(batch["image"])
        image_patch_features = self.patch_embed(batch["image"])
        x = batch["rna_list"]
        y = batch['pro_list']
        RNA_geneID = x[:,1,:].long().cuda().clone().detach().requires_grad_(False)
        Protein_geneID = y[:,1,:].long().cuda().clone().detach().requires_grad_(False)
        # rna_mask = x[:,2,:].bool().cuda().clone().detach().requires_grad_(False)
        # pro_mask = y[:,2,:].bool().cuda().clone().detach().requires_grad_(False)
        rna_seq = x[:,0,:].float().cuda().clone().detach().requires_grad_(False)
        # print(RNA_geneID)
        # print(type(rna_mask)
        # print(rna_seq)
        # Protein_geneID = y[:,1,:].long().cuda().clone().detach().requires_grad_(False)
        spot_pretrained_feature, _ = self.spot_pretrained_model(rna_seq, RNA_geneID, Protein_geneID)
#       spot_features = self.spot_encoder(batch["reduced_expression"])
        # print('spot pretrained feature:', spot_pretrained_feature.shape)
        # Getting Image and Spot Embeddings (with same dimension) 
        # print("spot_feature:", spot_pretrained_feature.shape)
        image_embeddings = self.image_projection(image_patch_features)
        spot_embeddings = self.spot_projection(spot_pretrained_feature)
        # print('image shape:', image_embeddings.shape)
        # print('spot shape:', spot_embeddings.shape)
        # spot_embeddings = self.spot_trans(spot_embeddings.transpose(1,2)).squeeze()
        # Calculating the Attention
        image_spot_attention = F.softmax( torch.matmul(image_embeddings, spot_embeddings.transpose(1,2)), dim=1)
        spot_image_attention = F.softmax( torch.matmul(spot_embeddings, image_embeddings.transpose(1,2)), dim=1)
        
        # Fusing the Information
        fused_image_embeddings = image_embeddings + torch.bmm(image_spot_attention, spot_embeddings)
        fused_spot_embeddings = spot_embeddings + torch.bmm(spot_image_attention, image_embeddings)

        fused_image_embeddings = torch.mean(fused_image_embeddings, dim=1)
        fused_spot_embeddings = torch.mean(fused_spot_embeddings, dim=1)
        # print('fuse_image_embeddings:', fused_image_embeddings.shape)
        # print('fuse_spot_embeddings:', fused_spot_embeddings.shape)
        
        # Calculating the Loss

        logits = (fused_spot_embeddings @ fused_image_embeddings.T) / self.temperature
        images_similarity = fused_image_embeddings @ fused_image_embeddings.T
        spots_similarity = fused_spot_embeddings @ fused_spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

class CLIPModel_CLIP_CrossAtt(nn.Module):
    def __init__(
        self,
        args,
        
    ):
        super().__init__()
        self.gene_dim = 256
        self.gene_num = args.gene_total
        self.image_projection = ProjectionHead(args=args,embedding_dim=args.image_embedding_dim) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(args=args,embedding_dim=self.gene_dim) #3467 shared hvgs
        self.temperature = args.temperature
        self.gene_embed = nn.Linear(1,self.gene_dim)
        self.gene_id_emb = nn.Embedding(self.gene_num, self.gene_dim, padding_idx=self.gene_num-1)# There are 75500 NCBI Gene ID
        self.spot_pretrained_model = pytorchTransformerModule(max_seq_len=args.max_length+1, dim=self.gene_dim, depth=4, heads=8)
        # self.patch_embed = PatchEmbed(224, 16, 3, 128)
        self.patch_embed = ImageEncoder_Mae_ViT(args)
        self.dropout = nn.Dropout(args.dropout)
        self.device = args.current_device

    def forward(self, batch):
        # Getting Image and spot Features
        # image_features = self.image_encoder(batch["image"])
        image_patch_features = self.patch_embed(batch["image"]).to(self.device)
        # print("pretrain vit image shape: ", image_patch_features.shape)
        x = batch["rna_list"]
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        rna_id = x[:,1,:].long().requires_grad_(False).to(self.device)
        rna_mask = x[:,2,:].bool().requires_grad_(False).to(self.device)
        rna_seq = x[:,0,:].float().requires_grad_(False).to(self.device)
        if len(rna_seq.shape) < 3:
            rna_seq = torch.unsqueeze(rna_seq, dim=2)
            rna_seq = self.gene_embed(rna_seq)
        gene_pos = self.gene_id_emb(rna_id)
        x = rna_seq + gene_pos
        x = self.dropout(x)
        # print(rna_mask)
        # print(RNA_geneID)
        # print(type(rna_mask)
        # print(rna_seq)
        # Protein_geneID = y[:,1,:].long().cuda().clone().detach().requires_grad_(False)
        spot_pretrained_feature = self.spot_pretrained_model(x, rna_mask)
        # print(spot_pretrained_feature)
#       spot_features = self.spot_encoder(batch["reduced_expression"])
        # print('spot pretrained feature:', spot_pretrained_feature.shape)
        # Getting Image and Spot Embeddings (with same dimension) 
        # print("spot_feature:", spot_pretrained_feature.shape)
        image_embeddings = self.image_projection(image_patch_features)
        spot_embeddings = self.spot_projection(spot_pretrained_feature)
        # print(image_patch_features)
        # print('image shape:', image_embeddings.shape)
        # print('spot shape:', spot_embeddings.shape)
        # spot_embeddings = self.spot_trans(spot_embeddings.transpose(1,2)).squeeze()
        # Calculating the Attention
        image_spot_attention = F.softmax( torch.matmul(image_embeddings, spot_embeddings.transpose(1,2)), dim=1)
        spot_image_attention = F.softmax( torch.matmul(spot_embeddings, image_embeddings.transpose(1,2)), dim=1)
        
        # Fusing the Information
        fused_image_embeddings = image_embeddings + torch.bmm(image_spot_attention, spot_embeddings)
        fused_spot_embeddings = spot_embeddings + torch.bmm(spot_image_attention, image_embeddings)

        fused_image_embeddings = torch.mean(fused_image_embeddings, dim=1)
        fused_spot_embeddings = torch.mean(fused_spot_embeddings, dim=1)
        # print('fuse_image_embeddings:', fused_image_embeddings.shape)
        # fused_image_embeddings = fused_image_embeddings[:,0,:].squeeze()
        # fused_spot_embeddings = fused_spot_embeddings[:,0,:].squeeze()
        # print('fuse_spot_embeddings:', fused_spot_embeddings.shape)
        # print(fused_image_embeddings)
        # Calculating the Loss

        logits = (fused_spot_embeddings @ fused_image_embeddings.T) / self.temperature
        images_similarity = fused_image_embeddings @ fused_image_embeddings.T
        spots_similarity = fused_spot_embeddings @ fused_spot_embeddings.T
        # print(images_similarity)
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    
class CLIPModel_CrossAtt_Multi(nn.Module):
    def __init__(
        self,
        args,
        
    ):
        super().__init__()
        self.gene_dim = 256
        self.temperature=args.temperature,
        self.gene_num = args.gene_total
        self.image_projection = ProjectionHead(args=args,embedding_dim=args.image_embedding_dim) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(args=args,embedding_dim=self.gene_dim)
        self.gene_embed = nn.Linear(1,self.gene_dim)
        self.gene_id_emb = nn.Embedding(self.gene_num, self.gene_dim, padding_idx=self.gene_num-1)# There are 75500 NCBI Gene ID
        # self.spot_pretrained_model = pytorchTransformerModule(max_seq_len=args.max_length+1, dim=self.gene_dim, depth=2, heads=4)
        self.spot_pretrained_model = Performer(dim=self.gene_dim, depth=1, heads=4, dim_head=128)
        # self.patch_embed = PatchEmbed(224, 16, 3, 128)
        self.patch_embed = ImageEncoder_Mae_ViT(args)
        self.dropout = nn.Dropout(args.dropout)
        # self.regression = nn.Linear(self.gene_dim, args.max_length)
        self.device = args.current_device

    def forward(self, batch):
        # Getting Image and spot Features
        # image_features = self.image_encoder(batch["image"])
        image_patch_features = self.patch_embed(batch["image"].to(self.device))
        # print("pretrain vit image shape: ", image_patch_features.shape)
        x = batch["rna_list"]
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        rna_id = x[:,1,:].long().requires_grad_(False).to(self.device)
        rna_mask = x[:,2,:].bool().requires_grad_(False).to(self.device)
        rna_value = x[:,0,:].float().requires_grad_(False).to(self.device)
        if len(rna_value.shape) < 3:
            rna_seq = torch.unsqueeze(rna_value, dim=2)
            rna_seq = self.gene_embed(rna_seq)
        gene_pos = self.gene_id_emb(rna_id)
        x = rna_seq + gene_pos
        x = self.dropout(x)
        # print(rna_mask)
        # print(RNA_geneID)
        # print(type(rna_mask)
        # print(rna_seq)
        # Protein_geneID = y[:,1,:].long().cuda().clone().detach().requires_grad_(False)
        spot_pretrained_feature = self.spot_pretrained_model(x, mask=rna_mask)
        # print(spot_pretrained_feature)
#       spot_features = self.spot_encoder(batch["reduced_expression"])
        # print('spot pretrained feature:', spot_pretrained_feature.shape)
        # Getting Image and Spot Embeddings (with same dimension) 
        # print("spot_feature:", spot_pretrained_feature.shape)
        image_embeddings = self.image_projection(image_patch_features)
        spot_embeddings = self.spot_projection(spot_pretrained_feature)
        # print(image_patch_features)
        # print('image shape:', image_embeddings.shape)
        # print('spot shape:', spot_embeddings.shape)
        # spot_embeddings = self.spot_trans(spot_embeddings.transpose(1,2)).squeeze()
        # Calculating the Attention
        image_spot_attention = F.softmax( torch.matmul(image_embeddings, spot_embeddings.transpose(1,2)), dim=1)
        spot_image_attention = F.softmax( torch.matmul(spot_embeddings, image_embeddings.transpose(1,2)), dim=1)
        
        # Fusing the Information
        fused_image_embeddings = image_embeddings + torch.bmm(image_spot_attention, spot_embeddings)
        fused_spot_embeddings = spot_embeddings + torch.bmm(spot_image_attention, image_embeddings)
        # print(fused_spot_embeddings[:,1:,:].shape)
        # print(rna_value.shape)

        fused_image_embeddings = torch.mean(fused_image_embeddings, dim=1)
        fused_spot_embeddings = torch.mean(fused_spot_embeddings, dim=1)

        # pred = self.regression(fused_image_embeddings)
        # print(pred.shape)
        # loss_predict = F.mse_loss(pred, rna_value[:,1:])
        logits = (fused_spot_embeddings @ fused_image_embeddings.T) / self.temperature
        images_similarity = fused_image_embeddings @ fused_image_embeddings.T
        spots_similarity = fused_spot_embeddings @ fused_spot_embeddings.T
        # print(images_similarity)
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        # total_loss = (loss + loss_predict)/2.0
        return loss.mean()

class GeneQueryName(nn.Module):
    def __init__(
        self,
        args,
        
    ):
        super().__init__()
        self.gene_dim = 256
        self.temperature=args.temperature,
        self.gene_num = args.gene_total
        self.image_projection = ProjectionHead(args=args,embedding_dim=args.image_embedding_dim) #aka the input dim, 2048 for resnet50
        # self.spot_projection = ProjectionHead(args=args,embedding_dim=self.gene_dim)
        # self.gene_embed = nn.Linear(1,self.gene_dim)
        # self.gene_id_emb = nn.Embedding(self.gene_num, self.gene_dim, padding_idx=self.gene_num-1)# There are 75500 NCBI Gene ID
        # self.spot_pretrained_model = pytorchTransformerModule(max_seq_len=args.max_length+1, dim=self.gene_dim, depth=2, heads=4)
        # self.spot_pretrained_model = Performer(dim=self.gene_dim, depth=1, heads=4, dim_head=128)
        # self.patch_embed = PatchEmbed(224, 16, 3, 128)
        self.patch_embed = ImageEncoder_resnet50(args)
        self.dropout = nn.Dropout(args.dropout)
        self.regression = nn.Linear(self.gene_dim,1)
        self.device = args.current_device
        self.gelu = nn.GELU()
        # if self.use_genename:
        #     self.sentence_model = SentenceTransformer('/root/autodl-tmp/wsy/models/bert-base-uncased/')
        self.relu = nn.ReLU()
        self.trans = nn.Linear(768, self.gene_dim)
        self.loss = TweedieLoss()
    def forward(self, batch):
        # Getting Image and spot Features
        # image_features = self.image_encoder(batch["image"])
        # print('image input shape:', batch["image"].shape)
        image_patch_features = self.patch_embed(batch["image"].to(self.device))
        # print("pretrain vit image shape: ", image_patch_features.shape)
        x = batch["rna_list"]
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        # rna_id = x[:,1,:].long().requires_grad_(False).to(self.device)
        # rna_mask = x[:,2,:].bool().requires_grad_(False).to(self.device)
        rna_value = x[:,0,:].float().requires_grad_(False).to(self.device)
        # if len(rna_value.shape) < 3:
        #     rna_seq = torch.unsqueeze(rna_value, dim=2)
        #     rna_seq = self.gene_embed(rna_seq)
        # if self.use_genename:
        #     gene_pos = self.sentence_model.encode(batch['gene_name'][:, 1:])
        gene_pos = self.trans(batch['gene_name'].to(self.device))
        np.save('gene_name_her2_emb.npy', gene_pos.detach().cpu().numpy())
        # gene_pos = self.trans(batch['random_name'].to(self.device).to(torch.float32))
        # else:
        #     gene_pos = self.gene_id_emb(rna_id[:, 1:])
        # x = rna_seq + gene_pos
        # x = self.dropout(x)
        image_embeddings = self.image_projection(image_patch_features)
        image_embeddings = image_embeddings.unsqueeze(1)
        # print('image shape:', image_embeddings.shape)
        # print('gene pos:', gene_pos.shape)
        image_embeddings_query_gene = image_embeddings + gene_pos
        image_embeddings_query_gene = self.gelu(image_embeddings_query_gene)
        pred = self.regression(image_embeddings_query_gene)
        # pred = self.relu(pred)
        # print('pred:', pred.shape)
        # print('gold:', rna_value[:,1:].shape)
        loss_predict = F.mse_loss(pred.squeeze(), rna_value[:,1:])
        # loss_predict = self.loss(pred.squeeze(), rna_value[:,1:])
        
        # total_loss = (loss + loss_predict)/2.0
        return loss_predict.mean(dim=0), self.loss.to_prediction(pred)

class GeneQueryNameDes(nn.Module):
    def __init__(
        self,
        args,
        
    ):
        super().__init__()
        self.gene_dim = 256
        self.temperature=args.temperature,
        self.gene_num = args.gene_total
        self.image_projection = ProjectionHead(args=args,embedding_dim=args.image_embedding_dim) #aka the input dim, 2048 for resnet50
        # self.spot_projection = ProjectionHead(args=args,embedding_dim=self.gene_dim)
        # self.gene_embed = nn.Linear(1,self.gene_dim)
        # self.gene_id_emb = nn.Embedding(self.gene_num, self.gene_dim, padding_idx=self.gene_num-1)# There are 75500 NCBI Gene ID
        # self.spot_pretrained_model = pytorchTransformerModule(max_seq_len=args.max_length+1, dim=self.gene_dim, depth=2, heads=4)
        # self.spot_pretrained_model = Performer(dim=self.gene_dim, depth=1, heads=4, dim_head=128)
        # self.patch_embed = PatchEmbed(224, 16, 3, 128)
        self.patch_embed = ImageEncoder_resnet50(args)
        self.dropout = nn.Dropout(args.dropout)
        self.regression = nn.Linear(self.gene_dim,1)
        self.device = args.current_device
        self.gelu = nn.GELU()
        # if self.use_genename:
        #     self.sentence_model = SentenceTransformer('/root/autodl-tmp/wsy/models/bert-base-uncased/')
        self.relu = nn.ReLU()
        self.trans = nn.Linear(768, self.gene_dim)
    def forward(self, batch):
        # Getting Image and spot Features
        # image_features = self.image_encoder(batch["image"])
        # print('image input shape:', batch["image"].shape)
        image_patch_features = self.patch_embed(batch["image"].to(self.device))
        # print("pretrain vit image shape: ", image_patch_features.shape)
        x = batch["rna_list"]
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        # rna_id = x[:,1,:].long().requires_grad_(False).to(self.device)
        # rna_mask = x[:,2,:].bool().requires_grad_(False).to(self.device)
        rna_value = x[:,0,:].float().requires_grad_(False).to(self.device)
        # if len(rna_value.shape) < 3:
        #     rna_seq = torch.unsqueeze(rna_value, dim=2)
        #     rna_seq = self.gene_embed(rna_seq)
        # if self.use_genename:
        #     gene_pos = self.sentence_model.encode(batch['gene_name'][:, 1:])
        gene_pos = self.trans(batch['gene_name_des'].to(self.device))
        np.save('gene_des_her2_emb.npy', gene_pos.detach().cpu().numpy())
        # gene_pos = self.trans(batch['random_name'].to(self.device).to(torch.float32))
        # else:
        #     gene_pos = self.gene_id_emb(rna_id[:, 1:])
        # x = rna_seq + gene_pos
        # x = self.dropout(x)
        image_embeddings = self.image_projection(image_patch_features)
        image_embeddings = image_embeddings.unsqueeze(1)
        # print('image shape:', image_embeddings.shape)
        # print('gene pos:', gene_pos.shape)
        image_embeddings_query_gene = image_embeddings + gene_pos
        image_embeddings_query_gene = self.gelu(image_embeddings_query_gene)
        pred = self.regression(image_embeddings_query_gene)
        # pred = self.relu(pred)
        # print('pred:', pred.shape)
        # print('gold:', rna_value[:,1:].shape)
        loss_predict = F.mse_loss(pred.squeeze(), rna_value[:,1:])
        
        # total_loss = (loss + loss_predict)/2.0
        return loss_predict.mean(dim=0), pred

class GeneQueryID(nn.Module):
    def __init__(
        self,
        args,
        
    ):
        super().__init__()
        self.gene_dim = 256
        self.temperature=args.temperature,
        self.gene_num = args.gene_total
        self.max_length = args.max_length
        self.image_projection = ProjectionHead(args=args,embedding_dim=args.image_embedding_dim) #aka the input dim, 2048 for resnet50
        # self.spot_projection = ProjectionHead(args=args,embedding_dim=self.gene_dim)
        # self.gene_embed = nn.Linear(1,self.gene_dim)
        self.gene_id_emb = nn.Embedding(self.gene_num, self.gene_dim, padding_idx=self.gene_num-1)# There are 75500 NCBI Gene ID
        # self.gene_position_id = nn.Embedding(self.max_length, self.gene_dim)# There are 75500 NCBI Gene ID
        # self.spot_pretrained_model = pytorchTransformerModule(max_seq_len=args.max_length+1, dim=self.gene_dim, depth=2, heads=4)
        # self.spot_pretrained_model = Performer(dim=self.gene_dim, depth=1, heads=4, dim_head=128)
        # self.patch_embed = PatchEmbed(224, 16, 3, 128)
        self.patch_embed = ImageEncoder_resnet50(args)
        self.dropout = nn.Dropout(args.dropout)
        self.regression = nn.Linear(self.gene_dim,1)
        self.relu = nn.ReLU()
        self.device = args.current_device
        self.gelu = nn.GELU()
       
    def forward(self, batch):
        # Getting Image and spot Features
        # image_features = self.image_encoder(batch["image"])
        # print('image input shape:', batch["image"].shape)
        image_patch_features = self.patch_embed(batch["image"].to(self.device))
        # print("pretrain vit image shape: ", image_patch_features.shape)
        x = batch["rna_list"]
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        rna_id = x[:,1,:].long().requires_grad_(False).to(self.device)
        # rna_mask = x[:,2,:].bool().requires_grad_(False).to(self.device)
        rna_value = x[:,0,:].float().requires_grad_(False).to(self.device)
        # if len(rna_value.shape) < 3:
        #     rna_seq = torch.unsqueeze(rna_value, dim=2)
        #     rna_seq = self.gene_embed(rna_seq)
        # if self.use_genename:
        #     gene_pos = self.sentence_model.encode(batch['gene_name'][:, 1:])
        gene_pos = self.gene_id_emb(rna_id[:, 1:])
        # np.save('gene_id_emb.npy', gene_pos.detach().cpu().numpy())

        # x = rna_seq + gene_pos
        # x = self.dropout(x)
        image_embeddings = self.image_projection(image_patch_features)
        image_embeddings = image_embeddings.unsqueeze(1)
        # print('image shape:', image_embeddings.shape)
        # print('gene pos:', gene_pos.shape)
        image_embeddings_query_gene = image_embeddings + gene_pos
        # gene_position_id = torch.arange(self.max_length).to(self.device)
        # image_embeddings_query_gene = image_embeddings + self.gene_position_id(gene_position_id)
        image_embeddings_query_gene = self.gelu(image_embeddings_query_gene)
        pred = self.regression(image_embeddings_query_gene)
        # pred = self.relu(pred)
        print('***********pred:', pred.shape)
        # print('gold:', rna_value[:,1:].shape)
        loss_predict = F.mse_loss(pred.squeeze(), rna_value[:,1:])
        # loss_predict = F.l1_loss(pred.squeeze(), rna_value[:,1:])
        
        # total_loss = (loss + loss_predict)/2.0
        return loss_predict.mean(dim=0), pred

class STNet(nn.Module):
    def __init__(
        self,
        args,
        
    ):
        super().__init__()
        self.gene_dim = 256
        self.temperature=args.temperature,
        self.gene_num = args.gene_total
        self.image_projection = ProjectionHead(args=args,embedding_dim=args.image_embedding_dim) #aka the input dim, 2048 for resnet50
        # self.spot_projection = ProjectionHead(args=args,embedding_dim=self.gene_dim)
        # self.gene_embed = nn.Linear(1,self.gene_dim)
        # self.gene_id_emb = nn.Embedding(self.gene_num, self.gene_dim, padding_idx=self.gene_num-1)# There are 75500 NCBI Gene ID
        # self.spot_pretrained_model = pytorchTransformerModule(max_seq_len=args.max_length+1, dim=self.gene_dim, depth=2, heads=4)
        # self.spot_pretrained_model = Performer(dim=self.gene_dim, depth=1, heads=4, dim_head=128)
        # self.patch_embed = PatchEmbed(224, 16, 3, 128)
        self.patch_embed = ImageEncoder_resnet50(args)
        self.dropout = nn.Dropout(args.dropout)
        self.regression = nn.Linear(self.gene_dim,args.max_length)
        self.device = args.current_device
        # self.gelu = nn.GELU()
        print('using stnet....')
       
    def forward(self, batch):
        # Getting Image and spot Features
        # image_features = self.image_encoder(batch["image"])
        # print('image input shape:', batch["image"].shape)
        image_patch_features = self.patch_embed(batch["image"].to(self.device))
        # print("pretrain vit image shape: ", image_patch_features.shape)
        x = batch["rna_list"]
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        rna_id = x[:,1,:].long().requires_grad_(False).to(self.device)
        # rna_mask = x[:,2,:].bool().requires_grad_(False).to(self.device)
        rna_value = x[:,0,:].float().requires_grad_(False).to(self.device)
        # if len(rna_value.shape) < 3:
        #     rna_seq = torch.unsqueeze(rna_value, dim=2)
        #     rna_seq = self.gene_embed(rna_seq)
        # if self.use_genename:
        #     gene_pos = self.sentence_model.encode(batch['gene_name'][:, 1:])
        # gene_pos = self.gene_id_emb(rna_id[:, 1:])
        # x = rna_seq + gene_pos
        # x = self.dropout(x)
        image_embeddings = self.image_projection(image_patch_features)
        pred = self.regression(image_embeddings)
        print('=============pred:', pred.shape)
        # print('gold:', rna_value[:,1:].shape)
        loss_predict = F.mse_loss(pred.squeeze(), rna_value[:,1:])
        # loss_predict = F.l1_loss(pred.squeeze(), rna_value[:,1:])
        
        # total_loss = (loss + loss_predict)/2.0
        return loss_predict.mean(dim=0), pred
    
class GeneQueryTansformer(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.gene_dim = 256
        self.temperature=args.temperature,
        self.gene_num = args.gene_total
        self.image_projection = ProjectionHead(args=args,embedding_dim=args.image_embedding_dim) #aka the input dim, 2048 for resnet50
        # self.spot_projection = ProjectionHead(args=args,embedding_dim=self.gene_dim)
        # self.gene_embed = nn.Linear(1,self.gene_dim)
        self.gene_id_emb = nn.Embedding(self.gene_num, self.gene_dim, padding_idx=self.gene_num-1)# There are 75500 NCBI Gene ID
        self.spot_pretrained_model = PerformerGene(num_tokens=1, max_seq_len=args.max_length, dim=self.gene_dim, depth=2, heads=4)
       
        # self.patch_embed = PatchEmbed(224, 16, 3, 128)
        self.patch_embed = ImageEncoder_resnet50(args)
        # self.dropout = nn.Dropout(args.dropout)
        # self.regression = nn.Linear(self.gene_dim,args.max_length)
        self.device = args.current_device
        self.gelu = nn.GELU()
       
    def forward(self, batch):
        image_patch_features = self.patch_embed(batch["image"].to(self.device))
        # print("pretrain vit image shape: ", image_patch_features.shape)
        x = batch["rna_list"]
        rna_id = x[:,1,:].long().requires_grad_(False).to(self.device)
        rna_mask = x[:,2,:].bool().requires_grad_(False).to(self.device)
        rna_value = x[:,0,:].float().requires_grad_(False).to(self.device)
        gene_pos = self.gene_id_emb(rna_id[:, 1:])
        image_embeddings = self.image_projection(image_patch_features)
        image_embeddings = image_embeddings.unsqueeze(1)
        pred = self.spot_pretrained_model(image_embeddings+gene_pos, mask=rna_mask[:,1:])
        # pred = self.regression(image_embeddings)
        loss_predict = F.mse_loss(pred.squeeze(), rna_value[:,1:])

        return loss_predict.mean(dim=0), pred

class GeneQueryNameTansformer(nn.Module):
    def __init__(
        self,
        args,
        
    ):
        super().__init__()
        self.gene_dim = 256
        self.temperature=args.temperature,
        self.gene_num = args.gene_total
        self.image_projection = ProjectionHead(args=args,embedding_dim=args.image_embedding_dim) #aka the input dim, 2048 for resnet50
        self.spot_pretrained_model = PerformerGene(num_tokens=1, max_seq_len=args.max_length, dim=self.gene_dim, depth=2, heads=4)
        self.patch_embed = ImageEncoder_resnet50(args)
        self.dropout = nn.Dropout(args.dropout)
        # self.regression = nn.Linear(self.gene_dim,1)
        self.device = args.current_device
        self.gelu = nn.GELU()
        # if self.use_genename:
        #     self.sentence_model = SentenceTransformer('/root/autodl-tmp/wsy/models/bert-base-uncased/')
        self.trans = nn.Linear(768, self.gene_dim)
    def forward(self, batch):
        # Getting Image and spot Features
        # image_features = self.image_encoder(batch["image"])
        # print('image input shape:', batch["image"].shape)
        image_patch_features = self.patch_embed(batch["image"].to(self.device))
        # print("pretrain vit image shape: ", image_patch_features.shape)
        x = batch["rna_list"]
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        # rna_id = x[:,1,:].long().requires_grad_(False).to(self.device)
        rna_mask = x[:,2,:].bool().requires_grad_(False).to(self.device)
        rna_value = x[:,0,:].float().requires_grad_(False).to(self.device)
        # if len(rna_value.shape) < 3:
        #     rna_seq = torch.unsqueeze(rna_value, dim=2)
        #     rna_seq = self.gene_embed(rna_seq)
        # if self.use_genename:
        #     gene_pos = self.sentence_model.encode(batch['gene_name'][:, 1:])
        gene_pos = self.trans(batch['gene_name'].to(self.device))
        # else:
        #     gene_pos = self.gene_id_emb(rna_id[:, 1:])
        # x = rna_seq + gene_pos
        # x = self.dropout(x)
        image_embeddings = self.image_projection(image_patch_features)
        image_embeddings = image_embeddings.unsqueeze(1)
        # print('image shape:', image_embeddings.shape)
        # print('gene pos:', gene_pos.shape)
        image_embeddings_query_gene = image_embeddings + gene_pos
        image_embeddings_query_gene = self.gelu(image_embeddings_query_gene)
        pred = self.spot_pretrained_model(image_embeddings+gene_pos, mask=rna_mask[:,1:])
        # print('pred:', pred.shape)
        # print('gold:', rna_value[:,1:].shape)
        loss_predict = F.mse_loss(pred.squeeze(), rna_value[:,1:])
        
        # total_loss = (loss + loss_predict)/2.0
        return loss_predict.mean(dim=0), pred

class AugGeneQueryID(nn.Module):
    def __init__(
        self,
        args,
        
    ):
        super().__init__()
        self.gene_dim = 256
        self.temperature=args.temperature,
        self.gene_num = args.gene_total
        self.image_projection = ProjectionHead(args=args,embedding_dim=args.image_embedding_dim) #aka the input dim, 2048 for resnet50
        # self.spot_projection = ProjectionHead(args=args,embedding_dim=self.gene_dim)
        # self.gene_embed = nn.Linear(1,self.gene_dim)
        self.gene_id_emb = nn.Embedding(self.gene_num, self.gene_dim, padding_idx=self.gene_num-1)# There are 75500 NCBI Gene ID
        # self.spot_pretrained_model = pytorchTransformerModule(max_seq_len=args.max_length+1, dim=self.gene_dim, depth=2, heads=4)
        # self.spot_pretrained_model = Performer(dim=self.gene_dim, depth=1, heads=4, dim_head=128)
        # self.patch_embed = PatchEmbed(224, 16, 3, 128)
        self.patch_embed = ImageEncoder_resnet50(args)
        self.dropout = nn.Dropout(args.dropout)
        self.device = args.current_device
        self.gelu = nn.GELU()
        kernel_sizes = [2,3]
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, 100, (k, self.gene_dim), padding=(k-2,0)) 
            for k in kernel_sizes])
        
        self.fc = nn.Linear(len(kernel_sizes) * 100, self.gene_dim)
        self.regression = nn.Linear(self.gene_dim,1)

    def forward(self, batch):
        # Getting Image and spot Features
        # image_features = self.image_encoder(batch["image"])
        # print('image input shape:', batch["image"].shape)
        image_patch_features = self.patch_embed(batch["image"].to(self.device))
        # print("pretrain vit image shape: ", image_patch_features.shape)
        x = batch["rna_list"]
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        # neighbor = batch['neighbor'].to(self.device)
        batch['neighbor_raw'] = batch['neighbor_raw'].reshape(-1, 3, 224, 224)
        neighbor_raw = self.patch_embed(batch['neighbor_raw'].to(self.device))
        
        # print('neighbor raw:', neighbor_raw.shape)
        rna_id = x[:,1,:].long().requires_grad_(False).to(self.device)
        # rna_mask = x[:,2,:].bool().requires_grad_(False).to(self.device)
        rna_value = x[:,0,:].float().requires_grad_(False).to(self.device)
        # if len(rna_value.shape) < 3:
        #     rna_seq = torch.unsqueeze(rna_value, dim=2)
        #     rna_seq = self.gene_embed(rna_seq)
        # if self.use_genename:
        #     gene_pos = self.sentence_model.encode(batch['gene_name'][:, 1:])
        gene_pos = self.gene_id_emb(rna_id[:, 1:])
        # x = rna_seq + gene_pos
        # x = self.dropout(x)
        image_embeddings = self.image_projection(image_patch_features)
        neighbor_raw_encode = self.image_projection(neighbor_raw)
        neighbor_raw_encode = neighbor_raw_encode.reshape(image_embeddings.shape[0], -1, self.gene_dim)
        # img_f = torch.cat([image_embeddings.unsqueeze(1), neighbor_raw_encode], dim=1)
        # img_f = img_f.unsqueeze(1)
        # # print('img_f:', img_f.shape)
        # conv_out = [F.relu(conv(img_f)).squeeze(3) for conv in self.convs_1d]
        # x_max = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in conv_out]
        # x = torch.cat(x_max, 1)
        # x = self.fc(x)
        # print('conv_out:', x.shape)
        x = image_embeddings.unsqueeze(1)
        neighbor_raw_encode = neighbor_raw_encode.unsqueeze(1)
        image_embeddings_query_gene = x + gene_pos
        gene_pos = gene_pos.unsqueeze(2)
        neighbor_raw_encode = neighbor_raw_encode + gene_pos
        image_embeddings_query_gene = self.gelu(image_embeddings_query_gene)
        neighbor_raw_encode = self.gelu(neighbor_raw_encode)
        # pred_neighbor = self.regression(neighbor_raw_encode)
        neighbor_raw_encode = torch.mean(neighbor_raw_encode, dim=2)
        # print('neighbor_raw_encode', neighbor_raw_encode.shape)
        # print('imge_embeddings;', image_embeddings_query_gene.shape)
        image_embeddings_query_gene += neighbor_raw_encode
        # pred_neighbor = pred_neighbor.reshape(x.shape[0], -1, pred_neighbor.shape[1])
        # pred_neighbor = torch.mean(pred_neighbor, dim=1)
        # print('pred_neighbor:', pred_neighbor.shape)

        # print('image_embeddings_query_gene shape:', image_embeddings_query_gene.shape)

        # image_embeddings += self.image_projection(neighbor)
        # image_embeddings = image_embeddings.unsqueeze(1)
        # print('image shape:', image_embeddings.shape)
        # print('gene pos:', gene_pos.shape)
        # image_embeddings_query_gene = image_embeddings + gene_pos
        # image_embeddings_query_gene = self.gelu(image_embeddings_query_gene)
        # print(image_embeddings_query_gene.shape)
        pred = self.regression(image_embeddings_query_gene)
        # pred = (pred.squeeze() + pred_neighbor) / 2.0
        # print('pred:', pred.shape)
        # print('gold:', rna_value[:,1:].shape)
        loss_predict = F.mse_loss(pred.squeeze(), rna_value[:,1:])
        # loss_predict = F.l1_loss(pred.squeeze(), rna_value[:,1:])
        
        # total_loss = (loss + loss_predict)/2.0
        return loss_predict.mean(dim=0), pred

class AugGeneQueryName(nn.Module):
    def __init__(
        self,
        args,
        
    ):
        super().__init__()
        self.gene_dim = 256
        self.temperature=args.temperature,
        self.gene_num = args.gene_total
        self.image_projection = ProjectionHead(args=args,embedding_dim=args.image_embedding_dim) #aka the input dim, 2048 for resnet50
        # self.spot_projection = ProjectionHead(args=args,embedding_dim=self.gene_dim)
        # self.gene_embed = nn.Linear(1,self.gene_dim)
        self.gene_id_emb = nn.Embedding(self.gene_num, self.gene_dim, padding_idx=self.gene_num-1)# There are 75500 NCBI Gene ID
        # self.spot_pretrained_model = pytorchTransformerModule(max_seq_len=args.max_length+1, dim=self.gene_dim, depth=2, heads=4)
        # self.spot_pretrained_model = Performer(dim=self.gene_dim, depth=1, heads=4, dim_head=128)
        # self.patch_embed = PatchEmbed(224, 16, 3, 128)
        self.patch_embed = ImageEncoder_resnet50(args)
        self.dropout = nn.Dropout(args.dropout)
        self.device = args.current_device
        self.gelu = nn.GELU()
        self.regression = nn.Linear(self.gene_dim,1)
        self.trans = nn.Linear(768, self.gene_dim)

    def forward(self, batch):
        # Getting Image and spot Features
        # image_features = self.image_encoder(batch["image"])
        # print('image input shape:', batch["image"].shape)
        image_patch_features = self.patch_embed(batch["image"].to(self.device))
        # print("pretrain vit image shape: ", image_patch_features.shape)
        x = batch["rna_list"]
        # print(x[0].shape)
        print(x[1].shape)
        # print(x[2].shape)
        # neighbor = batch['neighbor'].to(self.device)
        batch['neighbor_raw'] = batch['neighbor_raw'].reshape(-1, 3, 224, 224)
        neighbor_raw = self.patch_embed(batch['neighbor_raw'].to(self.device))
        
        # print('neighbor raw:', neighbor_raw.shape)
        # rna_id = x[:,1,:].long().requires_grad_(False).to(self.device)
        # rna_mask = x[:,2,:].bool().requires_grad_(False).to(self.device)
        rna_value = x[:,0,:].float().requires_grad_(False).to(self.device)
        # if len(rna_value.shape) < 3:
        #     rna_seq = torch.unsqueeze(rna_value, dim=2)
        #     rna_seq = self.gene_embed(rna_seq)
        # if self.use_genename:
        #     gene_pos = self.sentence_model.encode(batch['gene_name'][:, 1:])
        # gene_pos = self.gene_id_emb(rna_id[:, 1:])
        gene_pos = self.trans(batch['gene_name'].to(self.device))

        # x = rna_seq + gene_pos
        # x = self.dropout(x)
        image_embeddings = self.image_projection(image_patch_features)
        neighbor_raw_encode = self.image_projection(neighbor_raw)
        neighbor_raw_encode = neighbor_raw_encode.reshape(image_embeddings.shape[0], -1, self.gene_dim)
        # img_f = torch.cat([image_embeddings.unsqueeze(1), neighbor_raw_encode], dim=1)
        # img_f = img_f.unsqueeze(1)
        # # print('img_f:', img_f.shape)
        # conv_out = [F.relu(conv(img_f)).squeeze(3) for conv in self.convs_1d]
        # x_max = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in conv_out]
        # x = torch.cat(x_max, 1)
        # x = self.fc(x)
        # print('conv_out:', x.shape)
        x = image_embeddings.unsqueeze(1)
        neighbor_raw_encode = neighbor_raw_encode.unsqueeze(1)
        image_embeddings_query_gene = x + gene_pos
        gene_pos = gene_pos.unsqueeze(2)
        neighbor_raw_encode = neighbor_raw_encode + gene_pos
        image_embeddings_query_gene = self.gelu(image_embeddings_query_gene)
        neighbor_raw_encode = self.gelu(neighbor_raw_encode)
        pred_neighbor = self.regression(neighbor_raw_encode)
        pred_neighbor = pred_neighbor.reshape(x.shape[0], -1, pred_neighbor.shape[1])
        pred_neighbor = torch.mean(pred_neighbor, dim=1)
        # print('pred_neighbor:', pred_neighbor.shape)

        # print('image_embeddings_query_gene shape:', image_embeddings_query_gene.shape)

        # image_embeddings += self.image_projection(neighbor)
        # image_embeddings = image_embeddings.unsqueeze(1)
        # print('image shape:', image_embeddings.shape)
        # print('gene pos:', gene_pos.shape)
        # image_embeddings_query_gene = image_embeddings + gene_pos
        # image_embeddings_query_gene = self.gelu(image_embeddings_query_gene)
        pred = self.regression(image_embeddings_query_gene)
        pred = (pred.squeeze() + pred_neighbor)
        # print('pred:', pred.shape)
        # print('gold:', rna_value[:,1:].shape)
        loss_predict = F.mse_loss(pred, rna_value[:,1:])
        # loss_predict = F.l1_loss(pred.squeeze(), rna_value[:,1:])
        
        # total_loss = (loss + loss_predict)/2.0
        return loss_predict.mean(dim=0), pred



def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")