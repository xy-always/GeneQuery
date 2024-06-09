import torch
from torch import nn
import timm
import config as CFG


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, 
        args
    ):
        super().__init__()
        self.model_name='resnet50'
        self.pretrained=args.pretrained
        self.trainable=args.trainable
        self.model = timm.create_model(
            self.model_name, self.pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_resnet50(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, 
        args=None
    ):
        super().__init__()
        self.model_name='resnet50'
        if args is not None:
            self.pretrained=args.pretrained
            self.trainable=args.trainable
        else:
            self.pretrained=True
            self.trainable=False
        self.model=timm.create_model(self.model_name, pretrained=True, num_classes=0, global_pool=None, pretrained_cfg_overlay=dict(file="/root/autodl-tmp/xy/BLEEP/pretrained_image_model/timm/resnet50-19c8e357.pth"))
        # self.model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=0, global_pool="avg", pretrained_cfg_overlay=dict(file="/root/autodl-tmp/xy/BLEEP/pretrained_image_model/timm/resnet50-19c8e357.pth"))
        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, x):
        im_encode =  self.model(x)
        # print('im_encode shape:', im_encode.shape)
        return im_encode
    
class ImageEncoder_resnet101(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, 
        args
    ):
        super().__init__()
        self.model_name='resnet101'
        self.pretrained=args.pretrained
        self.trainable=args.trainable
        self.model = timm.create_model(
            self.model_name, self.pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_resnet152(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, 
        args
    ):
        super().__init__()
        self.model_name='resnet152'
        self.pretrained=args.pretrained
        self.trainable=args.trainable
        self.model = timm.create_model(
            self.model_name, self.pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_ViT(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self,
        args
    ):
        super().__init__()
        self.model_name='vit'
        self.pretrained=args.pretrained
        self.trainable=args.trainable
        self.model = timm.create_model(
            self.model_name, self.pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, x):
        return self.model(x)
    

class ImageEncoder_CLIP(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, 
        args
    ):
        super().__init__()
        self.model_name="vit_base_patch32_224_clip_laion2b"
        self.pretrained=args.pretrained
        self.trainable=args.trainable
        self.model = timm.create_model(
            self.model_name, self.pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class ImageEncoder_ViT_L(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, 
        args
    ):
        super().__init__()
        # self.model = timm.create_model(
        #     model_name, pretrained, num_classes=0, global_pool="avg"
        # )
        self.model_name="vit_large_patch32_224_in21k"
        self.pretrained=args.pretrained
        self.trainable=args.trainable
        self.model=timm.create_model(
            self.model_name,pretrained=True,num_classes=0, pretrained_cfg_overlay=dict(file="/root/autodl-tmp/xy/BLEEP/pretrained_image_model/timm/vit_base_patch32_224.augreg_in21k_ft_in1k.bin"))
        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, x):
        return self.model(x)

class ImageEncoder_Mae_ViT(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self,
        args
    ):
        super().__init__()
        # self.model = timm.create_model(
        #     model_name, pretrained, num_classes=0, global_pool="avg"
        # )
        print('init mae model')
        self.model_name="vit_base_patch16_224.mae"
        self.pretrained=args.pretrained
        self.trainable=args.trainable
        self.model=timm.create_model(
            self.model_name, pretrained=True, num_classes=0, global_pool=None, pretrained_cfg_overlay=dict(file="/root/autodl-tmp/xy/BLEEP/pretrained_image_model/timm/mae_pretrain_vit_base.pth"))
        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, x):
        return self.model.forward_features(x)

#  'vit_base_patch32_224',
#  'vit_base_patch32_224_clip_laion2b',
#  'vit_base_patch32_224_in21k',
#  'vit_base_patch32_224_sam',
#  'vit_base_patch16_224.mae',   


# class SpotEncoder(nn.Module):
#     #to change...
#     def __init__(self, model_name=CFG.spot_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
#         super().__init__()
#         if pretrained:
#             self.model = DistilBertModel.from_pretrained(model_name)
#         else:
#             self.model = DistilBertModel(config=DistilBertConfig())
            
#         for p in self.model.parameters():
#             p.requires_grad = trainable

#         # we are using the CLS token hidden representation as the sentence's embedding
#         self.target_token_idx = 0

#     def forward(self, input_ids, attention_mask):
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = output.last_hidden_state
#         return last_hidden_state[:, self.target_token_idx, :]



class ProjectionHead(nn.Module):
    def __init__(
        self,
        args,
        embedding_dim
    ):
        super().__init__()
        self.projection_dim=args.projection_dim
        dropout=args.dropout
        print(embedding_dim)
        print(self.projection_dim)
        self.projection = nn.Linear(embedding_dim, self.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.projection_dim, self.projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x