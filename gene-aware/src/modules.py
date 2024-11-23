import math
import timm
import torch
from torch import nn
from transformers.pytorch_utils import apply_chunking_to_forward
from typing import Optional, Tuple

class ImageEncoder(nn.Module):
    def __init__(
        self, 
        config
    ):
        super().__init__()
        pretrained_image_encoder = config.pretrained_image_encoder
        trainable_image_encoder = config.trainable_image_encoder
        num_classes = config.num_classes
        global_pool = config.global_pool
        
        self.model = timm.create_model(
            config.image_encoder_name, 
            pretrained_image_encoder, 
            num_classes=num_classes, 
            global_pool=global_pool
        )
        for p in self.model.parameters():
            p.requires_grad = trainable_image_encoder

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
        self.model=timm.create_model(
            self.model_name, pretrained=self.pretrained, num_classes=0, global_pool=None)
        for name, p in self.model.named_parameters():
            p.requires_grad = self.trainable

    def forward(self, x):
        im_encode =  self.model(x)
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
        self.model_name='vit_base_patch32_224'
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
            p.requires_grad = self.trainable

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
            self.model_name,pretrained=True,num_classes=0, pretrained_cfg_overlay=dict(file="../pretrained_image_model/timm/vit_base_patch32_224.augreg_in21k_ft_in1k.bin"))
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
            self.model_name, pretrained=True, num_classes=0, global_pool=None, pretrained_cfg_overlay=dict(file="../pretrained_image_model/timm/mae_pretrain_vit_base.pth"))
        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, x):
        return self.model.forward_features(x)

# Gene Modules

class GeneSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_gene_attention_heads
        self.attention_head_size = int(config.gene_projection_dim / config.num_gene_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.gene_projection_dim, self.all_head_size)
        self.key = nn.Linear(config.gene_projection_dim, self.all_head_size)
        self.value = nn.Linear(config.gene_projection_dim, self.all_head_size)
        self.dropout = nn.Dropout(config.gene_dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer,)
        return outputs

class GeneSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.gene_projection_dim, config.gene_projection_dim)
        self.LayerNorm = nn.LayerNorm(config.gene_projection_dim, eps=config.gene_layer_norm_eps)
        self.dropout = nn.Dropout(config.gene_dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class GeneAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = GeneSelfAttention(config)
        self.output = GeneSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class GeneIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.gene_projection_dim, config.gene_intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class GeneOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.gene_intermediate_size, config.gene_projection_dim)
        self.LayerNorm = nn.LayerNorm(config.gene_projection_dim, eps=config.gene_layer_norm_eps)
        self.dropout = nn.Dropout(config.gene_dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class GeneTransLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GeneAttention(config)
        self.intermediate = GeneIntermediate(config)
        self.output = GeneOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

# class GeneMLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.mlp = GeneIntermediate(config)
#         self.output = GeneOutput(config)
    
#     def forward(self, x):
#         res = x
#         x = self.mlp(x)
#         x = self.output(x, res)
#         return x

class ProjectionHead(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(config.gene_dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x




