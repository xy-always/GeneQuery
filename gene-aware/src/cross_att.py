import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# from visualizer import get_local
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    # @get_local('attn')
    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        adder = (1.0 - mask) * -10000.0
        adder = rearrange(adder, 'b n -> b () n ()')
        # adder = (1.0 - )) * -10000.0
        attn = self.attend(dots+adder)
        # print(attn.shape)
        # quit()
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        print('use transformer to catch the gene attention...')
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

    def forward(self, x, mask=None):
        x = self.dropout(x)
        x = self.transformer(x, mask=mask)
        x = self.to_latent(x)
        return x

class Transformer_fuse(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.trans = nn.Linear(dim, dim//2)

    def fuse(self, image_embeds, gene_embeds, fuse_method):
        if fuse_method == 'add':
            return image_embeds + gene_embeds
        elif fuse_method == 'concat':
            if image_embeds.size(-1) != gene_embeds.size(-1):
                image_embeds = self.trans(image_embeds)
            return torch.cat((gene_embeds, image_embeds), dim=-1)
        
    def forward(self, x, mask, fuse_method):
        image_embeds, gene_embeds = x
        for attn, ff in self.layers:
            image_embeds = self.fuse(image_embeds, gene_embeds, fuse_method=fuse_method)
            image_embeds = attn(image_embeds, mask=mask) + image_embeds
            image_embeds = ff(image_embeds) + image_embeds
        return image_embeds

class ViT_fuse(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer_fuse(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

    def forward(self, x, mask=None, fuse_method='add'):
        # x = self.dropout(x)
        x = self.transformer(x, mask=mask, fuse_method=fuse_method)
        x = self.to_latent(x)
        return x