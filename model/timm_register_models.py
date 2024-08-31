import torch
import torch.nn as nn
from functools import partial

# from timm.models.vision_transformer import VisionTransformer, _cfg

from model.conformer import Conformer
from model.mtb.lymph.metatrans import MT_Model
from model.ConvNet import Fourlayers
from timm.models.registry import register_model

@register_model
def Conformer_tiny_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model

@register_model
def Conformer_small_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_small_patch32(pretrained=False, **kwargs):
    model = Conformer(patch_size=32, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_base_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                      num_heads=9, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


@register_model
def mt_small_model_lymph(pretrained=False, **kwargs):
    model = MT_Model(patch_size=16,embed_dim=384, num_heads=6, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def mt_tiny_model_lymph(pretrained=False, **kwargs):
    model = MT_Model(patch_size=16,embed_dim=192, num_heads=6, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def convnet_4(pretrained=False, **kwargs):
    model = Fourlayers(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model
