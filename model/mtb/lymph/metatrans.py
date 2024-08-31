from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from model.transformer.vit_model import Attention, Mlp, Block, PatchEmbed
    
# 别忘了删掉
class MT_Model_store(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None,pretrained_cfg=None,pretrained_cfg_overlay=None):
        """
        四层卷积特征提取，并classification c1
        输入到trans里，并classification c2
        c1 c2 concat融合。
    
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(MT_Model, self).__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1 
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches
        self.meta_feature_extra = meta_feature_extra()
        self.meta_cnn_fc = meta_cnn_fc(num_classes=num_classes)
        
        self.trans_patch_conv = nn.Conv2d(128, embed_dim, kernel_size=1, stride=1, padding=1)

        num_patches = patch_size * patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.meta_trans_fc = nn.Sequential(*[
            nn.Linear(embed_dim, num_classes)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_ratio)
        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # # Classifier head(s)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head_dist = None
        # if distilled:
        #     self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        
        self.fusion_res = list(range(depth//2, depth, 1))
        self.meta_fc = meta_fc(num_classes,num_classes * (len(self.fusion_res) + 1), out_features=embed_dim)


        # Weight init
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # if self.dist_token is not None:
        #     nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

        


    def forward(self, x):
        # # [B, C, H, W] -> [B, num_patches, embed_dim]
        # x = self.patch_embed(x)  # [B, 196, 768]
        # # [1, 1, 768] -> [B, 1, 768]
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # get features map [N,3,224,224] -> [N,128, 14, 14]
        x_base = self.meta_feature_extra(x)
        # 特征提取器分类结果
        x_cnn_res = self.meta_cnn_fc(x_base)

        # [N,128, 14, 14] -->[N, 384, 196]
        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1,2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)

        # x = self.pos_drop(x + self.pos_embed)

        
        trans_fc_res = []
        for i in range(self.depth):
            x_t = self.blocks[i](x_t)
            sk_x = self.meta_trans_fc[i](x_t[:,0])
            # meta_features.append(self.norm(x))
            trans_fc_res.append(sk_x)

        # c = x_cnn_res
        c = trans_fc_res[0]
        for i in self.fusion_res:
            c = torch.cat((c,trans_fc_res[i]),1)

        x = self.meta_fc(c)
        return x

class MT_Model(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None,pretrained_cfg=None,pretrained_cfg_overlay=None):
        """
        四层卷积特征提取，并classification c1
        输入到trans里，并classification c2
        c1 c2 concat融合。
    
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(MT_Model, self).__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1 
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.meta_feature_extra = meta_feature_extra()
        self.meta_cnn_fc = meta_cnn_fc(num_classes=num_classes)
        
        self.trans_patch_conv = nn.Conv2d(128, embed_dim, kernel_size=1, stride=1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.fusion_res = list(range(depth//2, depth, 1))

        self.meta_trans_fc = nn.Sequential(OrderedDict([
            ('linear_{}'.format(i), nn.Linear(embed_dim, num_classes))
            for i in self.fusion_res
        ]))

        self.meta_fc = meta_fc(num_classes,num_classes * (len(self.fusion_res) + 1), out_features=embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)



    def forward(self, x):
        # # [B, C, H, W] -> [B, num_patches, embed_dim]
        # x = self.patch_embed(x)  # [B, 196, 768]
        # # [1, 1, 768] -> [B, 1, 768]
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # get features map [N,3,224,224] -> [N,128, 14, 14]
        x = self.meta_feature_extra(x)
        # 特征提取器分类结果
        x_cnn_res = self.meta_cnn_fc(x)

        # # [N,128, 14, 14] -->[N, 196, 384]
        x_t = self.trans_patch_conv(x).flatten(2).transpose(1,2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
    
        trans_fc_res = {}
        for i in range(self.depth):
            x_t = self.blocks[i](x_t)
            if i in self.fusion_res:
                sk_x = self.meta_trans_fc[self.fusion_res.index(i)](x_t[:,0])
                # meta_features.append(self.norm(x))
                trans_fc_res[i] = sk_x

        c = x_cnn_res
        for i in self.fusion_res:
            c = torch.cat((c,trans_fc_res[i]),1)

        x = self.meta_fc(c)
        return x

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

'''
把每个trans_block的输出 concat一起
'''
class MT_Model_1(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None,pretrained_cfg=None,pretrained_cfg_overlay=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(MT_Model, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1 
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches
        self.cnn_four_layers = CNN_Four_Layers()
        num_patches = patch_size * patch_size


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.fc_learners = nn.Sequential(*[
            nn.Linear(embed_dim, num_classes)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_ratio)
        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        
        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

        self.meta_learner = mtb_learner(num_classes, num_classes * depth, out_features=embed_dim)

    def forward_features(self, x):
        # # [B, C, H, W] -> [B, num_patches, embed_dim]
        # x = self.patch_embed(x)  # [B, 196, 768]
        # # [1, 1, 768] -> [B, 1, 768]


        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        meta_features = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            sk_x = self.fc_learners[i](x[:,0])
            # meta_features.append(self.norm(x))
            meta_features.append(sk_x)
        return meta_features

        # x = self.blocks(x)
        # x = self.norm(x)
        # x = self.head_drop(x)
        # if self.dist_token is None:
        #     return self.pre_logits(x[:, 0])
        # else:
        #     return x[:, 0], x[:, 1]

    def forward(self, x):
        meta_features = self.forward_features(x)
        c = meta_features[0]
        for i in range(len(meta_features)-1):
            c = torch.cat((c, meta_features[i+1]), 1)
        x = self.meta_learner(c)

        # x = self.forward_features(x)
        # if self.head_dist is not None:
        #     x, x_dist = self.head(x[0]), self.head_dist(x[1])
        #     if self.training and not torch.jit.is_scripting():
        #         # during inference, return the average of both classifier predictions
        #         return x, x_dist
        #     else:
        #         return (x + x_dist) / 2
        # else:
        #     x =  self.head(x)
        return x
# patch16 224
def mt_model(num_classes: int = 1000, depth: int = 12):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = MT_Model(img_size=224,
                    patch_size=16,
                    embed_dim=768,
                    depth=depth,
                    num_heads=12,                    
                    representation_size=None,
                    num_classes=num_classes)
    return model


class meta_fc(nn.Module):
    def __init__(self, num_classes, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features # 这层要不小一点，参数少一点会好一点？
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(out_features)
        
        self.num_features = out_features
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.pre_logits = nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.norm(x)
        # x = self.pre_logits(x[:,0])
        x = self.head(x)

        return x
    
class trans_fc(nn.Module):
    def __init__(self, num_classes, in_features, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, num_classes)

        # self.act = act_layer()
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc(x)

        return x

# 元特征提取器
class meta_feature_extra(nn.Module):
    def __init__(self):
        super(meta_feature_extra, self).__init__() 
        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积层
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三层卷积层
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第四层卷积层
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # # 全连接层
        # self.fc1 = nn.Linear(128 * 14 * 14, 256)
        # self.bn5 = nn.BatchNorm1d(256)
        # self.relu5 = nn.ReLU()
        # self.fc2 = nn.Linear(256, 2)  # num_classes是输出类别数

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        # x = x.view(x.size(0), -1)  # 将特征图展平
        # x = self.relu5(self.bn5(self.fc1(x)))
        # x = self.fc2(x)
        return x
    
class meta_cnn_fc(nn.Module):
    def __init__(self, num_classes):
        super(meta_cnn_fc, self).__init__() 
        # 全连接层
        self.fc1 = nn.Linear(128 * 14 * 14, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)  # num_classes是输出类别数

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将特征图展平
        x = self.relu5(self.bn5(self.fc1(x)))
        x = self.fc2(x)
        return x

    
if __name__ == '__main__':
    device = torch.device('cuda')

    model = MT_Model(num_classes=2,embed_dim=384, num_heads=6).to(device)
    input = torch.randn(4, 3, 224, 224).to(device)

    target = torch.tensor([1,1,0,1]).to(device)

    output = model(input)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output,target)

    grad = torch.autograd.grad(loss, model.parameters())
    print('nihao')
