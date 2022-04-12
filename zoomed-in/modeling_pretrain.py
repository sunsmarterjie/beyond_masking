# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_mae_base_patch16_224',
    'pretrain_mae_large_patch16_224',
]


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def random_masking(self, x, mask_ratio=0.75):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_features(self, x, mask=0.75):
        x = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        if mask > 0.:
            x_vis, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio=mask)
        else:
            x_vis, mask, ids_restore = x, 1.0, None

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis, mask, ids_restore, ids_keep

    def forward(self, x, mask):
        x, mask, ids_restore, ids_keep = self.forward_features(x, mask)
        x = self.head(x)
        return x, mask, ids_restore, ids_keep


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=4,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.head(self.norm(x))

        return x


class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=768,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        self.inp = 288

        self.resize_to224 = lambda x: F.interpolate(x, (224, 224), mode='area')
        self.resize_to288 = lambda x: F.interpolate(x, (self.inp, self.inp), mode='area')

        self.loss_func = nn.MSELoss()

        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]

        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        if x.shape[1] > 196:
            x = x[:, 1:, :]
        p = 16
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_loss(self, imgs_inp, imgs_ori, pred_x, mask):

        device = torch.device('cuda')
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]

        unnorm_imgs = imgs_inp * std + mean  # in [0, 1]
        imgs_squeeze = rearrange(unnorm_imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=16, p2=16)
        imgs_norm = (imgs_squeeze - imgs_squeeze.mean(dim=-2, keepdim=True)) / (
                imgs_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        target_center = rearrange(imgs_norm, 'b n p c -> b n (p c)')

        sv = 224
        pred_imgs = self.unpatchify(pred_x)
        pred_center = self.resize_to288(pred_imgs)[:, :, (self.inp - sv) // 2:-(self.inp - sv) // 2,
                      (self.inp - sv) // 2:-(self.inp - sv) // 2]
        pred_center = self.patchify(pred_center)
        loss_mae = (pred_center - target_center) ** 2
        loss_mae = loss_mae.mean(dim=-1)
        loss_mae = (loss_mae * mask).sum() / mask.sum()

        imgs_ori = self.resize_to224(imgs_ori)
        unnorm_imgs = imgs_ori * std + mean  # in [0, 1]
        imgs_squeeze = rearrange(unnorm_imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=16, p2=16)
        imgs_norm = (imgs_squeeze - imgs_squeeze.mean(dim=-2, keepdim=True)) / (
                imgs_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        target_overall = rearrange(imgs_norm, 'b n p c -> b n (p c)')
        target_overall = self.unpatchify(target_overall)
        target_overall = self.resize_to288(target_overall)
        pred_imgs = self.resize_to288(pred_imgs)
        loss_edge = self.loss_func(pred_imgs[:, :, :(self.inp - sv) // 2, :],
                                   target_overall[:, :, :(self.inp - sv) // 2, :]) + \
                    self.loss_func(pred_imgs[:, :, -(self.inp - sv) // 2:, :],
                                   target_overall[:, :, -(self.inp - sv) // 2:, :]) + \
                    self.loss_func(pred_imgs[:, :, (self.inp - sv) // 2: -(self.inp - sv) // 2, :(self.inp - sv) // 2],
                                   target_overall[:, :, (self.inp - sv) // 2: -(self.inp - sv) // 2,
                                   :(self.inp - sv) // 2]) + \
                    self.loss_func(pred_imgs[:, :, (self.inp - sv) // 2: -(self.inp - sv) // 2, -(self.inp - sv) // 2:],
                                   target_overall[:, :, (self.inp - sv) // 2: -(self.inp - sv) // 2,
                                   -(self.inp - sv) // 2:])

        return (loss_mae, loss_edge)

    def forward(self, imgs_inp, imgs_ori=None, mask=0.75):
        mask_value = mask
        x_vis, mask, ids_restore, ids_keep = self.encoder(imgs_inp, mask)  # [B, N_vis, C_e]

        x = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]

        # append mask tokens to sequence
        if mask_value > 0.:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            # append mask tokens to sequence

        # add pos embed
        expand_pos_embed = self.decoder_pos_embed.expand(imgs_inp.size(0), -1, -1).type_as(x).to(
            x.device).clone().detach()
        x = x + expand_pos_embed

        # x = self.decoder(x_full, pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16]
        pred_x = self.decoder(x)  # [B, N_mask, 3 * 16 * 16]

        loss_mae, loss_edge = self.forward_loss(imgs_inp, imgs_ori, pred_x, mask)

        return (loss_mae, loss_edge)


@register_model
def pretrain_mae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_mae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=12,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
