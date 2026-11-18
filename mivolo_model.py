"""MiVOLO: the network that estimates age and gender from a face crop and a body crop.

Vendored from MiVOLO, https://github.com/WildChlamydia/MiVOLO (commit 37475e3f), which is
Apache-2.0: Copyright 2023, Irina Tolstykh, Maxim Kuprashevich. Its licence is in
third_party/MiVOLO/LICENSE. MiVOLO's model code is itself adapted from timm,
https://github.com/huggingface/pytorch-image-models (Apache-2.0).

Taken from mivolo/model/mivolo_model.py, mivolo/model/cross_bottleneck_attn.py and the
preprocessing half of mivolo/data/misc.py. Why vendored instead of installed: the MiVOLO
package pins timm 0.8.13.dev0, torch 2.5.1 and transformers 4.51.0, and open_clip and the
SigLIP 2 tokenizer need the current ones. What changed on the way in:

- MiVOLOModel calls VOLO.__init__ with keyword arguments. MiVOLO passed them by position,
  and timm 1.0 inserted `pos_drop_rate` into the middle of that signature, which shifts
  every later argument by one (`post_layers` would land in `norm_layer`).
- The model is built directly instead of through timm's registry and
  build_model_with_cfg, and only mivolo_d1_384 -- the one checkpoint served -- is kept.
- Training-only parts are gone: the FDS hook in forward_head and the timm pretrained
  configs pointing at VOLO ImageNet weights.
- prepare_classification_images() no longer takes a device and returns a float32 tensor
  on the CPU; the caller moves and casts it.

Imported only in full mode: it needs timm and OpenCV at module level.
"""

from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from timm.layers.bottleneck_attn import PosEmbedRel
from timm.layers.helpers import make_divisible
from timm.layers.mlp import Mlp
from timm.models.volo import VOLO

# ImageNet statistics, what timm resolves for mivolo_d1_384 and what the checkpoint's
# preprocessor_config.json lists.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CrossBottleneckAttn(nn.Module):
    """Cross attention between the face and the body feature maps (MiVOLO's addition)."""

    def __init__(
        self,
        dim,
        dim_out=None,
        feat_size=None,
        stride=1,
        num_heads=4,
        dim_head=None,
        qk_ratio=1.0,
        qkv_bias=False,
        scale_pos_embed=False,
    ):
        super().__init__()
        assert feat_size is not None, "A concrete feature size matching expected input (H, W) is required"
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0

        self.num_heads = num_heads
        self.dim_head_qk = dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk**-0.5
        self.scale_pos_embed = scale_pos_embed

        self.qkv_f = nn.Conv2d(dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=qkv_bias)
        self.qkv_p = nn.Conv2d(dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=qkv_bias)

        # Only relative position embedding is supported.
        self.pos_embed = PosEmbedRel(feat_size, dim_head=self.dim_head_qk, scale=self.scale)

        self.norm = nn.LayerNorm([self.dim_out_v * 2, *feat_size])
        mlp_ratio = 4
        self.mlp = Mlp(
            in_features=self.dim_out_v * 2,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            out_features=dim_out,
            drop=0,
            use_conv=True,
        )

        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.qkv_f.weight, std=self.qkv_f.weight.shape[1] ** -0.5)  # fan-in
        trunc_normal_(self.qkv_p.weight, std=self.qkv_p.weight.shape[1] ** -0.5)  # fan-in
        trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def get_qkv(self, x, qvk_conv):
        B, C, H, W = x.shape

        x = qvk_conv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W

        q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)

        q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
        v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)

        return q, k, v

    def apply_attn(self, q, k, v, B, H, W, dropout=None):
        if self.scale_pos_embed:
            attn = (q @ k + self.pos_embed(q)) * self.scale  # B * num_heads, H * W, H * W
        else:
            attn = (q @ k) * self.scale + self.pos_embed(q)
        attn = attn.softmax(dim=-1)
        if dropout:
            attn = dropout(attn)

        out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        return out

    def forward(self, x):
        B, C, H, W = x.shape

        dim = int(C / 2)
        x1 = x[:, :dim, :, :]
        x2 = x[:, dim:, :, :]

        assert H == self.pos_embed.height
        assert W == self.pos_embed.width

        q_f, k_f, v_f = self.get_qkv(x1, self.qkv_f)
        q_p, k_p, v_p = self.get_qkv(x2, self.qkv_p)

        # person to face
        out_f = self.apply_attn(q_f, k_p, v_p, B, H, W)
        # face to person
        out_p = self.apply_attn(q_p, k_f, v_f, B, H, W)

        x_pf = torch.cat((out_f, out_p), dim=1)  # B, dim_out * 2, H, W
        x_pf = self.norm(x_pf)
        x_pf = self.mlp(x_pf)  # B, dim_out, H, W

        out = self.pool(x_pf)
        return out


def _stem_output_size(input_size, stem):
    output_size = input_size
    for module in stem:
        if isinstance(module, nn.Conv2d):
            output_size = [
                (
                    (output_size[i] + 2 * module.padding[i] - module.dilation[i] * (module.kernel_size[i] - 1) - 1)
                    // module.stride[i]
                )
                + 1
                for i in range(2)
            ]
    return output_size


def _conv_output_size(input_shape, conv_layer):
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    return [
        ((input_shape[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i]) + 1
        for i in range(2)
    ]


class PatchEmbed(nn.Module):
    """Two convolutional stems, one per crop, joined by CrossBottleneckAttn.

    The 6 input channels are the face crop (0-2) and the body crop (3-5), each RGB and
    ImageNet-normalised. Only the two-input (`in_chans=6`) variant is kept.
    """

    def __init__(self, img_size=224, stem_stride=1, patch_size=8, in_chans=6, hidden_dim=64, embed_dim=384):
        super().__init__()
        assert patch_size in [4, 8, 16]
        assert in_chans == 6, "only the face + body model is vendored"

        self.conv1 = self.create_stem(stem_stride, 3, hidden_dim)
        self.conv2 = self.create_stem(stem_stride, 3, hidden_dim)

        self.proj1 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride)
        self.proj2 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride)

        stem_out_shape = _stem_output_size((img_size, img_size), self.conv1)
        self.proj_output_size = _conv_output_size(stem_out_shape, self.proj1)

        self.map = CrossBottleneckAttn(embed_dim, dim_out=embed_dim, num_heads=1, feat_size=self.proj_output_size)

        self.patch_dim = img_size // patch_size
        self.num_patches = self.patch_dim**2

    @staticmethod
    def create_stem(stem_stride, in_chans, hidden_dim):
        return nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride, padding=3, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.proj1(self.conv1(x[:, :3]))
        x2 = self.proj2(self.conv2(x[:, 3:]))
        x = torch.cat([x1, x2], dim=1)
        return self.map(x)


class MiVOLOModel(VOLO):
    """VOLO with MiVOLO's two-input patch embedding."""

    def __init__(
        self,
        layers,
        img_size=224,
        in_chans=6,
        num_classes=3,
        patch_size=8,
        stem_hidden_dim=64,
        embed_dims=None,
        num_heads=None,
        **kwargs,
    ):
        super().__init__(
            layers=layers,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=num_classes,
            patch_size=patch_size,
            stem_hidden_dim=stem_hidden_dim,
            embed_dims=embed_dims,
            num_heads=num_heads,
            **kwargs,
        )

        im_size = img_size[0] if isinstance(img_size, tuple) else img_size
        self.patch_embed = PatchEmbed(
            img_size=im_size,
            stem_stride=2,
            patch_size=patch_size,
            in_chans=in_chans,
            hidden_dim=stem_hidden_dim,
            embed_dim=embed_dims[0],
        )

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def forward_features(self, x):
        x = self.patch_embed(x).permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C
        x = self.forward_tokens(x)
        if self.post_network is not None:
            x = self.forward_cls(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == "avg":
            out = x.mean(dim=1)
        elif self.global_pool == "token":
            out = x[:, 0]
        else:
            out = x
        if pre_logits:
            return out

        out = self.head(out)
        if self.aux_head is not None:
            # classes from all feature tokens, see token labeling
            aux = self.aux_head(x[:, 1:])
            out = out + 0.5 * aux.max(1)[0]
        return out

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def mivolo_d1_384(num_classes: int = 3, in_chans: int = 6) -> MiVOLOModel:
    """The architecture of the mivolo_v2 checkpoint (~57M parameters)."""
    return MiVOLOModel(
        layers=(4, 4, 8, 2),
        img_size=384,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dims=(192, 384, 384, 384),
        num_heads=(6, 12, 12, 12),
    )


def class_letterbox(im, new_shape=(640, 640), color=(0, 0, 0), scaleup=True):
    """Resize keeping the aspect ratio, then pad to `new_shape` with `color`."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    if im.shape[0] == new_shape[0] and im.shape[1] == new_shape[1]:
        return im

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im


def prepare_classification_images(
    img_list: Sequence[Optional[np.ndarray]],
    target_size: int = 384,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Optional[torch.Tensor]:
    """BGR crops (or None) -> a float32 N x 3 x target x target batch, RGB and normalised.

    A missing crop becomes an all-zero image before normalisation, which is what MiVOLO
    feeds for a face without a body (or a body without a face).
    """
    mean_arr = np.asarray(mean, dtype=np.float64)
    std_arr = np.asarray(std, dtype=np.float64)
    prepared: List[torch.Tensor] = []

    for img in img_list:
        if img is None:
            blank = torch.zeros((3, target_size, target_size), dtype=torch.float32)
            blank = (blank - torch.tensor(mean, dtype=torch.float32)[:, None, None]) / torch.tensor(
                std, dtype=torch.float32
            )[:, None, None]
            prepared.append(blank.unsqueeze(0))
            continue
        img = class_letterbox(img, new_shape=(target_size, target_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img / 255.0
        img = (img - mean_arr) / std_arr
        img = img.astype(dtype=np.float32)

        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        prepared.append(torch.from_numpy(img).unsqueeze(0))

    if not prepared:
        return None
    return torch.cat(prepared)
