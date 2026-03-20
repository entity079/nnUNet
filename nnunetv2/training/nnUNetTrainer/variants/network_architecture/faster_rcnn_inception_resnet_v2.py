from __future__ import annotations

import pydoc
from typing import Sequence, Type

import torch
from torch import nn
import torch.nn.functional as F


def _locate_type(maybe_type, fallback=None):
    if maybe_type is None:
        return fallback
    if isinstance(maybe_type, str):
        return pydoc.locate(maybe_type)
    return maybe_type


def _ensure_tuple(value, ndim: int):
    if isinstance(value, (tuple, list)):
        if len(value) == ndim:
            return tuple(int(v) for v in value)
        if len(value) == 1:
            return tuple(int(value[0]) for _ in range(ndim))
        raise ValueError(f'Cannot broadcast value {value} to ndim={ndim}')
    return tuple(int(value) for _ in range(ndim))


def _matching_conv_op(ndim: int) -> Type[nn.Module]:
    if ndim == 2:
        return nn.Conv2d
    if ndim == 3:
        return nn.Conv3d
    raise ValueError(f'Only 2D and 3D convolutions are supported, got ndim={ndim}')


def _matching_norm_op(ndim: int) -> Type[nn.Module]:
    return nn.InstanceNorm2d if ndim == 2 else nn.InstanceNorm3d


class ConvNormAct(nn.Module):
    def __init__(
            self,
            conv_op: Type[nn.Module],
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride,
            norm_op: Type[nn.Module] | None,
            norm_op_kwargs: dict | None,
            act_op: Type[nn.Module],
            act_op_kwargs: dict | None,
    ):
        super().__init__()
        ndim = len(stride) if isinstance(stride, (tuple, list)) else (2 if conv_op == nn.Conv2d else 3)
        kernel_size = _ensure_tuple(kernel_size, ndim)
        stride = _ensure_tuple(stride, ndim)
        padding = tuple(k // 2 for k in kernel_size)
        layers = [
            conv_op(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=norm_op is None,
            )
        ]
        if norm_op is not None:
            layers.append(norm_op(out_channels, **(norm_op_kwargs or {})))
        layers.append(act_op(**(act_op_kwargs or {'inplace': True})))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InceptionResidualBlock(nn.Module):
    def __init__(
            self,
            conv_op: Type[nn.Module],
            channels: int,
            kernel_size,
            norm_op: Type[nn.Module] | None,
            norm_op_kwargs: dict | None,
            act_op: Type[nn.Module],
            act_op_kwargs: dict | None,
            scale: float = 0.2,
    ):
        super().__init__()
        ndim = 2 if conv_op == nn.Conv2d else 3
        branch_channels = max(channels // 4, 8)
        kernel_size = _ensure_tuple(kernel_size, ndim)

        self.branch1 = ConvNormAct(
            conv_op, channels, branch_channels, 1, 1, norm_op, norm_op_kwargs, act_op, act_op_kwargs
        )
        self.branch2 = nn.Sequential(
            ConvNormAct(conv_op, channels, branch_channels, 1, 1, norm_op, norm_op_kwargs, act_op, act_op_kwargs),
            ConvNormAct(conv_op, branch_channels, branch_channels, kernel_size, 1, norm_op, norm_op_kwargs,
                        act_op, act_op_kwargs),
        )
        self.branch3 = nn.Sequential(
            ConvNormAct(conv_op, channels, branch_channels, 1, 1, norm_op, norm_op_kwargs, act_op, act_op_kwargs),
            ConvNormAct(conv_op, branch_channels, branch_channels, kernel_size, 1, norm_op, norm_op_kwargs,
                        act_op, act_op_kwargs),
            ConvNormAct(conv_op, branch_channels, branch_channels, kernel_size, 1, norm_op, norm_op_kwargs,
                        act_op, act_op_kwargs),
        )
        self.projection = conv_op(branch_channels * 3, channels, kernel_size=1, bias=True)
        self.activation = act_op(**(act_op_kwargs or {'inplace': True}))
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = torch.cat((self.branch1(x), self.branch2(x), self.branch3(x)), dim=1)
        residual = self.projection(residual)
        return self.activation(x + self.scale * residual)


class BackboneStage(nn.Module):
    def __init__(
            self,
            conv_op: Type[nn.Module],
            in_channels: int,
            out_channels: int,
            stride,
            kernel_size,
            num_blocks: int,
            norm_op: Type[nn.Module] | None,
            norm_op_kwargs: dict | None,
            act_op: Type[nn.Module],
            act_op_kwargs: dict | None,
    ):
        super().__init__()
        blocks = [
            ConvNormAct(conv_op, in_channels, out_channels, kernel_size, stride, norm_op, norm_op_kwargs,
                        act_op, act_op_kwargs)
        ]
        for _ in range(max(1, num_blocks)):
            blocks.append(
                InceptionResidualBlock(conv_op, out_channels, kernel_size, norm_op, norm_op_kwargs,
                                       act_op, act_op_kwargs)
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class PyramidLateral(nn.Module):
    def __init__(self, conv_op: Type[nn.Module], in_channels: int, out_channels: int):
        super().__init__()
        self.proj = conv_op(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PyramidSmoothing(nn.Module):
    def __init__(self, conv_op: Type[nn.Module], channels: int):
        super().__init__()
        ndim = 2 if conv_op == nn.Conv2d else 3
        self.conv = conv_op(channels, channels, kernel_size=_ensure_tuple(3, ndim), padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class _DeepSupervisionDecoderProxy:
    def __init__(self, parent: "FasterRCNNInceptionResNetV2SegmentationModel"):
        self.parent = parent

    @property
    def deep_supervision(self) -> bool:
        return self.parent.deep_supervision

    @deep_supervision.setter
    def deep_supervision(self, enabled: bool):
        self.parent.deep_supervision = enabled

class FasterRCNNInceptionResNetV2SegmentationModel(nn.Module):
    def __init__(
            self,
            input_channels: int,
            num_classes: int,
            features_per_stage: Sequence[int] | None = None,
            kernel_sizes: Sequence[Sequence[int]] | None = None,
            strides: Sequence[Sequence[int]] | None = None,
            n_blocks_per_stage: Sequence[int] | None = None,
            conv_op: str | Type[nn.Module] | None = None,
            norm_op: str | Type[nn.Module] | None = None,
            norm_op_kwargs: dict | None = None,
            nonlin: str | Type[nn.Module] | None = None,
            nonlin_kwargs: dict | None = None,
            backbone_channels: int = 128,
            deep_supervision: bool = True,
            **_: dict,
    ):
        super().__init__()
        if strides is None:
            strides = ((1, 1), (2, 2), (2, 2), (2, 2), (2, 2))
        ndim = len(strides[0])
        conv_op = _locate_type(conv_op, _matching_conv_op(ndim)) or _matching_conv_op(ndim)
        norm_op = _locate_type(norm_op, _matching_norm_op(ndim))
        act_op = _locate_type(nonlin, nn.LeakyReLU) or nn.LeakyReLU

        if features_per_stage is None:
            features_per_stage = (32, 64, 128, 192, 256)
        if kernel_sizes is None:
            kernel_sizes = tuple((_ensure_tuple(3, ndim) for _ in range(len(features_per_stage))))
        if n_blocks_per_stage is None:
            n_blocks_per_stage = (1,) * len(features_per_stage)

        num_feature_levels = min(len(features_per_stage), len(kernel_sizes), len(strides), len(n_blocks_per_stage))
        if num_feature_levels < 2:
            raise ValueError('The custom Faster R-CNN/Inception-ResNet-v2 backbone requires at least two stages.')

        self.deep_supervision = deep_supervision
        self.decoder = _DeepSupervisionDecoderProxy(self)
        self.num_deep_supervision_outputs = num_feature_levels - 1
        self.interp_mode = 'bilinear' if ndim == 2 else 'trilinear'

        stages = []
        in_channels = input_channels
        self.stage_channels = []
        for idx in range(num_feature_levels):
            out_channels = int(features_per_stage[idx])
            stages.append(
                BackboneStage(
                    conv_op=conv_op,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=_ensure_tuple(strides[idx], ndim),
                    kernel_size=_ensure_tuple(kernel_sizes[idx], ndim),
                    num_blocks=int(n_blocks_per_stage[idx]),
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    act_op=act_op,
                    act_op_kwargs=nonlin_kwargs,
                )
            )
            self.stage_channels.append(out_channels)
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)

        self.laterals = nn.ModuleList(
            PyramidLateral(conv_op, c, backbone_channels) for c in self.stage_channels
        )
        self.smooth = nn.ModuleList(PyramidSmoothing(conv_op, backbone_channels) for _ in self.stage_channels)

        fusion_in_channels = backbone_channels * self.num_deep_supervision_outputs
        self.fusion = nn.Sequential(
            ConvNormAct(conv_op, fusion_in_channels, backbone_channels, 3, 1, norm_op, norm_op_kwargs,
                        act_op, nonlin_kwargs),
            conv_op(backbone_channels, num_classes, kernel_size=1, bias=True),
        )
        self.aux_heads = nn.ModuleList(
            conv_op(backbone_channels, num_classes, kernel_size=1, bias=True)
            for _ in range(max(0, self.num_deep_supervision_outputs - 1))
        )

    def _upsample_to(self, x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[2:], mode=self.interp_mode, align_corners=False)

    def forward(self, x: torch.Tensor):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        pyramid = [None] * len(features)
        top_down = None
        for idx in reversed(range(len(features))):
            lateral = self.laterals[idx](features[idx])
            if top_down is not None:
                top_down = self._upsample_to(top_down, lateral)
                lateral = lateral + top_down
            top_down = self.smooth[idx](lateral)
            pyramid[idx] = top_down

        supervised_pyramid = pyramid[:self.num_deep_supervision_outputs]
        highest_resolution = supervised_pyramid[0]
        fused = torch.cat([
            level if i == 0 else self._upsample_to(level, highest_resolution)
            for i, level in enumerate(supervised_pyramid)
        ], dim=1)
        main_logits = self.fusion(fused)

        if not self.deep_supervision:
            return main_logits

        outputs = [main_logits]
        for idx, head in enumerate(self.aux_heads, start=1):
            outputs.append(head(supervised_pyramid[idx]))
        return outputs


def build_faster_rcnn_inception_resnet_v2_segmentation_model(
        input_channels: int,
        num_classes: int,
        **kwargs,
) -> FasterRCNNInceptionResNetV2SegmentationModel:
    return FasterRCNNInceptionResNetV2SegmentationModel(
        input_channels=input_channels,
        num_classes=num_classes,
        **kwargs,
    )
