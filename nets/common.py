import logging
import torch
from torch import nn
from torch.nn import functional as F


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class CR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class CGR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CGR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.gn = nn.GroupNorm(32, out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x


class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CBR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class FPNExtractor(nn.Module):
    def __init__(self, c3, c4, c5, inner_channel=256, bias=True):
        super(FPNExtractor, self).__init__()
        self.c3_latent = nn.Conv2d(c3, inner_channel, 1, 1, 0, bias=bias)
        self.c4_latent = nn.Conv2d(c4, inner_channel, 1, 1, 0, bias=bias)
        self.c5_latent = nn.Conv2d(c5, inner_channel, 1, 1, 0, bias=bias)
        self.c5_to_c6 = nn.Conv2d(c5, inner_channel, 3, 2, 1, bias=bias)
        self.c6_to_c7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(inner_channel, inner_channel, 3, 2, 1, bias=bias)
        )

    def forward(self, xs):
        c3, c4, c5 = xs
        f3 = self.c3_latent(c3)
        f4 = self.c4_latent(c4)
        f5 = self.c5_latent(c5)
        f6 = self.c5_to_c6(c5)
        f7 = self.c6_to_c7(f6)
        return [f3, f4, f5, f6, f7]


class FPN(nn.Module):
    def __init__(self, c3, c4, c5, out_channel, bias=True):
        super(FPN, self).__init__()
        self.fpn_extractor = FPNExtractor(c3, c4, c5, out_channel, bias)
        self.p3_out = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=bias)
        self.p4_out = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=bias)
        self.p5_out = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=bias)

    def forward(self, xs):
        f3, f4, f5, f6, f7 = self.fpn_extractor(xs)
        p5 = self.p5_out(f5)
        f4 = f4 + nn.UpsamplingBilinear2d(size=(f4.shape[2:]))(f5)
        p4 = self.p4_out(f4)
        f3 = f3 + nn.UpsamplingBilinear2d(size=(f3.shape[2:]))(f4)
        p3 = self.p3_out(f3)
        return [p3, p4, p5, f6, f7]
# class FPN(nn.Module):
#     def __init__(self, c3, c4, c5, out_channel=256):
#         super(FPN, self).__init__()
#
#         self.bones = nn.Sequential(
#             nn.Conv2d(c5, out_channel, 1, 1, 0),  # 0
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1),  # 1
#
#             nn.Conv2d(c4, out_channel, 1, 1, 0),  # 2
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1),  # 3
#
#             nn.Conv2d(c3, out_channel, 1, 1, 0),  # 4
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1),  # 5
#             nn.Conv2d(c5, out_channel, 3, 2, 1),  # 6
#             nn.ReLU(),  # 7
#             nn.Conv2d(out_channel, out_channel, 3, 2, 1),  # 8
#         )
#
#     def forward(self, x):
#         c3, c4, c5 = x
#         f5 = self.bones[0](c5)
#         p5 = self.bones[1](f5)
#
#         f4 = self.bones[2](c4) + nn.UpsamplingNearest2d(size=(c4.shape[2:]))(f5)
#         p4 = self.bones[3](f4)
#
#         f3 = self.bones[4](c3) + nn.UpsamplingNearest2d(size=(c3.shape[2:]))(f4)
#         p3 = self.bones[5](f3)
#
#         p6 = self.bones[6](c5)
#
#         p7 = self.bones[8](self.bones[7](p6))
#
#         return [p3, p4, p5, p6, p7]
