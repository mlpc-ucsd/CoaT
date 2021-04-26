import torch

from detectron2.layers import (
    ShapeSpec,
)
from detectron2.modeling import Backbone, BACKBONE_REGISTRY, FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .models.coat import coat_tiny, coat_mini, coat_lite_tiny, coat_lite_mini, coat_lite_small 

__all__ = [
    "build_coat_fpn_backbone",
]


def load_pretrained_weights(model, pretrained_weights_path=''):
    """ Load CoaT model weights from pretrained checkpoint. """
    print("CoaT model: Loading weights from {} ...".format(pretrained_weights_path))
    checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
    remove_list = [
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "norm3.weight", "norm3.bias", "norm4.weight", "norm4.bias", 
        "head.weight", "head.bias", "aggregate.weight", "aggregate.bias"
    ]
    model_params = {k: v for k, v in checkpoint["model"].items() if k not in remove_list}
    model.load_state_dict(model_params)


class CoaT_Backbone(Backbone):
    """
    Implement CoaT backbone.
    """
    def __init__(self, name, out_features, pretrained_weights_path, model_kwargs):
        super().__init__()
        self._out_features = out_features
        self._out_feature_strides  = {"x1_nocls": 4,   "x2_nocls": 8,   "x3_nocls": 16,  "x4_nocls": 32}
        
        if name == 'coat_tiny':
            model_func = coat_tiny
            self._out_feature_channels = {"x1_nocls": 152, "x2_nocls": 152, "x3_nocls": 152, "x4_nocls": 152}
        elif name == 'coat_mini':
            model_func = coat_mini
            self._out_feature_channels = {"x1_nocls": 152, "x2_nocls": 216, "x3_nocls": 216, "x4_nocls": 216}
        elif name == 'coat_lite_tiny':
            model_func = coat_lite_tiny
            self._out_feature_channels = {"x1_nocls": 64,  "x2_nocls": 128, "x3_nocls": 256, "x4_nocls": 320}
        elif name == 'coat_lite_mini':
            model_func = coat_lite_mini
            self._out_feature_channels = {"x1_nocls": 64,  "x2_nocls": 128, "x3_nocls": 320, "x4_nocls": 512}
        elif name == 'coat_lite_small':
            model_func = coat_lite_small
            self._out_feature_channels = {"x1_nocls": 64,  "x2_nocls": 128, "x3_nocls": 320, "x4_nocls": 512}
        else:
            raise ValueError()

        self.backbone = model_func(return_interm_layers=True, out_features=out_features, **model_kwargs)
        if pretrained_weights_path:
            load_pretrained_weights(self.backbone, pretrained_weights_path)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"CoaT takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        return self.backbone.forward_features(x)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def build_coat_backbone(cfg):
    """
    Create a CoaT instance from config.

    Args:
        cfg: a detectron2 CfgNode

    Returns:
        A CoaT backbone instance.
    """
    # fmt: off
    name                    = cfg.MODEL.COAT.NAME
    out_features            = cfg.MODEL.COAT.OUT_FEATURES
    pretrained_weights_path = cfg.MODEL.COAT.WEIGHTS
    model_kwargs            = eval(str(cfg.MODEL.COAT.MODEL_KWARGS).replace("`", ""))

    return CoaT_Backbone(name, out_features, pretrained_weights_path, model_kwargs)


@BACKBONE_REGISTRY.register()
def build_coat_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a CoaT w/ FPN backbone.

    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_coat_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone