import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List

from util.misc import NestedTensor

from ..position_encoding import build_position_encoding
from .models.coat import coat_tiny, coat_mini, coat_lite_tiny, coat_lite_mini, coat_lite_small 


__all__ = [
    "build_coat_backbone",
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


class CoaT_Backbone(nn.Module):
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

        # Extract strides and number of channels for output features.
        self.strides = [self._out_feature_strides[x] for x in self._out_features]
        self.num_channels = [self._out_feature_channels[x] for x in self._out_features]

        # Create backbone model with pretrained weights.
        self.backbone = model_func(return_interm_layers=True, out_features=out_features, **model_kwargs)
        if pretrained_weights_path:
            load_pretrained_weights(self.backbone, pretrained_weights_path)
        
    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone.forward_features(tensor_list.tensors)   # NOTE tensor_list.tensors is merged tensors (padded).
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out     # Returns a dict of NestedTensors, containing the features and corresponding (interpolated) masks.


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_coat_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    # Always train the backbone with 4 feature levels (including 3 intermediate outputs from backbone).
    assert train_backbone and return_interm_layers and args.num_feature_levels == 4
    backbone = CoaT_Backbone(
        name=args.backbone, 
        out_features=["x2_nocls", "x3_nocls", "x4_nocls"],
        pretrained_weights_path=args.backbone_weights,
        model_kwargs=eval(str(args.backbone_kwargs).replace("`", "")),
    )
    model = Joiner(backbone, position_embedding)
    return model