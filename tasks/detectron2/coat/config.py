# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN


def add_coat_config(cfg):
    """
    Add config for CoaT.
    """
    _C = cfg

    _C.MODEL.COAT = CN()

    # CoaT model name.
    _C.MODEL.COAT.NAME = ""

    # Output features from CoaT backbone.
    _C.MODEL.COAT.OUT_FEATURES = ["x1_nocls", "x2_nocls", "x3_nocls", "x4_nocls"]

    # Pre-trained weights setting for CoaT.
    _C.MODEL.COAT.WEIGHTS = ""

    # Other arguments for CoaT model.
    _C.MODEL.COAT.MODEL_KWARGS = "{}"