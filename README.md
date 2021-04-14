# CoaT: Co-Scale Conv-Attentional Image Transformers

## Introduction
This repository contains the official code and pretrained models for [CoaT: Co-Scale Conv-Attentional Image Transformers](http://arxiv.org/abs/2104.06399). It introduces (1) a co-scale mechanism to realize fine-to-coarse, coarse-to-fine and cross-scale attention modeling and (2) an efficient conv-attention module to realize relative position encoding in the factorized attention.

<img src="./figures/model-acc.svg" alt="Model Accuracy" width="600" />

For more details, please refer to [CoaT: Co-Scale Conv-Attentional Image Transformers](http://arxiv.org/abs/2104.06399) by [Weijian Xu*](https://weijianxu.com/), [Yifan Xu*](https://yfxu.com/), [Tyler Chang](https://tylerachang.github.io/), and [Zhuowen Tu](https://pages.ucsd.edu/~ztu/).

## Updates
Code will be released soon.

## Citation
```
@misc{xu2021coscale,
      title={Co-Scale Conv-Attentional Image Transformers}, 
      author={Weijian Xu and Yifan Xu and Tyler Chang and Zhuowen Tu},
      year={2021},
      eprint={2104.06399},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
This repository is released under the Apache License 2.0. License can be found in [LICENSE](LICENSE) file.

## Acknowledgment
Thanks to [DeiT](https://github.com/facebookresearch/deit) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) for a clear and data-efficient implementation of [ViT](https://openreview.net/forum?id=YicbFdNTTy). Thanks to [lucidrains' implementation](https://github.com/lucidrains/lambda-networks) of [Lambda Networks](https://openreview.net/forum?id=xTJEN-ggl1b) and [CPVT](https://github.com/Meituan-AutoML/CPVT).