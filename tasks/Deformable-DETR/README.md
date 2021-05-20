# CoaT: Co-Scale Conv-Attentional Image Transformers

## Object Detection
This folder contains the object detection experiments using [Deformable DETR](https://arxiv.org/abs/2010.04159) framework with CoaT backbone. We use its [official implementation](https://github.com/fundamentalvision/Deformable-DETR) as the base implementation and follow its default settings (with multi-scale) in our experiments.

## Usage
### Environment Preparation
Activate the environment and install required packages. See Deformable DETR's [original readme](./ORIGINAL_README.md) for more details.
   ```bash
   # Activate the environment (assume the conda environment has already been created following the steps for classification task).
   conda activate coat

   # Install the required packages.
   cd ./tasks/Deformable-DETR
   pip install -r ./requirements.txt

   # Build and install MultiScaleDeformableAttention operator.
   # Note: 1. It may requires CUDA installation. In our environment, we install CUDA 11.3 
   #          which is compatible with CUDA 11.0 bundled with PyTorch and RTX 30 series graphic cards.
   #       2. If you found error "no kernel image is available for execution on the device" during training,
   #          please use `pip uninstall MultiScaleDeformableAttention` to remove the installed package,
   #          delete all build folders (e.g. ./build, ./dist and ./*.egg-info), and then re-run `./make.sh`.
   cd ./models/ops
   sh ./make.sh
   cd ../../
   ```

### Code and Dataset Preparation
Please follow the steps in [instance segmentation](../detectron2/README.md) to download COCO 2017 dataset and extract. Here we simply create symbolic links for models and the dataset folder.
   ```bash
   # Enter the Deformable-DETR folder.
   cd ./tasks/Deformable-DETR

   # Create symbolic links.
   # Note: Here we directly create a symbolic link to COCO dataset which has set up for instance segmentation task. You may
   #       refer to the [corresponding readme](../detectron2/README.md) to download COCO dataset in the instance segmentation task first. 
   ln -sfT ../../../../src/models ./models/coat/models
   mkdir -p ./data
   ln -sfT ../../detectron2/datasets/coco ./data/coco
   ```

### Evaluate Pre-trained Checkpoint
We provide the CoaT checkpoints pre-trained on the ImageNet dataset.

| Name | AP | AP50 | AP75 | APS | APM | APL | SHA-256 (first 8 chars) | URL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoaT-Lite Small | 47.0 | 66.5 | 51.2 | 28.8 | 50.3 | 63.3 | 1801ee09 | [model](http://vcl.ucsd.edu/coat/pretrained/tasks/Deformable-DETR/coat_lite_small_deformable_detr_1801ee09.pth), [log](http://vcl.ucsd.edu/coat/pretrained/tasks/Deformable-DETR/coat_lite_small_deformable_detr_1801ee09.txt) |


The following commands provide an example (CoaT-Lite Small) to evaluate the pre-trained checkpoint.
   ```bash
   # Download the pretrained checkpoint.
   # Note: You need to have CoaT-Lite Small checkpoint for classification (coat_lite_small_8d362f48.pth) to run the following evaluation.
   #       Please refer to the [corresponding readme](../../README.md) to download the CoaT-Lite Small checkpoint for classification first.
   mkdir -p ./exps/pretrained
   wget http://vcl.ucsd.edu/coat/pretrained/tasks/Deformable-DETR/coat_lite_small_deformable_detr_1801ee09.pth -P ./exps/pretrained
   sha256sum ./exps/pretrained/coat_lite_small_deformable_detr_1801ee09.pth  # Make sure it matches the SHA-256 hash (first 8 characters) in the table.

   # Evaluate.
   # Usage: Please see [Deformable DETR's document](./ORIGINAL_README.md) for more details.
   GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/coat_lite_small_deformable_detr.sh --resume ./exps/pretrained/coat_lite_small_deformable_detr_1801ee09.pth --eval --batch_size 1
   # It should output similar results to the below ones:
   #   IoU metric: bbox
   #    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.470
   #    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.665
   #    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.512
   #    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.288
   #    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.503
   #    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.633
   #    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.362
   #    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.599
   #    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.641
   #    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.427
   #    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.687
   #    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.830
   ```
   
### Train
The following commands provide an example (CoaT-Lite Small, 8-GPU) to train the Deformable DETR w/ CoaT backbone.
   ```bash
   # Usage: Please see [Deformable DETR's document](./ORIGINAL_README.md) for more details.
   GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/coat_lite_small_deformable_detr.sh
   ```

### Evaluate
The following commands provide an example (CoaT-Lite Small) to evaluate the checkpoint after training.
   ```bash
   # Usage: Please see [Deformable DETR's document](./ORIGINAL_README.md) for more details.
   GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/coat_lite_small_deformable_detr.sh --resume ./exps/coat_lite_small_deformable_detr/checkpoint0049.pth --eval --batch_size 1
   ```

## Acknowledgment
Thanks to Deformable DETR for its [official implementation](https://github.com/fundamentalvision/Deformable-DETR).