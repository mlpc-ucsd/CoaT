# CoaT: Co-Scale Conv-Attentional Image Transformers

## Instance Segmentation
This folder contains the instance segmentation experiments using [Mask R-CNN](https://arxiv.org/abs/1703.06870) and Cascade Mask R-CNN (based on [Cascade R-CNN](https://arxiv.org/abs/1906.09756)) framework with CoaT backbone. Specifically, [feature pyramid networks](https://arxiv.org/abs/1612.03144) (FPN) are enabled. We use the [MMDetection](https://github.com/open-mmlab/mmdetection) from [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) as the base implementation and follow its `Mask R-CNN` and `Cascade Mask R-CNN` settings in our experiments.

## Usage
### Environment Preparation
Create the environment and install required packages.
   ```bash
   # Create conda environment.
   conda create -n open-mmlab python=3.7 -y
   conda activate open-mmlab

   # Install PyTorch and MMDetection.
   conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
   pip install openmim
   mim install mmdet
   pip install einops timm
   pip install mmpycocotools

   # Install CUDA 11.1 Update 1.
   # Note: We only need to install CUDA 11.1 and do not need to install the NVIDIA driver in this package.
   #       Please uncheck the driver installation in the intermediate steps.
   wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
   sudo bash cuda_11.1.1_455.32.00_linux.run

   # Install Apex.
   git clone https://github.com/NVIDIA/apex
   cd apex
   CUDA_HOME=/usr/local/cuda-11.1 pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
   ```

### Code and Dataset Preparation
Link models, download COCO 2017 dataset and extract the dataset.
   ```bash
   # Enter the MMDetection folder.
   cd ./tasks/mmdet

   # Link the models folder.
   ln -sfT ../../../src/models ./mmdet/models/backbones/coat

   # Create dataset folder.
   mkdir -p ./data/coco

   # Download the dataset.
   wget http://images.cocodataset.org/zips/train2017.zip -P ./data/coco
   wget http://images.cocodataset.org/zips/val2017.zip -P ./data/coco
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ./data/coco

   # Extract the dataset.
   unzip ./data/COCO/train2017.zip -d ./datasets/coco
   unzip ./data/COCO/val2017.zip -d ./datasets/coco
   unzip ./data/COCO/annotations_trainval2017.zip -d ./datasets/coco
   # After the extraction, you should observe `train2017`, `val2017` and `annotations` folders in ./datasets/coco.
   # More details can be found from https://github.com/open-mmlab/mmdetection/blob/master/docs/1_exist_data_model.md.
   ```

### Evaluate Pre-trained Checkpoint
We provide the CoaT checkpoints pre-trained on the COCO dataset.

#### Mask R-CNN
| Name | Schedule | Bbox AP | Segm AP | SHA-256 (first 8 chars) | URL |
| --- | --- | --- | --- | --- | --- |
| CoaT-Lite Mini | 1x | 41.4 | 38.0 | 85230aa4 |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_lite_mini_mstrain_480-800_adamw_1x_coco_85230aa4.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_lite_mini_mstrain_480-800_adamw_1x_coco_85230aa4.json) |
| CoaT-Lite Mini | 3x | 42.9 | 38.9 | 564e80d7 |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_lite_mini_mstrain_480-800_adamw_3x_coco_564e80d7.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_lite_mini_mstrain_480-800_adamw_3x_coco_564e80d7.json) |
| CoaT-Lite Small | 1x | 45.2 | 40.7 | 4c9c3f44 |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_lite_small_mstrain_480-800_adamw_1x_coco_4c9c3f44.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_lite_small_mstrain_480-800_adamw_1x_coco_4c9c3f44.json) |
| CoaT-Lite Small | 3x | 45.7 | 41.1 | c7da01b6 |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_lite_small_mstrain_480-800_adamw_3x_coco_c7da01b6.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_lite_small_mstrain_480-800_adamw_3x_coco_c7da01b6.json) |
| CoaT Mini | 1x | 45.1 | 40.6 | 30d8566d |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_mini_mstrain_480-800_adamw_1x_coco_30d8566d.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_mini_mstrain_480-800_adamw_1x_coco_30d8566d.json) |
| CoaT Mini | 3x | 46.5 | 41.8 | 67e59b6f |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_mini_mstrain_480-800_adamw_3x_coco_67e59b6f.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_mini_mstrain_480-800_adamw_3x_coco_67e59b6f.json) |
| CoaT Small | 1x | 46.5 | 41.8 | 7396027e |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_small_mstrain_480-800_adamw_1x_coco_7396027e.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_small_mstrain_480-800_adamw_1x_coco_7396027e.json) |
| CoaT Small | 3x | 49.0 | 43.7 | 1152829c |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_small_mstrain_480-800_adamw_3x_coco_1152829c.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_small_mstrain_480-800_adamw_3x_coco_1152829c.json) |

#### Cascade Mask R-CNN
| Name | Schedule | Bbox AP | Segm AP | SHA-256 (first 8 chars) | URL |
| --- | --- | --- | --- | --- | --- |
| CoaT-Lite Small | 1x | 49.1 | 42.5 | 2ab58e20 |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/cascade_mask_rcnn_coat_lite_small_mstrain_480-800_giou_4conv1f_adamw_1x_coco_2ab58e20.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/cascade_mask_rcnn_coat_lite_small_mstrain_480-800_giou_4conv1f_adamw_1x_coco_2ab58e20.json) |
| CoaT-Lite Small | 3x | 48.9 | 42.6 | 3d224926 |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/cascade_mask_rcnn_coat_lite_small_mstrain_480-800_giou_4conv1f_adamw_3x_coco_3d224926.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/cascade_mask_rcnn_coat_lite_small_mstrain_480-800_giou_4conv1f_adamw_3x_coco_3d224926.json) |
| CoaT Small | 1x | 50.4 | 43.5 | 3185cd67 |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/cascade_mask_rcnn_coat_small_mstrain_480-800_giou_4conv1f_adamw_1x_coco_3185cd67.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/cascade_mask_rcnn_coat_small_mstrain_480-800_giou_4conv1f_adamw_1x_coco_3185cd67.json) |
CoaT Small | 3x | 52.2 | 45.1 | 4f7a069e |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/cascade_mask_rcnn_coat_small_mstrain_480-800_giou_4conv1f_adamw_3x_coco_4f7a069e.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/cascade_mask_rcnn_coat_small_mstrain_480-800_giou_4conv1f_adamw_3x_coco_4f7a069e.json) |


The following commands provide an example (CoaT-Lite Mini, 8-GPU) to evaluate the pre-trained checkpoint.
   ```bash
   # Download the pretrained checkpoint.
   mkdir -p ./work_dirs/pretrained
   wget https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/mask_rcnn_coat_lite_mini_mstrain_480-800_adamw_1x_coco_85230aa4.pth -P ./work_dirs/pretrained
   sha256sum ./work_dirs/pretrained/mask_rcnn_coat_lite_mini_mstrain_480-800_adamw_1x_coco_85230aa4.pth  # Make sure it matches the SHA-256 hash (first 8 characters) in the table.

   # Evaluate.
   # Usage: Please see [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) for more details.
   tools/dist_test.sh configs/coat/mask_rcnn_coat_lite_mini_mstrain_480-800_adamw_1x_coco.py './work_dirs/pretrained/mask_rcnn_coat_lite_mini_mstrain_480-800_adamw_1x_coco_85230aa4.pth' 8 --eval bbox segm
   # It should output similar results to the below ones:
   #   Evaluate annotation type *bbox*
   #   DONE (t=23.50s).
   #   Accumulating evaluation results...
   #   DONE (t=4.39s).
   #   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.415
   #   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.640
   #   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.450
   #   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.268
   #   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.449
   #   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.529
   #   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.550
   #   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.550
   #   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.550
   #   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.385
   #   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.585
   #   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.674
   #   ...
   #   Evaluate annotation type *segm*
   #   DONE (t=28.06s).
   #   Accumulating evaluation results...
   #   DONE (t=4.34s).
   #   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.380
   #   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.607
   #   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.406
   #   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.225
   #   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.415
   #   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.512
   #   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.505
   #   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.505
   #   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.505
   #   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.329
   #   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.541
   #   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.653
   ```
   
### Train
The following commands provide an example (CoaT-Lite Mini, 8-GPU) to train the Mask R-CNN w/ CoaT backbone.
   ```bash
   # Usage: Please see [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) for more details.
   tools/dist_train.sh configs/coat/mask_rcnn_coat_lite_mini_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained="../../output/pretrained/coat_lite_mini_6b4a8ae5.pth"
   ```

### Evaluate
The following commands provide an example (CoaT-Lite Mini, 8-GPU) to evaluate the checkpoint after training.
   ```bash
   # Usage: Please see [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) for more details.
   tools/dist_test.sh configs/coat/mask_rcnn_coat_lite_mini_mstrain_480-800_adamw_1x_coco.py './work_dirs/mask_rcnn_coat_lite_mini_mstrain_480-800_adamw_1x_coco/epoch_12.pth' 8 --eval bbox segm
   ```

## Acknowledgment
Thanks to [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) and [MMDetection](https://github.com/open-mmlab/mmdetection) for the Mask R-CNN implementation.