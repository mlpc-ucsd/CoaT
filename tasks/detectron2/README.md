# CoaT: Co-Scale Conv-Attentional Image Transformers

## Instance Segmentation
This folder contains the instance segmentation experiments using [Mask R-CNN](https://arxiv.org/abs/1703.06870) framework with CoaT backbone. Specifically, [feature pyramid networks](https://arxiv.org/abs/1612.03144) (FPN) are enabled. We use the [Detectron2](https://github.com/facebookresearch/detectron2) as the base implementation and follow its `Mask R-CNN w/ FPN 1x` and `3x` settings in our experiments.

## Usage
### Environment Preparation
Activate the environment and install required packages.
   ```bash
   # Activate the environment (assume the conda environment has already been created following the usage on the classification task).
   conda activate coat

   # Install Detectron2 v0.4.
   python -m pip install detectron2==0.4 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
   ```

### Code and Dataset Preparation
Link models, download COCO 2017 dataset and extract the dataset.
   ```bash
   # Enter the Detectron2 folder.
   cd ./tasks/detectron2

   # Link the models folder.
   ln -sfT ../../../src/models ./coat/models

   # Create dataset folder.
   mkdir -p ./datasets/coco

   # Download the dataset.
   wget http://images.cocodataset.org/zips/train2017.zip -P ./datasets/coco
   wget http://images.cocodataset.org/zips/val2017.zip -P ./datasets/coco
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ./datasets/coco

   # Extract the dataset.
   unzip ./data/COCO/train2017.zip -d ./datasets/coco
   unzip ./data/COCO/val2017.zip -d ./datasets/coco
   unzip ./data/COCO/annotations_trainval2017.zip -d ./datasets/coco
   # After the extraction, you should observe `train2017`, `val2017` and `annotations` folders in ./datasets/coco.
   # More details can be found from https://github.com/facebookresearch/detectron2/tree/master/datasets.
   ```

### Evaluate Pre-trained Checkpoint
We provide the CoaT checkpoints pre-trained on the ImageNet dataset.

| Name | Schedule | Bbox AP | Segm AP | SHA-256 (first 8 chars) | URL |
| --- | --- | --- | --- | --- | --- |
| CoaT-Lite Mini | 1x | 39.9 | 36.4 | 5bb8caf8 |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/detectron2/mask_rcnn_coat_lite_mini_FPN_1x_5bb8caf8.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/detectron2/mask_rcnn_coat_lite_mini_FPN_1x_5bb8caf8.json) |
| CoaT-Lite Mini | 3x | 41.8 | 37.7 | 4dec4a9b |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/detectron2/mask_rcnn_coat_lite_mini_FPN_3x_4dec4a9b.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/detectron2/mask_rcnn_coat_lite_mini_FPN_3x_4dec4a9b.json) |
| CoaT-Lite Small | 1x | 43.7 | 39.3 | 759f3316 |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/detectron2/mask_rcnn_coat_lite_small_FPN_1x_759f3316.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/detectron2/mask_rcnn_coat_lite_small_FPN_1x_759f3316.json) |
| CoaT-Lite Small | 3x | 44.5 | 39.8 | 0a5baf05 |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/detectron2/mask_rcnn_coat_lite_small_FPN_3x_0a5baf05.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/detectron2/mask_rcnn_coat_lite_small_FPN_3x_0a5baf05.json) |
| CoaT Mini | 1x | 44.0 | 39.5 | 9ddaa460 |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/detectron2/mask_rcnn_coat_mini_FPN_1x_9ddaa460.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/detectron2/mask_rcnn_coat_mini_FPN_1x_9ddaa460.json) |
| CoaT Mini | 3x | 45.2 | 40.2 | f34f1182 |[model](https://vcl.ucsd.edu/coat/pretrained/tasks/detectron2/mask_rcnn_coat_mini_FPN_3x_f34f1182.pth), [metrics](https://vcl.ucsd.edu/coat/pretrained/tasks/detectron2/mask_rcnn_coat_mini_FPN_3x_f34f1182.json) |

The following commands provide an example (CoaT-Lite Mini) to evaluate the pre-trained checkpoint.
   ```bash
   # Download the pretrained checkpoint.
   mkdir -p ./output/pretrained
   wget http://vcl.ucsd.edu/coat/pretrained/tasks/detectron2/mask_rcnn_coat_lite_mini_FPN_1x_5bb8caf8.pth -P ./output/pretrained
   sha256sum ./output/pretrained/mask_rcnn_coat_lite_mini_FPN_1x_5bb8caf8.pth  # Make sure it matches the SHA-256 hash (first 8 characters) in the table.

   # Evaluate.
   # Usage: Please see [Detectron2's document](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) for more details.
   PYTHONPATH=. python train_net.py \
      --config-file ./configs/COCO-InstanceSegmentation/mask_rcnn_coat_lite_mini_FPN_1x.yaml \
      --eval-only \
      --num-gpus 8 \
      MODEL.WEIGHTS ./output/pretrained/mask_rcnn_coat_lite_mini_FPN_1x_5bb8caf8.pth
   # It should output similar results to the below ones:
   #    Task: bbox
   #    AP,AP50,AP75,APs,APm,APl
   #    39.8625,61.9343,43.1341,24.8811,42.8612,50.9974
   #    Task: segm
   #    AP,AP50,AP75,APs,APm,APl
   #    36.3920,58.3952,38.7107,18.6367,38.7183,51.6407
   ```
   
### Train
The following commands provide an example (CoaT-Lite Mini, 8-GPU) to train the Mask R-CNN w/ CoaT backbone.
   ```bash
   # Usage: Please see [Detectron2's document](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) for more details.
   PYTHONPATH=. python train_net.py \
      --config-file ./configs/COCO-InstanceSegmentation/mask_rcnn_coat_lite_mini_FPN_1x.yaml \
      --num-gpus 8 \
      SOLVER.IMS_PER_BATCH 16 SOLVER.BASE_LR 0.02 OUTPUT_DIR ./output/mask_rcnn_coat_lite_mini_FPN_1x
   ```

### Evaluate
The following commands provide an example (CoaT-Lite Mini) to evaluate the checkpoint after training.
   ```bash
   # Usage: Please see [Detectron2's document](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) for more details.
   PYTHONPATH=. python train_net.py \
      --config-file ./configs/COCO-InstanceSegmentation/mask_rcnn_coat_lite_mini_FPN_1x.yaml \
      --eval-only \
      --num-gpus 8 \
      MODEL.WEIGHTS ./output/mask_rcnn_coat_lite_mini_FPN_1x/model_0089999.pth
   ```

## Acknowledgment
Thanks to [Detectron2](https://github.com/facebookresearch/detectron2) for the Mask R-CNN implementation.