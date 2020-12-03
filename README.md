# SparseRCNN
This is an unofficial pytorch implementation of SparseRCNN object detection as described in [Sparse R-CNN: End-to-End Object Detection with Learnable Proposals](https://arxiv.org/abs/2011.12450) by Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, Ping Luo

## requirement
```text
tqdm
pyyaml
numpy
opencv-python
pycocotools
torch >= 1.5
torchvision >=0.6.0
```
## result
we trained this repo on 4 GPUs with batch size 16(4 image per node).the total epoch is 36(3x),AdamW with cosine lr decay is used for optimizing.
finally, this repo achieves 38.9 mAp at 640px(max side) resolution with resnet50 backbone.(about 30.95fps)

**attention** : there is a large mismatch between the official mAP(pycocotools) calculation and the mAP calculation in this repo. you don't need to pay too much attention to mAP in training log
```shell script
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.592
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.214
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.412
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.327
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.528
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.348
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.614
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.760

```

## difference from original implement
the main difference is about the input resolution.the original implement use *min_thresh* and *max_thresh* to keep the short
side of the input image larger than *min_thresh* while keep the long side smaller than *max_thresh*.for simplicity we fix the long
side a certain size, then we resize the input image while **keep the width/height ratio**, next we pad the short side.the final
width and height of the input are same.


## training
for now we only support coco detection data.

### COCO
* modify main.py (modify config file path)
```python
from solver.ddp_mix_solver import DDPMixSolver
if __name__ == '__main__':
    processor = DDPMixSolver(cfg_path="your own config path") 
    processor.run()
```
* custom some parameters in *config.yaml*
```yaml
model_name: sparse_rcnn
data:
  train_annotation_path: data/annotations/instances_train2017.json
  #  train_annotation_path: data/annotations/instances_val2017.json
  val_annotation_path: data/annotations/instances_val2017.json
  train_img_root: data/train2017
  #  train_img_root: data/val2017
  val_img_root: data/val2017
  max_thresh: 640
  use_crowd: False
  batch_size: 4
  num_workers: 4
  debug: False
  remove_blank: Ture

model:
  num_cls: 80
  backbone: resnet50
  pretrained: True
  alpha: 0.25
  gamma: 2.0
  iou_type: giou
  iou_weights: 2.0
  iou_cost: 1.0
  cls_weights: 2.0
  cls_cost: 1.0
  l1_weights: 5.0
  l1_cost: 1.0
  num_proposals: 128

optim:
  optimizer: AdamW
  lr: 0.000025
  milestones: [27,33]
  warm_up_iter: 1000
  weight_decay: 0.0001
  epochs: 36
  sync_bn: False
  amp: False
  alpha: 0.1
  warm_up_factor: 0.01
val:
  interval: 1
  weight_path: weights


gpus: 0,1,2,3
```
* run train scripts
```shell script
nohup python -m torch.distributed.launch --nproc_per_node=4 main.py >>train.log 2>&1 &
```

## TODO
- [x] Color Jitter
- [x] Perspective Transform
- [x] Mosaic Augment
- [x] MixUp Augment
- [x] IOU GIOU DIOU CIOU
- [x] Warming UP
- [x] Cosine Lr Decay
- [x] EMA(Exponential Moving Average)
- [x] Mixed Precision Training (supported by apex)
- [x] FrozenBatchNorm2d
- [ ] PANet(neck)
- [ ] BiFPN(EfficientDet neck)
- [ ] VOC data train\test scripts
- [ ] custom data train\test scripts
- [ ] MobileNet Backbone support
## Reference
[original official implement](https://github.com/PeizeSun/SparseR-CNN) based on detectron2 and DETR
```text
@article{peize2020sparse,
  title   =  {{SparseR-CNN}: End-to-End Object Detection with Learnable Proposals},
  author  =  {Peize Sun and Rufeng Zhang and Yi Jiang and Tao Kong and Chenfeng Xu and Wei Zhan and Masayoshi Tomizuka and Lei Li and Zehuan Yuan and Changhu Wang and Ping Luo},
  journal =  {arXiv preprint arXiv:2011.12450},
  year    =  {2020}
}
```
