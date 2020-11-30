import torch
from nets.sparse_rcnn import SparseRCNN
from datasets.coco import COCODataSets
from torch.utils.data.dataloader import DataLoader
from utils.optims_utils import split_optimizer_v2

if __name__ == '__main__':
    dataset = COCODataSets(img_root="/home/huffman/data/val2017",
                           annotation_path="/home/huffman/data/annotations/instances_val2017.json",
                           use_crowd=True,
                           augments=True,
                           remove_blank=True,
                           max_thresh=640
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=dataset.collect_fn)
    net = SparseRCNN(**{"backbone": "resnet50"})
    # optim_cfg = {
    #     "weight_decay": 0.0001,
    #     "optimizer": "AdamW",
    #     'lr': 0.000025
    # }
    # optimizer = split_optimizer_v2(net, optim_cfg)
    # print(optimizer)

    # print(net)
    # for n, v in net.named_parameters():
    #     print(n, v.shape)
    for img_input, targets, batch_len in dataloader:
        out = net(img_input, targets={"target": targets, "batch_len": batch_len})
        break
