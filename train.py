from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog

import os
import torch

from detectron2.data.datasets import register_coco_instances


print("Torch version:",torch.__version__)

print("Is CUDA enabled?",torch.cuda.is_available())

if __name__ == '__main__':  
    
    print("Start of the training...")

    #for d in ["train", "test"]:
    register_coco_instances("custom_train", {}, "data/train/train.json", "data/train/")
    register_coco_instances("custom_test", {}, "data/test/test.json", "test/train/")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ("custom_test")
    cfg.DATALOADER.NUM_WORKERS = 1 #2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.CHECKPOINT_PERIOD = 10
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.DEVICE = "cpu" #cuda
    
    # Define your custom class names
    custom_classes = ['wall', 'plant', 'tv', 'carpet', 'sofa', 'window', 'curtain', 'picture', 'chair', 'lamp']

    # Create a dataset dictionary with your custom class names
    MetadataCatalog.get("custom_train").set(thing_classes=custom_classes)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()