from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import random
from datetime import datetime

def get_dicts(img_dir):
    """This function loads the JSON file created with the annotator and converts it to 
    the detectron2 metadata specifications.
    """
    # load the JSON file
    json_file = os.path.join(img_dir, "train.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    # loop through the entries in the JSON file
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        # add file_name, image_id, height and width information to the records
        print(img_dir)
        print(v["filename"])
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]

        objs = []
        # one image can have multiple annotations, therefore this loop is needed
        for annotation in annos:
            # reformat the polygon information to fit the specifications
            anno = annotation["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
  
            region_attributes = annotation["region_attributes"]["class"]
            # specify the category_id to match with the class. 

            if "wall" in region_attributes:
                category_id = 1
            elif "plant" in region_attributes:
                category_id = 0

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
                "iscrowd": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts
    
if __name__ == "__main__":
    # the data has to be registered within detectron2, once for the train and once for
    # the val data
    for d in ["train", "val"]:
        DatasetCatalog.register(
            "room_" + d,
            lambda d=d: get_dicts("./data/room/" + d),
        )
        custom_classes = ['wall', 'plant', 'lamp']
        MetadataCatalog.get("room_" + d).set(thing_classes=custom_classes)
        
    building_metadata = MetadataCatalog.get("room_train")

    dataset_dicts = get_dicts("./data/room/train")
    
    
    #DatasetCatalog.register("room_train", lambda: dataset_dicts);
    #custom_classes = ['wall', 'plant', 'tv', 'carpet', 'sofa', 'window', 'curtain', 'picture', 'chair', 'lamp']
    #MetadataCatalog.get("room_train").set(thing_classes=custom_classes)

    for i, d in enumerate(random.sample(dataset_dicts, 3)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=building_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)

        plt.imshow(vis.get_image()[:, :, ::-1])
        # the folder inputs has to be created first
        plt.savefig(f"./data/generated/input_{i}.jpg")
        
    
    print("Start of the training...")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("room_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.SOLVER.CHECKPOINT_PERIOD = 30
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.DEVICE = "cpu" #cuda

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    