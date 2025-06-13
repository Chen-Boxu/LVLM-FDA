import os
import json

import numpy as np

from dataset.base import BaseDataset
from utils.func import read_jsonl

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import random, torch
import torch.backends.cudnn as cudnn


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class CHAIRDataset(BaseDataset):
    def __init__(self, split="val", data_root="/data/coco/"):
        super(CHAIRDataset, self).__init__()
        self.ann_path = os.path.join(data_root, f"annotations/instances_{split}2014.json")
        self.caption_path = os.path.join(data_root, f"annotations/captions_{split}2014.json")
        self.img_root = os.path.join(data_root, f"{split}2014")
        self.split = split
        self.num_samples = 500
        self.image_id = [d['image_id'] for d in read_jsonl("/workspace/Attack/output/LLaVA-7B/edited_chair_all_oe_chat.jsonl")]
    

    def get_data(self):

        # setup_seeds(seed=114514)

        coco = COCO(self.caption_path)
        # img_ids = coco.getImgIds()

        # sampled_img_ids = random.sample(img_ids, self.num_samples)
        # # sampled_img_ids = img_ids[:self.num_samples]
        # print("sampled_img_ids", len(sampled_img_ids))

        pos_data = []
        for cur_img_id in self.image_id:
            cur_img = coco.loadImgs(cur_img_id)[0]
            cur_img_path = cur_img["file_name"]
            pos_data.append(
                {
                    "image_id": cur_img_id,
                    "img_path": os.path.join(self.img_root, cur_img_path),
                    "question": 'Please describe this image in detail.',
                    "label": 0
                }
            )

        return pos_data, None