import os
import json

import numpy as np
import pandas as pd

from dataset.base import BaseDataset
from utils.func import read_jsonl
import json

class LUREDataset(BaseDataset):
    def __init__(self, split="val", data_root="/data/coco/"):
        super(LUREDataset, self).__init__()
        self.ann_path = "./data/LURE/hallucination5k_train.jsonl"
        self.img_root = os.path.join(data_root, "train2014")

    def get_data(self):
        # with open(self.ann_path) as f:
        #     ann = json.load(f)['annotations']
        ann = read_jsonl(self.ann_path)
        pos_data = [
            {
                "img_path": os.path.join(self.img_root, ins['image']),
                "question": 'Please describe this image in detail.',
                "answer": ins['value'],
                "label": 0
            }
            for ins in ann
        ]
        neg_data = [
            {
                "img_path": os.path.join(self.img_root, ins['image']),
                "question": 'Please describe this image in detail.',
                "answer": ins['h_value'],
                "label": 1
            }
            for ins in ann
        ]

        return pos_data, neg_data