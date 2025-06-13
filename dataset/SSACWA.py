import os
import json

import numpy as np
import pandas as pd

from dataset.base import BaseDataset
from utils.func import read_jsonl
import json

class SSACWADataset(BaseDataset):
    def __init__(self, prompter, split="val", data_root="/data1/data4multitrust/robustness/adv_nips/"):
        super(SSACWADataset, self).__init__()
        self.ann_path = os.path.join(data_root, "label.json")
        self.clean_img_root = os.path.join(data_root, "source", "clean")
        self.target_img_root = os.path.join(data_root, "source", "target")
        self.untarget_img_root = os.path.join(data_root, "source", "untarget")
        self.prompter = prompter

    def get_data(self):
        # with open(self.ann_path) as f:
        #     ann = json.load(f)['annotations']
        ann = read_jsonl(self.ann_path)
        pos_data = [
            {
                "img_path": os.path.join(self.clean_img_root, ins['image_name']),
                "question": self.prompter.build_prompt('Please provide a detailed description of the image.'),
                "answer": None,
                "label": 0
            }
            for ins in ann
        ]
        neg_data = [
            {
                "img_path": os.path.join(self.target_img_root, ins['image_name']),
                "question": self.prompter.build_prompt('Please provide a detailed description of the image.'),
                "answer": None,
                "label": 1
            }
            for ins in ann
        ]

        return pos_data, neg_data