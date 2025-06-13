import os
import json

import numpy as np
import pandas as pd

from dataset.base import BaseDataset
from utils.func import read_jsonl
import json, csv

class VLSafeDataset(BaseDataset):
    def __init__(self, prompter, split="", data_root="/data1/coco2017/train2017/"):
        super(VLSafeDataset, self).__init__()
        self.ann_path = "./data/VLSafe_harmlessness_examine.jsonl"
        self.img_root = data_root
        self.prompter = prompter

    def get_data(self):

        ann = read_jsonl(self.ann_path)
        neg_data = [
            {
                "img_path": os.path.join(self.img_root, ins['image_id']),
                "question": self.prompter.build_prompt(ins['query']),
                "answer": None,
                "label": 1
            }
            for ins in ann
        ]

        return [], neg_data