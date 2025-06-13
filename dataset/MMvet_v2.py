import os
import json

import numpy as np
import pandas as pd

from dataset.base import BaseDataset
from utils.func import read_jsonl
import json, csv

class MMvetDataset_v2(BaseDataset):
    def __init__(self, prompter, split="val", data_root="/workspace/ZJY/MM-Vet/v2_data/"):
        super(MMvetDataset_v2, self).__init__()
        self.ann_path = os.path.join(data_root, "mm-vet-v2.json")
        self.img_root = os.path.join(data_root, "images/")
        self.prompter = prompter

    def get_data(self):

        pos_data = []
        with open(self.ann_path, 'r') as file:
            reader = json.load(file)
            for k,v in reader.items():
                print(v["question"].split('<IMG>')[-1],v["question"].split('<IMG>')[0])
                img_path = v["question"].split('<IMG>')[-1]
                pos_data.append(
                    {
                        "img_path": os.path.join(self.img_root, img_path),
                        "question": v["question"].split('<IMG>')[0],
                        "answer": None,
                        "label": 0
                    }
                )

        return pos_data,[]