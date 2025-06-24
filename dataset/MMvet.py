import os
import json

import numpy as np
import pandas as pd

from dataset.base import BaseDataset
from utils.func import read_jsonl
import json, csv

class MMvetDataset(BaseDataset):
    def __init__(self, prompter, split="val", data_root="/workspace/ZJY/MM-Vet/v1_data/", pred=False, pred_json=None):
        super(MMvetDataset, self).__init__()
        self.ann_path = os.path.join(data_root, "mm-vet.json")
        self.img_root = os.path.join(data_root, "images/")
        self.prompter = prompter
        if pred:
            with open("/workspace/safety_heads/Attack/eval/mmvet/qwen/vl32-alpha-seed42.json", "r", encoding="utf-8") as f:
                self.pred = json.load(f)
        else:
            self.pred = None
    def get_data(self):

        pos_data = []
        with open(self.ann_path, 'r') as file:
            reader = json.load(file)
            for k,v in reader.items():
                img_path = v["imagename"]
                entry = {
                        "img_path": os.path.join(self.img_root, img_path),
                        "question": v["question"],
                        "answer": None,
                        "label": 0
                        }
                if self.pred is not None:
                    entry["pred"] = self.pred.get(k, None)
                    # print(k, entry["pred"])
                pos_data.append(entry)

        return pos_data,[]