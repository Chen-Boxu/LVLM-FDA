import os
import json

import numpy as np
import pandas as pd

from dataset.base import BaseDataset
from utils.func import read_jsonl
import json

class ToxicDataset(BaseDataset):
    def __init__(self, prompter, split="val", data_root="/data/coco/"):
        super(ToxicDataset, self).__init__()
        self.ann_path_clean = os.path.join(data_root, "safe.jsonl")
        if split == 'train':
            self.ann_path_toxic = os.path.join(data_root, "toxic.jsonl")
        elif split == 'test':
            self.ann_path_toxic = os.path.join(data_root, "challenge100.jsonl")
        self.data_root = data_root
        self.prompter = prompter

    def get_data(self):
        ann = read_jsonl(self.ann_path_clean)
        pos_data = [
            {
                "img_path": os.path.join(self.data_root, ins['img_path']),
                "question": self.prompter.build_prompt(ins['prompt']),
                "answer": None,
                "label": 0
            }
            for ins in ann
        ]
        ann = read_jsonl(self.ann_path_toxic)
        neg_data = [
            {
                "img_path": os.path.join(self.data_root, ins['img_path']),
                "question": self.prompter.build_prompt(ins['prompt']),
                "answer": None,
                "label": 1
            }
            for ins in ann
        ]

        return pos_data, neg_data


class RealToxicDataset(BaseDataset):
    def __init__(self, prompter, split="train", data_root="/data/coco/"):
        super(RealToxicDataset, self).__init__()
        self.ann_path_clean = os.path.join(data_root, "safe_seed0.jsonl")
        if split == 'train':
            self.ann_path_toxic = os.path.join(data_root, "toxic_seed0.jsonl")
        elif split == 'test':
            self.ann_path_toxic = os.path.join(data_root, "challenge100.jsonl")
        self.data_root = data_root
        self.prompter = prompter

    def get_data(self):
        ann = read_jsonl(self.ann_path_clean)
        pos_data = [
            {
                "img_path": os.path.join(self.data_root, ins['img_path']),
                "question": self.prompter.build_prompt(ins['prompt']),
                "answer": None,
                "label": 0
            }
            for ins in ann
        ]
        ann = read_jsonl(self.ann_path_toxic)
        neg_data = [
            {
                "img_path": os.path.join(self.data_root, ins['img_path']),
                "question": self.prompter.build_prompt(ins['prompt']),
                "answer": None,
                "label": 1
            }
            for ins in ann
        ]

        return pos_data, neg_data