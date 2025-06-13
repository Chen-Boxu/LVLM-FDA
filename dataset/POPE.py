import os
import json

import numpy as np

from dataset.base import BaseDataset
from utils.func import read_jsonl

class POPEDataset(BaseDataset):
    def __init__(self, split="val", data_root="/data/coco/"):
        super(POPEDataset, self).__init__()
        self.ann_path = f"./data/pope/coco_{split}"
        self.img_root = os.path.join(data_root, f"{split}2014")
        self.split = split
         
    def get_data(self):
        pos_data, neg_data = [], []
        cats = ["adversarial", "popular", "random"]
        for category in cats:
            ann = read_jsonl(os.path.join(self.ann_path, f"coco_{self.split}_pope_{category}.json"))
            for ins in ann:
                # pos_data.append({
                #     "img_path": os.path.join(self.img_root, ins['image']),
                #     "question": f"{ins['text']}\nAnswer the question using a single word or phrase.",
                #     "answer": "Yes" if ins['label'] == 'yes' else "No",
                #     "gt_label": 0 if ins['label'] == 'yes' else 1,
                #     "label": 0,
                #     "question_id": ins["question_id"],
                #     "category": category
                # })
                # neg_data.append({
                #     "img_path": os.path.join(self.img_root, ins['image']),
                #     "question": f"{ins['text']}\nAnswer the question using a single word or phrase.",
                #     "answer": "No" if ins['label'] == 'yes' else "Yes",
                #     "gt_label": 0 if ins['label'] == 'yes' else 1,
                #     "label": 1,
                #     "question_id": ins["question_id"],
                #     "category": category
                # })
                if ins['label'] == 'yes':
                    pos_data.append({
                        "img_path": os.path.join(self.img_root, ins['image']),
                        "question": ins['text'],
                        "answer": None,
                        "label": 0,
                        "question_id": ins["question_id"],
                        "category": category
                    })
                else:
                    neg_data.append({
                        "img_path": os.path.join(self.img_root, ins['image']),
                        "question": ins['text'],
                        "answer": None,
                        "label": 1,
                        "question_id": ins["question_id"],
                        "category": category
                    })
            idx = np.random.choice(range(len(pos_data)), 1000, replace=False)
            pos_data = [pos_data[i] for i in idx]
            neg_data = [neg_data[i] for i in idx]

        return pos_data, neg_data