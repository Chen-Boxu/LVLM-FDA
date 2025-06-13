import os
import json

from dataset.base import BaseDataset


class CC3MDataset(BaseDataset):
    def __init__(self, prompter, split=None, data_root="/data1/LLaVA-CC3M-Pretrain-595K/"):
        super(CC3MDataset, self).__init__()
        self.ann_root = os.path.join(data_root, "chat.json")
        self.img_root = os.path.join(data_root, "images")
        self.prompter = prompter

    def get_data(self):
        ann = json.load(open(self.ann_root, 'r'))[:1000]
        pos_data = []
        for ins in ann:
            pos_data.append(
                {
                    "img_path": os.path.join(self.img_root, ins["image"]),
                    "question": self.prompter.build_prompt(ins["conversations"][0]["value"].replace('\n<image>', '').replace('<image>\n', '')),
                    "answer": None,
                    "label": 0,
                }
            )

        return pos_data, []