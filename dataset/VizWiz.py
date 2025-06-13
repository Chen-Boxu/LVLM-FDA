import os
import json

from dataset.base import BaseDataset

class VizWizDataset(BaseDataset):
    def __init__(self, prompter, split="val", data_root="/data/VizWiz/"):
        super(VizWizDataset, self).__init__()
        self.ann = json.load(open(os.path.join(data_root, 'annotations', f"{split}.json"), 'r'))
        self.img_root = os.path.join(data_root, f"{split}/") 
        self.prompter = prompter
        self.split = split
         
    def get_data(self):
        pos_data, neg_data = [], []
        for ins in self.ann:
            if ins['answerable'] == 1:
                pos_data.append(
                    {
                        "img_path": os.path.join(self.img_root, ins['image']),
                        "question": self.prompter.build_prompt(ins['question']),
                        "label": 0
                    }
                )
            else:
                neg_data.append(
                    {
                        "img_path": os.path.join(self.img_root, ins['image']),
                        "question": self.prompter.build_prompt(ins['question']),
                        "label": 1
                    }
                )

        return pos_data, neg_data