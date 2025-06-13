import os
import json

import numpy as np
import pandas as pd

from dataset.base import BaseDataset
from utils.func import read_jsonl
import json
import glob


class ImgJPDataset(BaseDataset):
    def __init__(self, prompter, split="val", data_root="/data1/ImgJP/"):
        super(ImgJPDataset, self).__init__()
        # self.clean_img_root = os.path.join(data_root, "clean_50")
        # self.target_img_root = os.path.join(data_root, "target_eps64_50")
        # self.text_prompt_path = os.path.join(data_root, "manual_harmful_instructions.csv")
        # with open(self.text_prompt_path, 'r') as f:
        #     text = f.readlines()
        # self.text_prompt = [t.split(',')[0] for t in text[1:]]

        self.data_root = data_root
        self.prompter = prompter

    def get_data(self):
        pos_data, neg_data = [], []

        for class_name in os.listdir(self.data_root):
            data_root = os.path.join(self.data_root, class_name)
            clean_img_root = os.path.join(data_root, 'clean')
            adv_img_root = os.path.join(data_root, 'adv')
            with open(glob.glob(data_root + '/*.csv')[0], 'r') as f:
                text = f.readlines()
            text_prompt = [t.split(',')[0] for t in text[1:]]

            for path in glob.glob(clean_img_root + '/*'):
                for text in text_prompt:
                    pos_data.append(
                        {
                            "img_path": path,
                            "question": self.prompter.build_prompt(text),
                            "answer": None,
                            "label": 0
                        }
                    )
            for path in glob.glob(adv_img_root + '/*'):
                for text in text_prompt:
                    neg_data.append(
                        {
                            "img_path": path,
                            "question": self.prompter.build_prompt(text),
                            "answer": None,
                            "label": 1
                        }
                    )

        return pos_data, neg_data