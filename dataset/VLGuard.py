import os
import json

from dataset.base import BaseDataset


class VLGuardDataset(BaseDataset):
    def __init__(self, prompter, split="train", data_root="/data1/VLGuard/", pred=False, pred_json=None):
        super(VLGuardDataset, self).__init__()
        self.split = split
        self.ann_root = os.path.join(data_root, f"{split}.json")
        self.img_root = os.path.join(data_root, split)
        self.prompter = prompter
        if pred:
            with open("/workspace/safety_heads/Attack/eval/vlguard/qwen/alpha_pos_seed42.json", "r", encoding="utf-8") as f:
                self.pred = json.load(f)
        else:
            self.pred = None

    def get_data(self):
        ann = json.load(open(self.ann_root, 'r'))
        pos_data, neg_data = [], []
        for ins in ann:
            if ins['safe'] == True:
                pos = {
                        "img_path": os.path.join(self.img_root, ins["image"]),
                        "question": self.prompter.build_prompt(ins["instr-resp"][0]["safe_instruction"]),
                        "answer": None,
                        "label": 0,
                    }
                if self.pred is not None:
                    pos["pred"] = self.pred.get(pos['img_path'], None)
                pos_data.append(pos)

                neg = {
                        "img_path": os.path.join(self.img_root, ins["image"]),
                        "question": self.prompter.build_prompt(ins["instr-resp"][1]["unsafe_instruction"]),
                        "answer": None,
                        "label": 1,
                    }
                if self.pred is not None:
                    neg["pred"] = self.pred.get(neg['img_path'], None)
                neg_data.append(neg)

        return pos_data, neg_data