import os
import json
import numpy as np

from dataset.base import BaseDataset

class MMSafetyBench(BaseDataset):
    def __init__(self, prompter, split="train", data_root="/data/MM-SafetyBench/", pred=False):
        super(MMSafetyBench, self).__init__()
        self.ann_root = data_root
        self.img_root = data_root
        self.split = split
        self.prompter = prompter
        if pred:
            with open("/workspace/safety_heads/Attack/eval/mmsafety/qwen/alpha_250612.json", "r", encoding="utf-8") as f:
                self.pred = json.load(f)
        else:
            self.pred = None
         
    def get_data(self):

        pos_data, neg_data = [], []

        # The original MMSafetyBench data
        # All data are unsafe queries.
        val_phrases = []
        for i, sc in enumerate([
            '01-Illegal_Activitiy',
            '02-HateSpeech', 
            '03-Malware_Generation', 
            '04-Physical_Harm',
            '05-EconomicHarm',
            '06-Fraud',
            '07-Sex',
            '08-Political_Lobbying',
            '09-Privacy_Violence',
            '10-Legal_Opinion',
            '11-Financial_Advice',
            '12-Health_Consultation',
            '13-Gov_Decision'
        ]):
            ann = json.load(open(os.path.join(self.ann_root, f"data/processed_questions/{sc}.json"), 'r'))
            val_phrases += [ins['Key Phrase'] for ins in ann.values()]
            print(sc)
            for k, v in ann.items():
                img_path = os.path.join(self.img_root, "data/imgs/", sc, "SD_TYPO", f"{k}.jpg")
                # img_path = os.path.join(self.img_root, "data/imgs/", sc, "TYPO", f"{k}.jpg")
                # img_path = os.path.join(self.img_root, "data/imgs/", sc, "SD", f"{k}.jpg")
                entry = {
                    "img_path": img_path,
                    "question": self.prompter.build_prompt(v['Rephrased Question']), # Rephrased Question
                    "answer": None,
                    "label": 1,
                    "scenario": i + 1
                }
                if self.pred is not None:
                    entry["pred"] = self.pred.get(img_path, None)
                neg_data.append(entry)
        
        # Generated safe data for both training and validation
        scenario_list = [
            "01-Daily_Activitiy",
            "02-Economics",
            "03-Physical",
            "04-Legal",
            "05-Politics",
            "06-Finance",
            "07-Health",
            "08-Sex",
            "09-Government",
        ]
        for sc in scenario_list:
            ann = json.load(open(os.path.join(self.ann_root, "safe_data/processed_questions/", f"{sc}.json")))
            print(sc)
            for k, v in ann.items():
                if v['Key Phrase'] not in val_phrases:
                    img_path = os.path.join(self.img_root, "safe_data/imgs/", sc, "SD_TYPO", f"{k}.jpg") # v['Rephrased Question']
                    # img_path = os.path.join(self.img_root, "safe_data/imgs/", sc, "TYPO", f"{k}.jpg") # v['Question']
                    # img_path = os.path.join(self.img_root, "data/imgs/", sc, "SD", f"{k}.jpg") # v['Rephrased Question(SD)']
                    entry = {
                            "img_path": img_path,
                            "question": self.prompter.build_prompt(v['Rephrased Question']), #Rephrased Question
                            "answer" : None,
                            "label": 0,
                            "scenario": 0
                        }
                    if self.pred is not None:
                        entry["pred"] = self.pred.get(img_path, None)
                    pos_data.append(entry)

        targets = ['SD_TYPO', 'SD', 'TYPO']
        if any(target in self.split for target in targets):
            return pos_data, neg_data
        
        train_num = int(min(len(pos_data), len(neg_data)) * 0.15)
        pos_data_train, pos_data_test = split(pos_data, train_num)
        neg_data_train, neg_data_test = split(neg_data, train_num)

        if self.split == 'train':
            return pos_data_train, neg_data_train
        else:
            return pos_data_test, neg_data_test


def split(data, train_num):
    train_idx = np.linspace(0, len(data)-1, train_num).astype(int).tolist()
    val_idx = list(set(range(len(data))) - set(train_idx))
    return [data[i] for i in train_idx], [data[i] for i in val_idx]