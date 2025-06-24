import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import argparse
import time

import cv2
import json
import numpy as np
from tqdm import tqdm

from model import build_model
from dataset import build_dataset
from utils.func import split_list, get_chunk
from utils.prompt import Prompter

import random, pickle, torch
import torch.backends.cudnn as cudnn


def get_model_activtion(args, data, model, answer_file):

    all_results = []
    
    for ins in tqdm(data):
        img_path = ins['img_path']
        img_id = img_path.split("/")[-1]
        prompt = ins['question']
        answer = ins['answer']
        label = ins["label"]
 
        hidden_states, mlp_residual, attn_residual, attn_heads, vit_attn_heads = model.get_activations(img_path, prompt, answer)

        out = {
            "image": img_id,
            "img_path": img_path,
            "model_name": args.model_name,
            "question": prompt,
            "answer": answer,
            "label": label,
            "attn_residual": attn_residual[:, -1].cpu(),
            "hidden_states": hidden_states[:, -1].cpu(),
            "mlp_residual": mlp_residual[:, -1].cpu(),
            "attn_heads": attn_heads[:, -1].cpu(),
            "vit_attn_heads": vit_attn_heads.mean(1).cpu(),
            "hidden_states_mean": hidden_states.mean(1).cpu(),
            "scenario": ins.get('scenario', None)
        }
        all_results.append(out)
    
    with open(answer_file, 'wb') as file:
        pickle.dump(all_results, file)


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def main(args):

    setup_seeds()

    model = build_model(args)
    
    prompter = Prompter(args.prompt, args.theme)
    
    pos_data, neg_data = build_dataset(args.dataset, args.split, prompter)

    data = pos_data + neg_data
        
    if not os.path.exists(f"./output/{args.model_name}/"):
        os.makedirs(f"./output/{args.model_name}/")
    
    saved_file = f"./output/{args.model_name}/{args.model_status}_{args.dataset}_{args.split}_{args.prompt}_activations.pkl"
    get_model_activtion(args, data, model, saved_file)
    print(f'Saved activations to {saved_file}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run a model')
    parser.add_argument("--model_name", default="LLaVA-7B")
    parser.add_argument("--model_path", default="./llava-v1.5-7b")
    # parser.add_argument("--model_name", default="Qwen-VL-Chat")
    # parser.add_argument("--model_path", default="/workspace/data1/huggingface/hub/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--sampling", choices=['first', 'random', 'class'], default='first')
    parser.add_argument("--split", default="test") # train/test 
    parser.add_argument("--dataset", default="vlsafe") # mmsafety vlguard mmvet vlsafe
    parser.add_argument("--prompt", default='oe')
    parser.add_argument("--theme", default='mad')

    parser.add_argument("--model_status", default="raw")

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--token_id", type=int, default=0)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    
    main(parser.parse_args())