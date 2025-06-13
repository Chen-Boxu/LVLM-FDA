#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
                # print(vars(model))
                #0.1%
                # layer_idx = 

                # head_idx = 
    #             #10%
    #             layer_idx = [ 0, 19,  5,  5, 24, 13, 16, 10, 17, 25, 28, 15,  0, 15,  8, 14, 24,
    #     6, 10,  5,  9, 29, 28, 11,  8, 22,  9,  8,  7,  4, 11,  2, 18, 31,
    #    22,  2,  6, 22,  9, 26, 13, 25, 14, 21,  9, 14, 31, 13, 17, 27,  4,
    #     9, 22, 31, 17, 22, 20,  5,  9, 18, 18,  6, 21, 16, 10, 13, 27, 14,
    #     8,  8, 15, 14,  6, 10,  6,  3, 28, 27, 18, 31, 17, 27, 23, 13,  8,
    #     1, 14, 17, 26, 20, 27, 30,  9, 13,  3, 24,  7,  2, 17, 16,  7,  6,
    #    25, 10, 16,  5, 21, 18,  5, 19,  5,  6, 23,  9,  5, 14, 12, 18, 25,
    #    13, 28,  4,  3, 25, 12, 28, 13,  8]

    #             head_idx = [ 0, 30,  9,  4, 27, 19,  7,  9,  7,  5, 29, 29,  8, 11, 30, 31, 21,
    #    31, 13,  1, 13, 25, 10, 18, 21, 10, 15, 11, 18, 28,  5,  8,  8, 22,
    #    31, 25,  0,  2,  2,  6,  1, 24, 25,  2,  9, 16, 30, 24,  8, 15, 23,
    #    30, 11,  6, 12,  6, 13, 10, 18, 20, 29, 13,  8,  0, 10, 18, 31, 30,
    #     2,  4, 31,  9, 22,  7, 30, 16, 20, 18,  1,  7, 20,  9, 25, 11, 10,
    #    25,  2, 27, 30,  5,  2,  0, 22, 15, 13,  5, 16, 22, 24, 23,  3, 29,
    #    31,  5, 29, 27,  0,  5,  2, 11,  6, 28,  6, 25, 30, 29, 28, 27, 22,
    #    23,  2, 27,  5, 23, 31, 22,  5,  5]
                #1%
        #         layer_idx = [23, 28,  9,  9,  4, 24, 10, 11, 18, 15, 25, 31, 29, 13,  5, 11, 27,
        # 17,  1,  2, 22, 18, 14, 15,  5, 14,  0, 16, 18,  9,  7,  6]
        #         head_idx = [ 4, 29, 20, 29, 30, 27, 13, 21, 29, 15, 24, 22, 18, 17,  4, 23, 11,
        #  6,  0,  8,  2, 27, 25, 21, 13, 15,  0, 25, 26, 13,  1,  6]
                # layer_idx = [12, 9, 11, 4, 11, 16, 11, 16, 15, 5, 16, 9, 7, 15, 13, 10, 20, 11, 19, 17, 12, 22, 12, 19, 16, 12, 8, 31, 10, 14, 31, 20, 26, 26, 11, 15, 6, 19, 23, 11, 12, 19, 10, 9, 19, 14, 22, 12, 27, 11, 15, 12, 18, 10, 10, 18, 15, 22, 14, 12, 8, 12, 17, 27, 30, 14, 9, 23, 14, 24, 18, 22, 13, 16, 21, 22, 20, 28, 22, 12, 27, 22, 21, 20, 30, 24, 21, 10, 18, 13, 9, 31, 18, 21, 7, 15, 18, 24, 26, 6]

                # head_idx = [22, 9, 13, 21, 3, 0, 5, 13, 20, 30, 18, 24, 22, 30, 4, 16, 13, 24, 1, 5, 1, 18, 9, 22, 5, 8, 12, 4, 26, 23, 27, 9, 4, 6, 16, 4, 2, 0, 10, 11, 20, 21, 7, 18, 10, 3, 26, 5, 12, 31, 23, 12, 4, 29, 2, 26, 18, 11, 27, 30, 13, 0, 8, 9, 31, 21, 2, 18, 31, 6, 0, 21, 30, 14, 22, 4, 28, 7, 17, 17, 7, 1, 17, 22, 9, 23, 25, 1, 23, 22, 0, 31, 30, 14, 11, 31, 3, 7, 22, 31]

                # low_head_idx = [26, 21, 2, 25, 2, 1, 2, 1, 25, 17, 1, 21, 10, 25, 27, 23, 26, 2, 11, 25, 26, 9, 26, 11, 1, 26, 26, 8, 23, 12, 8, 26, 11, 11, 2, 25, 16, 11, 8, 2, 26, 11, 23, 21, 11, 12, 9, 26, 27, 2, 25, 26, 11, 23, 23, 11, 25, 9, 12, 26, 26, 26, 25, 27, 18, 12, 21, 8, 12, 31, 11, 9, 27, 1, 1, 9, 26, 26, 9, 26, 27, 9, 1, 26, 18, 31, 1, 23, 11, 27, 21, 8, 11, 1, 10, 25, 11, 31, 11, 16]
        #         #0.1%
        #         layer_idx = [28, 18,  9, 22, 11, 25, 23, 24, 27, 31, 21, 31, 23, 10, 15,  4,  8,
        #  2, 18, 10, 16, 11, 27, 30, 18, 15,  9,  9, 12, 24, 17, 24]
        #         head_idx = [29, 27, 29,  2, 23, 24, 27, 27, 31, 22,  8,  4,  4, 25, 15, 30, 13,
        #  8, 26, 13, 25,  1, 11, 31, 29, 17, 13, 20, 23, 22, 12, 20]
        #         s = 1e-10
        #         n_groups=32
        #         for i in range(n_groups):
        #             layer = model._modules['model'].layers[layer_idx[i]]  
        #             attention = layer.self_attn
        #             print(i,"before\n",attention.q_proj.weight.data[head_idx[i]])
        #             # attention.q_proj.weight.data[head_idx[i]] = attention.q_proj.weight.data[low_head_idx[i]]
        #             # attention.k_proj.weight.data[head_idx[i]] = attention.k_proj.weight.data[low_head_idx[i]]
        #             # attention.v_proj.weight.data[head_idx[i]] = attention.v_proj.weight.data[low_head_idx[i]]
        #             # attention.o_proj.weight.data[head_idx[i]] = attention.o_proj.weight.data[low_head_idx[i]]
        #             attention.q_proj.weight.data[head_idx[i]] *= s
        #             attention.k_proj.weight.data[head_idx[i]] *= s
        #             attention.v_proj.weight.data[head_idx[i]] *= s
        #             attention.o_proj.weight.data[head_idx[i]] *= s
        #             # attention.q_proj.weight.requires_grad = True
        #             # attention.k_proj.weight.requires_grad = True
        #             # attention.v_proj.weight.requires_grad = True
        #             # attention.o_proj.weight.requires_grad = True

        #             print(f"Layer {layer_idx[i]}, Head {head_idx[i]} weights replaced with 1e-10.")
        #             # print(f"Layer {layer_idx[i]}, Head {head_idx[i]} weights replaced with Head {low_head_idx[i]}.")
        #             print(i,"after\n",attention.q_proj.weight.data[head_idx[i]])
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
