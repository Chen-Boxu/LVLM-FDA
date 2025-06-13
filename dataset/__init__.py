dataset_roots = {
    "vizwiz": "/data1/VizWiz-VQA/",
    "mmsafety": "/data1/MM-SafetyBench/",
    "mad": "/data1/coco2017/",   # MADBench uses COCO images
    "MathVista": "/data/MathVista/",
    "pope": "/data1/coco2014/",
    "chair": "/data1/coco2014/",
    "lure": "/data1/coco2014/",
    "ImageNet": "/data/ImageNet/",
    "ssacwa": "/data1/data4multitrust/robustness/adv_nips/",
    "diffpure": "/data1/diffpure/llava/",
    "vlm_toxic": "/data1/vlm_toxic/",
    "vlm_realtoxic": "/data1/vlm_realtoxic/",
    "fairness": "/data1/fairness_stereotype/",
    "figstep": "/data1/FigStep/",
    "imgjp": "/data1/ImgJP/",
    "vlguard": "/data1/VLGuard/",
    "vlsafe": "/workspace/safety_heads/Attack/data/coco2017/train2017/",
    "cc3m": "/data1/LLaVA-CC3M-Pretrain-595K/",
    "mmvet": "/workspace/ZJY/MM-Vet/v1_data/",
    "mmvet_v2": "/workspace/ZJY/MM-Vet/v2_data/",
}


def build_dataset(dataset_name, split, prompter, pred=False):
    if dataset_name == "vizwiz":
        from .VizWiz import VizWizDataset
        dataset = VizWizDataset(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "mad":
        from .MADBench import MADBench
        dataset = MADBench(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "ImageNet":
        from .ImageNet import ImageNetDataset
        dataset = ImageNetDataset(split, dataset_roots[dataset_name])
    elif dataset_name == "MathVista":
        from .MathV import MathVista
        dataset = MathVista(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "mmsafety":
        from .MMSafety import MMSafetyBench
        dataset = MMSafetyBench(prompter, split, dataset_roots[dataset_name], pred)
    elif dataset_name == "pope":
        from .POPE import POPEDataset
        dataset = POPEDataset(split, dataset_roots[dataset_name])
    elif dataset_name == "chair":
        from .CHAIR import CHAIRDataset
        dataset = CHAIRDataset(split, dataset_roots[dataset_name])
    elif dataset_name == "lure":
        from .LURE import LUREDataset
        dataset = LUREDataset(split, dataset_roots[dataset_name])
    elif dataset_name == "ssacwa":
        from .SSACWA import SSACWADataset
        dataset = SSACWADataset(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "diffpure":
        from .DiffPure import DiffPureDataset
        dataset = DiffPureDataset(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "vlm_toxic":
        from .vlm_toxic import ToxicDataset
        dataset = ToxicDataset(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "vlm_realtoxic":
        from .vlm_toxic import RealToxicDataset
        dataset = RealToxicDataset(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "fairness":
        from .fairness import FairnessDataset
        dataset = FairnessDataset(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "figstep":
        from .FigStep import FigStepDataset
        dataset = FigStepDataset(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "imgjp":
        from .ImgJP import ImgJPDataset
        dataset = ImgJPDataset(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "vlguard":
        from .VLGuard import VLGuardDataset
        dataset = VLGuardDataset(prompter, split, dataset_roots[dataset_name], pred)
    elif dataset_name == "vlsafe":
        from .VLSafe import VLSafeDataset
        dataset = VLSafeDataset(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "cc3m":
        from .CC3M import CC3MDataset
        dataset = CC3MDataset(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "mmvet":
        from .MMvet import MMvetDataset
        dataset = MMvetDataset(prompter, split, dataset_roots[dataset_name], pred)
    elif dataset_name == "mmvet_v2":
        from .MMvet_v2 import MMvetDataset_v2
        dataset = MMvetDataset_v2(prompter, split, dataset_roots[dataset_name])
    else:
        from .base import BaseDataset
        dataset = BaseDataset()
        
    return dataset.get_data()
