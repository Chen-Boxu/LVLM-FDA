dataset_roots = {
    "mmsafety": "/data1/MM-SafetyBench/",
    "vlguard": "/data1/VLGuard/",
    "vlsafe": "/workspace/safety_heads/Attack/data/coco2017/train2017/",
    "mmvet": "/workspace/ZJY/MM-Vet/v1_data/",
}


def build_dataset(dataset_name, split, prompter, pred=False):
    if dataset_name == "mmsafety":
        from .MMSafety import MMSafetyBench
        dataset = MMSafetyBench(prompter, split, dataset_roots[dataset_name], pred)
    elif dataset_name == "vlguard":
        from .VLGuard import VLGuardDataset
        dataset = VLGuardDataset(prompter, split, dataset_roots[dataset_name], pred)
    elif dataset_name == "vlsafe":
        from .VLSafe import VLSafeDataset
        dataset = VLSafeDataset(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "mmvet":
        from .MMvet import MMvetDataset
        dataset = MMvetDataset(prompter, split, dataset_roots[dataset_name], pred)
    else:
        from .base import BaseDataset
        dataset = BaseDataset()
        
    return dataset.get_data()
