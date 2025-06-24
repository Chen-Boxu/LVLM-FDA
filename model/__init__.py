def build_model(args):
    if args.model_name == "LLaVA-7B":
        from .LLaVA import LLaVA
        model = LLaVA(args)
    elif args.model_name == "Qwen-VL-Chat":
        from .Qwen_VL_Chat import Qwen_VL_Chat
        model = Qwen_VL_Chat(args)
    else:
        model = None
        
    return model
