import timm
from models.vit_lora import LoRA_ViT_timm

def vit_with_classifiers(num_classes=10):
    weightInfo={
        # "small":"WinKawaks/vit-small-patch16-224",
        "base":"vit_base_patch16_224",
        "base_dino":"vit_base_patch16_224.dino", # 21k -> 1k
        "base_sam":"vit_base_patch16_224.sam", # 1k
        "base_mill":"vit_base_patch16_224_miil.in21k_ft_in1k", # 1k
        "base_beit":"beitv2_base_patch16_224.in1k_ft_in22k_in1k",
        "base_clip":"vit_base_patch16_clip_224.laion2b_ft_in1k", # 1k
        "base_deit":"deit_base_distilled_patch16_224", # 1k
        "large":"google/vit-large-patch16-224",
        "large_clip":"vit_large_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
        "large_beit":"beitv2_large_patch16_224.in1k_ft_in22k_in1k", 
        "huge_clip":"vit_huge_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
        "giant_eva":"eva_giant_patch14_224.clip_ft_in1k", # laion-> 1k
        "giant_clip":"vit_giant_patch14_clip_224.laion2b",
        "giga_clip":"vit_gigantic_patch14_clip_224.laion2b"
        }

    rank = 4
    alpha = 8
    # num_classes = 2

    model = timm.create_model(weightInfo["base_dino"], pretrained=True, pretrained_cfg_overlay=dict(file='/home/liyan/LoRA-ViT/model/pytorch_model.bin'))
    model.reset_classifier(num_classes)
    # model_with_lora = LoRA_ViT_timm(model, r=rank, alpha=alpha, num_classes=num_classes)
    return model

def vit_lora(num_classes=10):
    weightInfo={
        # "small":"WinKawaks/vit-small-patch16-224",
        "base":"vit_base_patch16_224",
        "base_dino":"vit_base_patch16_224.dino", # 21k -> 1k
        "base_sam":"vit_base_patch16_224.sam", # 1k
        "base_mill":"vit_base_patch16_224_miil.in21k_ft_in1k", # 1k
        "base_beit":"beitv2_base_patch16_224.in1k_ft_in22k_in1k",
        "base_clip":"vit_base_patch16_clip_224.laion2b_ft_in1k", # 1k
        "base_deit":"deit_base_distilled_patch16_224", # 1k
        "large":"google/vit-large-patch16-224",
        "large_clip":"vit_large_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
        "large_beit":"beitv2_large_patch16_224.in1k_ft_in22k_in1k", 
        "huge_clip":"vit_huge_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
        "giant_eva":"eva_giant_patch14_224.clip_ft_in1k", # laion-> 1k
        "giant_clip":"vit_giant_patch14_clip_224.laion2b",
        "giga_clip":"vit_gigantic_patch14_clip_224.laion2b"
        }

    rank = 4
    alpha = 8
    # num_classes = 2

    model = timm.create_model(weightInfo["base_dino"], pretrained=True, pretrained_cfg_overlay=dict(file='/home/liyan/LoRA-ViT/model/pytorch_model.bin'))
    # model.reset_classifier(num_classes)
    model_with_lora = LoRA_ViT_timm(model, r=rank, alpha=alpha, num_classes=num_classes)
    return model_with_lora

def param_name_to_module_id_vit(name = 'depth'):
    if name.startswith('lora_vit.blocks'):
        depth = eval(list(name.split('.'))[2])
        return depth
    elif name.startswith('lora_vit.head'):
        return 12
    elif name == 'depth':
        return 13
    else:
        raise NotImplementedError

# my_model = vit_lora(num_classes=10)
# # print(my_model.state_dict().keys())
# for name,param in my_model.named_parameters():
#     if param.requires_grad:
#         print(name)