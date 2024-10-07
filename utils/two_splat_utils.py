import torch

def combine_splatter_images(front, back):
    combined = {}

    for key in front.keys():
        combined[key] = torch.cat([front[key], back[key]], dim=1)
    
    return combined