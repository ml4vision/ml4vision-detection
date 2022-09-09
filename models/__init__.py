import torch
from segmentation_models_pytorch import create_model


def get_model(name, model_kwargs={}):
    model = create_model(name, **model_kwargs)

    # init output to match output statistics
    with torch.no_grad():
        model.segmentation_head[0].bias[0].fill_(-2.19)
        model.segmentation_head[0].bias[1].fill_(15)
        model.segmentation_head[0].bias[2].fill_(15)
    
    return model