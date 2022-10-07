import os
from yacs.config import CfgNode as CN

def get_config(project_location='./data', categories=None):

    # define defaults
    config = CN()

    config.save = True
    config.save_location = './output'

    config.display = True
    config.display_it = 50

    config.pretrained_model = None

    config.train_dataset = CN(dict(
        name = 'ml4vision', 
        params = CN(dict(
            location = project_location,
            split = 'TRAIN',
            fake_size = 500,
        )),
        batch_size = 4,
        num_workers = 4
    ))

    config.val_dataset = CN(dict(
        name = 'ml4vision',
        params = CN(dict(
            location = project_location,
            split = 'VAL'
        )),
        batch_size = 1,
        num_workers = 4
    ))

    config.model = CN(dict(
        name = 'unet',
        params = CN(dict(
            encoder_name = 'resnet18',
            classes = 3 + (len(categories) if len(categories) > 1 else 0)
        ))
    ))

    config.loss = CN(dict(
        name = 'centernet',
        params = CN(dict(
        ))
    ))

    config.solver = CN(dict(
        lr = 5e-4,
        patience = 3, # number of epochs with no improvement after which learning rate will be reduced
        max_epochs = 50
    ))

    config.transform = CN(dict(
        resize = True,
        min_size = 512,
        random_crop = True,
        crop_size = 256,
        scale = 0,
        flip_horizontal = True,
        flip_vertical = True,
        random_brightness_contrast = True
    ))

    return config
