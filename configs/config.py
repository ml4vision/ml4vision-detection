from yacs.config import CfgNode as CN

def get_config(client, project_name, project_owner=None):

    # load project
    project = client.get_project_by_name(project_name, owner=project_owner)
    
    # make split
    split = project.make_split()

    assert split['n_train'] > 0, f'Number of training samples should be bigger than zero'
    assert split['n_val'] > 0, f'Number of validation samples should be bigger than zero'

    # define defaults
    config = CN()

    config.project_uuid = project.uuid
    config.project_name = project.name
    config.project_owner = project.owner['username']
    config.categories = project.categories
    config.n_categories = len(project.categories)

    config.save = True
    config.save_location = './output'

    config.display = True
    config.display_it = 50

    config.pretrained_model = None

    config.train_dataset = CN(dict(
        name = 'remote', 
        params = CN(dict(
            api_key = client.apikey,
            name = project.name,
            owner = project_owner,
            labeled_only = True,
            approved_only = False,
            split = 'TRAIN',
            cache_location = './dataset',
            min_size = 500,
        )),
        batch_size = 4,
        num_workers = 4
    ))

    config.val_dataset = CN(dict(
        name = 'remote',
        params = CN(dict(
            api_key = client.apikey,
            name = project.name,
            owner = project_owner,
            labeled_only = True,
            approved_only = False,
            split = 'VAL',
            cache_location = './dataset',
        )),
        batch_size = 1,
        num_workers = 4
    ))

    config.model = CN(dict(
        name = 'unet',
        params = CN(dict(
            encoder_name = 'resnet18',
            classes = 3 + (len(project.categories) if len(project.categories) > 1 else 0)
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
