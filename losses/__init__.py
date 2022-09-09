from .losses import centernet_loss

def get_loss(name, loss_kwargs={}):
    if name == 'centernet':
        return centernet_loss
    else:
        raise RuntimeError(f'Loss {name} is not available!')