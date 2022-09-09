import os
from models import get_model
import torch

def upload_model(client, config):

    print('Tracing best ap model')
    
    model = get_model(config.model.name, config.model.params)
    state = torch.load(os.path.join(config.save_location, 'best_ap_model.pth'), map_location='cpu')
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.eval()

    traced_model = torch.jit.trace(model, torch.randn(1, 3, config.transform.min_size, config.transform.min_size)) 
    traced_model.save(os.path.join(config.save_location, 'best_ap_model.pt'))

    print('Creating remote model')

    remote_model = client.get_or_create_model(
        f"{config.project_name}-model",
        description='',
        project = config.project_uuid,
        categories=config.categories,
        annotation_type="BBOX",
        architecture="object_detection_fn"
    )

    print('adding version')

    remote_model.add_version(
        os.path.join(config.save_location, 'best_ap_model.pt'),
        params = {
            'min_size': config.transform.min_size if config.transform.resize else False,
            'pad': 32,
            'normalize': True,
            'threshold': float(state["metrics"]["working_point"]["confidence"])
        },
        metrics = {
            'map': round(state["metrics"]["map"], 3),
            'precision': round(state["metrics"]["working_point"]["precision"], 3),
            'recall': round(state["metrics"]["working_point"]["recall"], 3)
        }
    )

