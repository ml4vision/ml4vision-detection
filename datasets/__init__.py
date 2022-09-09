from ml4vision.ml import ObjectDetectionDataset as RemoteObjectDetectionDataset
from .mapping import mapping

def get_dataset(name, dataset_kwargs={}):
    if name == "remote":
        return RemoteObjectDetectionDataset(**dataset_kwargs, mapping=mapping)
    else:
        raise RuntimeError(f'Dataset {name} is not available!') 