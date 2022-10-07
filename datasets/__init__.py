from ml4vision.ml.datasets import ObjectDetectionDataset
from .mapping import mapping

def get_dataset(name, dataset_kwargs={}):
    if name == "ml4vision":
        return ObjectDetectionDataset(**dataset_kwargs, mapping=mapping)
    else:
        raise RuntimeError(f'Dataset {name} is not available!') 