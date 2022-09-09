import torch

def box_collate_fn(batch):
    # extract boxes & category_ids from batch
    boxes_batch = [torch.as_tensor(item.pop('boxes')) for item in batch]
    # process batch using normal collate_fn
    others_batch = torch.utils.data._utils.collate.default_collate(batch)
    # append boxes to output
    others_batch['boxes'] = boxes_batch
    
    return others_batch