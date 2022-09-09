from PIL import Image, ExifTags
import torch

def handle_exif_rotation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())

        if exif[orientation] == 3:
            image=image.transpose(Image.ROTATE_180)
        elif exif[orientation] == 6:
            image=image.transpose(Image.ROTATE_270)
        elif exif[orientation] == 8:
            image=image.transpose(Image.ROTATE_90)
        return image
    except (AttributeError, KeyError, IndexError):
        return image

def load_image(path, rgb=True):
    image = Image.open(path)
    image = handle_exif_rotation(image)

    if rgb and image.mode != 'rgb':
        image = image.convert('RGB')

    return image

def denormalize_img_tensor(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.as_tensor(mean, dtype=img.dtype, device=img.device)
    std = torch.as_tensor(std, dtype=img.dtype, device=img.device)

    return img.mul(std[:,None,None]).add(mean[:,None,None])