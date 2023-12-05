import random

from torchvision import transforms
from torchvision.transforms.functional import crop


def get_vae_train_transforms(args):
    train_resize = transforms.Resize(args.image_resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(args.image_resolution) if args.image_center_crop else transforms.RandomCrop(args.image_resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    return {
        "args": {
            "resolution": args.image_resolution,
            "center_crop": args.image_center_crop,
            "random_flip": args.image_random_flip,
        },
        "resize": train_resize,
        "crop": train_crop,
        "flip": train_flip,
        "norm": train_transforms,
    }

def preprocess_train_image(image, transform_dict):
    image = image.convert("RGB")
    # image aug
    original_size = (image.height, image.width)
    image = transform_dict["resize"](image)
    if transform_dict["args"]["center_crop"]:
        y1 = max(0, int(round((image.height - transform_dict["args"]["resolution"]) / 2.0)))
        x1 = max(0, int(round((image.width - transform_dict["args"]["resolution"]) / 2.0)))
        image = transform_dict["crop"](image)
    else:
        y1, x1, h, w = transform_dict["crop"].get_params(image, (transform_dict["args"]["resolution"], transform_dict["args"]["resolution"]))
        image = crop(image, y1, x1, h, w)
    if transform_dict["args"]["random_flip"] and random.random() < 0.5:
        # flip
        x1 = image.width - x1
        image = transform_dict["flip"](image)
    crop_top_left = (y1, x1)
    image = transform_dict["norm"](image)

    target_size = (transform_dict["args"]["resolution"], transform_dict["args"]["resolution"])

    return image, original_size, crop_top_left, target_size

def preprocess_train_image_fn(args):
    transform_dict = get_vae_train_transforms(args)
    def transform_fn(image):
        return preprocess_train_image(image, transform_dict)
    return transform_fn
