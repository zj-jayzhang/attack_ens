import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


import matplotlib.pyplot as plt


from typing import Sequence, Tuple

import numpy as np
import torch
import random
from torch import nn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# to get light adversarial training going, off by default
def fgsm_attack(model, xs, ys, epsilon, random_reps=1, batch_size=64):

    model = model.eval()

    its = int(np.ceil(xs.shape[0]/batch_size))

    all_perturbed_images = []

    for it in range(its):
        i1 = it*batch_size
        i2 = min([(it+1)*batch_size,xs.shape[0]])

        x = torch.Tensor(xs[i1:i2].transpose([0,3,1,2])).to("cuda")
        y = torch.Tensor(ys[i1:i2]).to("cuda").to(torch.long)

        x.requires_grad = True

        for _ in range(random_reps):
            outputs = model(x)
            loss = nn.CrossEntropyLoss()(outputs, y)
            loss.backward()

    perturbed_image = x + epsilon * x.grad.data.sign()
    perturbed_image = torch.clip(perturbed_image, 0, 1)

    all_perturbed_images.append(perturbed_image.detach().cpu().numpy().transpose([0,2,3,1]))

    return np.concatenate(all_perturbed_images,axis=0)



def cifar100_class_to_description(class_num):
    classes = [
        "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
        "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
        "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
        "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
        "house", "kangaroo", "keyboard", "lamp", "lawn mower", "leopard", "lion",
        "lizard", "lobster", "man", "maple tree", "motorcycle", "mountain", "mouse",
        "mushroom", "oak tree", "orange", "orchid", "otter", "palm tree", "pear",
        "pickup truck", "pine tree", "plain", "plate", "poppy", "porcupine",
        "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
        "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet pepper", "table", "tank",
        "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
        "turtle", "wardrobe", "whale", "willow tree", "wolf", "woman", "worm"
    ]

    if 0 <= class_num < len(classes):
        return classes[class_num]
    else:
        return "Invalid class number"
    
    

def apply_transformations(
    images: torch.Tensor, 
    down_res: int = 224, 
    up_res: int = 224, 
    jit_x: torch.Tensor = 0, 
    jit_y: torch.Tensor = 0, 
    down_noise: float = 0.0, 
    up_noise: float = 0.0, 
    contrast: torch.Tensor = 1.0, 
    color_amount: torch.Tensor = 1.0
) -> torch.Tensor:

    # # for MNIST alone
    # images = torch.mean(images,axis=1,keepdims=True)

    images_collected = []

    for i in range(images.shape[0]):

        image = images[i]

        # changing contrast
        image = torchvision.transforms.functional.adjust_contrast(image, contrast[i])

        # shift the result in x and y
        image = torch.roll(image,shifts=(jit_x[i], jit_y[i]),dims=(-2,-1))

        # shifting in the color <-> grayscale axis
        image = color_amount[i]*image + torch.mean(image,axis=0,keepdims=True)*(1-color_amount[i])

        images_collected.append(image)
    # import pdb; pdb.set_trace()
    images = torch.stack(images_collected, axis=0)

    # descrease the resolution
    images = F.interpolate(images, size=(down_res,down_res), mode='bicubic')

    # low res noise
    noise = down_noise * custom_rand((images.shape[0],3,down_res,down_res)).to("cuda")
    images = images + noise

    # increase the resolution
    images = F.interpolate(images, size=(up_res,up_res), mode='bicubic')

    # high res noise
    noise = up_noise * custom_rand((images.shape[0],3,up_res,up_res)).to("cuda")
    images = images + noise

    # clipping to the right range of values
    images = torch.clip(images,0,1)

    return images


def custom_rand(size: int) -> torch.Tensor:
    # setup_seed(0)
    return torch.Tensor(
        np.random.rand(*size)
    ).to("cuda")


def custom_choices(items: np.ndarray, tensor: torch.Tensor) -> torch.Tensor:
    # setup_seed(0)
    return np.random.choice(items,(len(tensor)))


def get_dataset(data_dir: str, classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if classes == 10:
        # Load the CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True,
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True)

        original_images_train_np = np.array(trainset.data)
        original_labels_train_np = np.array(trainset.targets)

        original_images_test_np = np.array(testset.data)
        original_labels_test_np = np.array(testset.targets)

    elif classes == 100:
        # Load the CIFAR-100 dataset
        trainset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True,
        )
        testset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True)

        original_images_train_np = np.array(trainset.data)
        original_labels_train_np = np.array(trainset.targets)

        original_images_test_np = np.array(testset.data)
        original_labels_test_np = np.array(testset.targets)
    
    else:
        raise ValueError("Invalid number of classes")

    images_train_np = original_images_train_np / 255.0
    images_test_np = original_images_test_np / 255.0
    
    return images_train_np, images_test_np, original_labels_train_np, original_labels_test_np



def make_multichannel_input(images: torch.Tensor,
                            up_res: int = 32,
                            resolutions: Sequence[int] = [32, 16, 8, 4],
                            jit_size: int = 3,
                            down_noise: float = 0.2,
                            up_noise: float = 0.2,
                            shuffle_image_versions_randomly: bool = False,
                            ):

    all_channels = []

    for i, down_res in enumerate(resolutions):
        jits_x = custom_choices(range(-jit_size,jit_size+1), images) # x-shift
        jits_y = custom_choices(range(-jit_size,jit_size+1), images) # y-shift
        contrasts = custom_choices(np.linspace(0.7,1.5,100), images) # change in contrast
        #? color_amounts = contrasts??
        color_amounts  = custom_choices(np.linspace(0.5,1.0,100), images) # change in color amount
        # import pdb; pdb.set_trace()
        
        
        # jits_x = custom_choices(range(-jit_size,jit_size+1), images+i) # x-shift
        # jits_y = custom_choices(range(-jit_size,jit_size+1), 51*images+7*i+125*r) # y-shift
        # contrasts = custom_choices(np.linspace(0.7,1.5,100), 7+3*images+9*i+5*r) # change in contrast
        # #? color_amounts = contrasts??
        # color_amounts = contrasts = custom_choices(np.linspace(0.5,1.0,100), 5+7*images+8*i+2*r) # change in color amount

        images_now = apply_transformations(
            images,
            down_res = down_res,
            up_res = up_res,
            jit_x = jits_x,
            jit_y = jits_y,
            down_noise = down_noise,
            up_noise = up_noise,
            contrast = contrasts,
            color_amount = color_amounts,
        )

        all_channels.append(images_now)

    if not shuffle_image_versions_randomly:
        return torch.concatenate(all_channels,axis=1)
    elif shuffle_image_versions_randomly:
        indices = torch.randperm(len(all_channels))
        shuffled_tensor_list = [all_channels[i] for i in indices]
        return torch.concatenate(shuffled_tensor_list,axis=1)

def plot_images(images_test_np: np.ndarray, resolutions: Sequence[int]) -> None:
    sample_images = images_test_np[:5]  # (5, 32, 32, 3)
    for j in [0,1]:
        # shape: (5, 32, 32, 12)
        multichannel_images = make_multichannel_input(
            torch.Tensor(sample_images.transpose([0,3,1,2])).to("cuda"),
        ).detach().cpu().numpy().transpose([0,2,3,1])

        N = 1 + multichannel_images.shape[3] // 3
        plt.figure(figsize=(N*5.5,5))

        plt.subplot(1,N,1)
        plt.title("original")
        plt.imshow(sample_images[j])
        plt.xticks([],[])
        plt.yticks([],[])

        for i in range(N-1):
            plt.subplot(1,N,i+2)
            plt.title(f"res={resolutions[i]}")
            plt.imshow(multichannel_images[j,:,:,3*i:3*(i+1)])
            plt.xticks([],[])
            plt.yticks([],[])

        plt.show()
    plt.savefig("example_images.png")
        
