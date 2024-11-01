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
    

#! per-sample randomness
def apply_transformations(
    images: torch.Tensor, 
    down_res: int = 224, 
    up_res: int = 224, 
    down_noise: float = 0.0, 
    up_noise: float = 0.0, 
    fix_seed: bool = False
) -> torch.Tensor:

    images_collected = []
    jit_size = 3
    for i in range(images.shape[0]):
        # Select a single sample
        image = images[i]
        
        jit_x = custom_choices(range(-jit_size,jit_size+1),  fix_seed)
        jit_y = custom_choices(range(-jit_size,jit_size+1), fix_seed)
        contrast = custom_choices(np.linspace(0.7,1.5,100),  fix_seed)
        color_amount  = custom_choices(np.linspace(0.5,1.0,100), fix_seed)
        # print("sum over all", jit_x+jit_y+contrast+color_amount)
        # Adjust contrast
        # import pdb; pdb.set_trace()
        image = torchvision.transforms.functional.adjust_contrast(image, contrast)

        # Shift in the x and y directions
        image = torch.roll(image, shifts=(jit_x, jit_y), dims=(-2, -1))

        # Shift between color and grayscale
        image = color_amount * image + torch.mean(image, axis=0, keepdims=True) * (1 - color_amount)

        # Downsample
        image = F.interpolate(image.unsqueeze(0), size=(down_res, down_res), mode='bicubic').squeeze(0)

        # Add low-resolution noise
        noise_down = down_noise * custom_rand((1, 3, down_res, down_res), fix_seed).to(image.device)
        # import pdb; pdb.set_trace()
        # print("down_noise", noise.sum())
        image = image + noise_down.squeeze(0)

        # Upsample
        image = F.interpolate(image.unsqueeze(0), size=(up_res, up_res), mode='bicubic').squeeze(0)

        # Add high-resolution noise
        noise_up = up_noise * custom_rand((1, 3, up_res, up_res), fix_seed).to(image.device)
        # print("up_noise", noise.sum())
        image = image + noise_up.squeeze(0)

        # Clip values to the valid range [0, 1]
        image = torch.clip(image, 0, 1)

        # Add to the result list
        images_collected.append(image)
        # print(f"down_res={down_res},  down_noise={noise_down.sum()}, up_noise={noise_up.sum()}, sum over all:{jit_x+jit_y+contrast+color_amount}")

    # Stack the results into a batch tensor
    images = torch.stack(images_collected, axis=0)

    return images





def custom_rand(size: int, fix_seed: bool = False) -> torch.Tensor:
    if fix_seed:
        setup_seed(0)
    # setup_seed(0)
    #! make sure len(size)=1
    # assert len(size) == 1
    random_tensor = torch.Tensor(
        np.random.rand(*size)
    )
    # print("stage 1", random_tensor.sum())
    # import pdb; pdb.set_trace()
    return random_tensor.to("cuda")
    # return torch.Tensor(
    #     np.random.rand(*size)
    # ).to("cuda")


def custom_choices(items: np.ndarray,  fix_seed: bool = False) -> torch.Tensor:
    if fix_seed:
        setup_seed(0)
    # setup_seed(0)
    #! make sure len(tensor)=1
    # assert len(tensor) == 1
    random_np = np.random.choice(items,(1)) 
    # print("stage 2", random_np.sum())
    # import pdb; pdb.set_trace()
    return random_np[0]
    # return np.random.choice(items,(len(tensor)))


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
                            fix_seed: bool = False
                            ):

    all_channels = []

    for i, down_res in enumerate(resolutions):
        # jits_x = custom_choices(range(-jit_size,jit_size+1), images, fix_seed)
        # jits_y = custom_choices(range(-jit_size,jit_size+1), images, fix_seed)
        # contrasts = custom_choices(np.linspace(0.7,1.5,100), images, fix_seed)
        # #? color_amounts = contrasts??
        # color_amounts  = custom_choices(np.linspace(0.5,1.0,100), images, fix_seed)


        images_now = apply_transformations(
            images,
            down_res = down_res,
            up_res = up_res,
            down_noise = down_noise,
            up_noise = up_noise,
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
        
