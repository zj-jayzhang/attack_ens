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
from collections import Counter
from tqdm import tqdm




# def visualize_adv_images(clean_xs, clean_ys, adv_xs, model, reps=10, description=""):
    
#     all_clean_pred_argmaxes = []
#     all_pred_argmaxes = []

#     # Compute predictions for clean images
#     for r in range(reps):
#         with torch.no_grad():
#             logits = model(clean_xs).detach().cpu().numpy()
#             pred_argmaxes = np.argmax(logits, axis=-1)
#             all_clean_pred_argmaxes.append(pred_argmaxes)

#     # Compute predictions for adversarial images
#     for r in range(reps):
#         with torch.no_grad():
#             logits = model(adv_xs).detach().cpu().numpy()
#             pred_argmaxes = np.argmax(logits, axis=-1)
#             all_pred_argmaxes.append(pred_argmaxes)

#     num_images = len(clean_xs)
#     images_per_page = 5

#     # Iterate over clean_xs in batches of 10 images
#     for page in range(0, num_images, images_per_page):
#         plt.figure(figsize=(15, images_per_page * 3))  # Set figure size
#         end = min(page + images_per_page, num_images)

#         for idx, i in enumerate(range(page, end)):
#             # Plot column 1: prediction distribution for clean images
#             plt.subplot(images_per_page, 3, idx * 3 + 1)
#             vals = [pred_argmaxes[i] for pred_argmaxes in all_clean_pred_argmaxes]
#             vals_sorted, freqs_sorted = zip(*Counter(vals).most_common())
#             plt.xticks([], [])
#             plt.yticks([], [])

#             title = f"ground truth={cifar100_class_to_description(clean_ys[i])} c={clean_ys[i]}\n"
#             for j in range(len(vals_sorted)):
#                 title += f"c={vals_sorted[j]} {cifar100_class_to_description(vals_sorted[j])} = {freqs_sorted[j]}/{reps}\n"
#             plt.title(title[:-1])

#             plt.imshow(clean_xs[i].detach().cpu().numpy().transpose([1, 2, 0]))

#             # Plot column 2: attack perturbation
#             plt.subplot(images_per_page, 3, idx * 3 + 2)
#             plt.title("attack perturbation")
#             plt.imshow(0.5 + clean_xs[i].detach().cpu().numpy().transpose([1, 2, 0]) - adv_xs[i].detach().cpu().numpy().transpose([1, 2, 0]))
#             plt.xticks([], [])
#             plt.yticks([], [])

#             # Plot column 3: prediction distribution for adversarial images
#             plt.subplot(images_per_page, 3, idx * 3 + 3)
#             vals = [pred_argmaxes[i] for pred_argmaxes in all_pred_argmaxes]
#             vals_sorted, freqs_sorted = zip(*Counter(vals).most_common())
#             plt.xticks([], [])
#             plt.yticks([], [])

#             title = ""
#             for j in range(len(vals_sorted)):
#                 title += f"c={vals_sorted[j]} {cifar100_class_to_description(vals_sorted[j])} = {freqs_sorted[j]}/{reps}\n"
#             plt.title(title[:-1])

#             plt.imshow(adv_xs[i].detach().cpu().numpy().transpose([1, 2, 0]))
#             plt.tight_layout()
            
#         # Save each batch of 10 images as a separate PNG
#         plt.savefig(f"imgs/{description}_adv_batch_{page//images_per_page + 1}.png")
#         plt.close()





def eval_model(model,images_in,labels_in,batch_size=128, forward_fn="original", layers=0):
    all_preds = []
    all_logits = []

    with torch.no_grad():
        its = int(np.ceil(float(len(images_in))/float(batch_size)))

        pbar = tqdm(range(its), desc='Eval', ncols=100)

        for it in pbar:
            i1 = it*batch_size
            i2 = min([(it+1)*batch_size, len(images_in)])

            inputs = torch.Tensor(images_in[i1:i2].transpose([0,3,1,2])).to("cuda")
            if forward_fn == "original":
                outputs = model.forward_original(inputs)
            elif forward_fn == "ensemble":
                outputs = model(inputs)
            elif forward_fn == "linear":
                    outputs = model.predict_from_layer(inputs, layers)
            all_logits.append(outputs.detach().cpu().numpy())

            preds = torch.argmax(outputs,axis=-1)
            all_preds.append(preds.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds,axis=0)
    all_logits = np.concatenate(all_logits,axis=0)

    return np.sum(all_preds == labels_in), all_preds.shape[0], all_logits


        
        

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
    
def cifar10_class_to_description(class_num):
    classes = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ]

    if 0 <= class_num < len(classes):
        return classes[class_num]
    else:
        return "Invalid class number"

# apply image augmentations to input images
def apply_transformations(
    images,
    down_res = 224,
    up_res = 224,
    jit_x = 0,
    jit_y = 0,
    down_noise = 0.0,
    up_noise = 0.0,
    contrast = 1.0,
    color_amount = 1.0,
    ):

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

  images = torch.stack(images_collected, axis=0)

  # descrease the resolution
  images = F.interpolate(images, size=(down_res,down_res), mode='bicubic')

  # low res noise
  noise = down_noise * custom_rand(images+312, (images.shape[0],3,down_res,down_res)).to("cuda")
  images = images + noise

  # increase the resolution
  images = F.interpolate(images, size=(up_res,up_res), mode='bicubic')

  # high res noise
  noise = up_noise * custom_rand(images+812,(images.shape[0],3,up_res,up_res)).to("cuda")
  images = images + noise

  # clipping to the right range of values
  images = torch.clip(images,0,1)

  return images



def make_multichannel_input(images: torch.Tensor,
                            up_res: int = 32,
                            resolutions: Sequence[int] = [32, 16, 8, 4],
                            jit_size: int = 3,
                            down_noise: float = 0.2,
                            up_noise: float = 0.2,
                            shuffle_image_versions_randomly: bool = False,
                            ):
  all_channels = []

  for i,r in enumerate(resolutions):

    down_res = r

    jits_x = custom_choices(range(-jit_size,jit_size+1), images+i) # x-shift
    jits_y = custom_choices(range(-jit_size,jit_size+1), 51*images+7*i+125*r) # y-shift
    contrasts = custom_choices(np.linspace(0.7,1.5,100), 7+3*images+9*i+5*r) # change in contrast
    color_amounts = contrasts = custom_choices(np.linspace(0.5,1.0,100), 5+7*images+8*i+2*r) # change in color amount

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


# to be able to replace the random number generator by other things if needed
def custom_rand(input_tensor, size):
    # setup_seed(42)
    return torch.Tensor(
        np.random.rand(*size)
    ).to("cuda")

def custom_choices(items, tensor):
    # setup_seed(42)
    return np.random.choice(items,(len(tensor)))





def get_dataset(data_dir: str, classes: int, batch_size: int = 64):
    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    
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
        
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

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
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
    else:
        raise ValueError("Invalid number of classes")

    images_train_np = original_images_train_np / 255.0
    images_test_np = original_images_test_np / 255.0
    
    return images_train_np, images_test_np, original_labels_train_np, original_labels_test_np, test_loader



def plot_images(images_test_np: np.ndarray, resolutions: Sequence[int], save_path: str = None):
    sample_images = images_test_np[:1]  

    multichannel_images = make_multichannel_input(
        torch.Tensor(sample_images.transpose([0,3,1,2])).to("cuda"),
    ).detach().cpu().numpy().transpose([0,2,3,1])

    N = 1 + multichannel_images.shape[3] // 3
    plt.figure(figsize=(N*5.5,5))

    plt.subplot(1,N,1)
    plt.title("original")
    plt.imshow(sample_images[0])
    plt.xticks([],[])
    plt.yticks([],[])

    for i in range(N-1):
        plt.subplot(1,N,i+2)
        plt.title(f"res={resolutions[i]}")
        plt.imshow(multichannel_images[0,:,:,3*i:3*(i+1)])
        plt.xticks([],[])
        plt.yticks([],[])

    plt.show()
    plt.savefig(f"{save_path}/example_images.png")
        
