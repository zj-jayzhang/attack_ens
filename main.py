import os
import time
from helpers.attack import adaptive_attack, benchmark, non_adaptive_attack
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from helpers.networks import TargetModel, get_network
from helpers.utils import eval_model, fgsm_attack, get_dataset, make_multichannel_input, plot_images, setup_seed
import random
import copy
import matplotlib.pyplot as plt
from contextlib import contextmanager
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt




@contextmanager
def isolated_environment():
    # Save the current state of random seeds and numpy precision
    np_random_state = np.random.get_state()
    python_random_state = random.getstate()
    torch_random_state = torch.get_rng_state()
    cuda_random_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_print_options = np.get_printoptions()
    try:
        # Execute the block of code
        yield
    finally:
        # Restore the saved state
        np.random.set_state(np_random_state)
        random.setstate(python_random_state)
        torch.set_rng_state(torch_random_state)
        if cuda_random_state:
            torch.cuda.set_rng_state_all(cuda_random_state)
        np.set_printoptions(**numpy_print_options)









def train_model(
    model_in,
    images_in,
    labels_in,
    epochs=10,
    lr=1e-3,
    batch_size=512,
    optimizer_in=optim.Adam,
    subset_only=None,
    mode="eval",
    use_adversarial_training=False,
    adversarial_epsilon=8/255,
    skip_test_set_eval=False,
    images_test_np=None,
    labels_test_np=None,
    forward_fn="original",
    layers=0,
):

    global storing_models
    train_accs, test_accs = [], []
    
    if mode == "train":
        model_in.train()
    elif mode == "eval":
        model_in.eval()

    criterion = nn.CrossEntropyLoss()

        
    if subset_only is None:
        train_optimizer = optimizer_in(model_in.imported_model.parameters(), lr=lr)
    else:
        train_optimizer = optimizer_in(model_in.linear_layers[layers].parameters(), lr=lr)
    setup_seed(1)
    for epoch in range(epochs):
        # setup_seed(1+epoch)
        randomized_ids = np.random.permutation(range(len(images_in)))
        its = int(np.ceil(float(len(images_in)) / float(batch_size)))
        pbar = tqdm(range(its), desc='Training', ncols=100)

        all_hits = []

        for it in pbar:
            i1 = it * batch_size
            i2 = min([(it + 1) * batch_size, len(images_in)])

            ids_now = randomized_ids[i1:i2]

            np_images_used = images_in[ids_now]
            np_labels_used = labels_in[ids_now]

            inputs = torch.Tensor(np_images_used.transpose([0, 3, 1, 2])).to("cuda")
            
            # Adversarial training if enabled
            if use_adversarial_training:
                attacked_images = fgsm_attack(
                    model_in.eval(),
                    np_images_used[:],
                    np_labels_used[:],
                    epsilon=adversarial_epsilon,
                    random_reps=1,
                    batch_size=batch_size // 2,
                )
                np_images_used = attacked_images
                np_labels_used = np_labels_used

                if mode == "train":
                    model_in.train()
                elif mode == "eval":
                    model_in.eval()

            inputs = torch.Tensor(np_images_used.transpose([0, 3, 1, 2])).to("cuda")
            labels = torch.Tensor(np_labels_used).to("cuda").to(torch.long)
            
            # Zero the parameter gradients
            train_optimizer.zero_grad()

            inputs_used = inputs

            # Actual optimization step
            if forward_fn == "original":    
                outputs = model_in.forward_original(inputs_used)
            elif forward_fn == "linear":
                # if subset_only:
                #     import pdb; pdb.set_trace() 
                outputs = model_in.predict_from_layer(inputs_used, layers)
            
            loss = criterion(outputs, labels)
            loss.backward()
            train_optimizer.step()
            
            # On-the-fly eval for the train set batches
            preds = torch.argmax(outputs, axis=-1)
            acc = torch.mean((preds == labels).to(torch.float), axis=-1)
            all_hits.append((preds == labels).to(torch.float).detach().cpu().numpy())
            train_accs.append(acc.detach().cpu().numpy())

            pbar.set_description(f"train acc={acc.detach().cpu().numpy()} loss={loss.item()}")

        if not skip_test_set_eval:
            with isolated_environment():
                eval_model_copy = copy.deepcopy(model_in)
                test_hits, test_count, _ = eval_model(eval_model_copy.eval(), images_test_np, labels_test_np, forward_fn=forward_fn, layers=layers)
        else:
            test_hits = 0
            test_count = 1

        # End of epoch evaluation
        train_hits = np.sum(np.concatenate(all_hits, axis=0).reshape([-1]))
        train_count = np.concatenate(all_hits, axis=0).reshape([-1]).shape[0]
        print(f"e={epoch} train {train_hits} / {train_count} = {train_hits/train_count},  test {test_hits} / {test_count} = {test_hits/test_count}")

        test_accs.append(test_hits / test_count)

    print('\nFinished Training')

    return model_in






def get_args():
    
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--bs', default=48, type=int, help='batch size')
    parser.add_argument('--steps', default=20, type=int, help='number of steps')
    parser.add_argument('--eot', default=1, type=int, help='number of eot')
    # save_path
    parser.add_argument('--save_path', default="/data/projects/ensem_adv/ckpts_test", type=str, help='save path')
    parser.add_argument('--img_path', default="./imgs", type=str, help='save path for images')
    # data_dir
    parser.add_argument('--data_dir', default="/local/home/jiezha/data/", type=str, help='data directory')
    # resolution
    parser.add_argument('--resolutions', default=[32,16,8,4], type=list, help='resolution')
    # dataset, cifar10 or cifar100
    parser.add_argument('--dataset', default="cifar100", type=str, choices=["cifar10", "cifar100"], help='dataset')
    # layers_to_use
    parser.add_argument('--layers_to_use', default=[20,30,35,40,45,50,52], type=list, help='layers to use')
    # epochs_cls
    parser.add_argument('--epochs_cls', default=6, type=int, help='epochs for classification')
    # epochs_all
    parser.add_argument('--epochs_all', default=1, type=int, help='epochs for all linear layers')
    # lr_cls
    parser.add_argument('--lr_cls', default=3.3e-5, type=float, help='lr for classification')
    # lr_all
    parser.add_argument('--lr_all', default=3.3e-5, type=float, help='lr for all linear layers')
    # num of test images
    parser.add_argument('--num_test', default=100, type=int, help='number of test images')
    args = parser.parse_args()
    return args


def main():
    # 1. Set up the parameters
    args = get_args()
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(args.img_path):
        os.makedirs(args.img_path)
    
    args.classes = 100 if args.dataset == "cifar100" else 10
    images_train_np, images_test_np, labels_train_np, labels_test_np, _ = get_dataset(data_dir=args.data_dir, classes=args.classes)
    # visualize the first image to show 4 resolutions
    plot_images(images_test_np, resolutions=args.resolutions, save_path=args.img_path)
    
    # 2. Set up the model
    network = get_network()
    # mannuallly setting the first conv layer to be multi-res
    network.conv1 = nn.Conv2d(len(args.resolutions)  * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    network.fc = nn.Linear(2048, args.classes)
    
    model = TargetModel(network, make_multichannel_input, classes=args.classes, resolutions=args.resolutions).to("cuda")
    

    # 3. Finetune with the classification layer
    if os.path.exists(f"{save_path}/model.pth"):
        print("============== Loading model ==============")
        model.load_state_dict(torch.load(f"{save_path}/model.pth"))
        # test_hits,test_count,_ = eval_model(model, images_test_np, labels_test_np)
        # print(f"loading model >>>,  test={test_hits}/{test_count}={test_hits/test_count}")
    else:
        print("============== Finetuning model ==============")
        model = train_model(
            model,
            images_train_np,
            labels_train_np,
            epochs=args.epochs_cls,
            lr=args.lr_cls,
            optimizer_in = optim.Adam,
            batch_size=128,
            mode="train",
            images_test_np=images_test_np,
            labels_test_np=labels_test_np,
            )
        torch.save(model.state_dict(), f"{save_path}/model.pth")
    
    
    # 4. Train the linear layers for intermediate layers
    # load the new weights from finetuned model
    model._layer_operations(model.imported_model)
    for layer_i in reversed(args.layers_to_use):
        # setup_seed(1)
        if os.path.exists(f"{save_path}/linear_model_{layer_i}.pth"):
            print(f"============== Loading linear model {layer_i} ==============")
            model.linear_layers[layer_i].load_state_dict(torch.load(f"{save_path}/linear_model_{layer_i}.pth"))
            # test_hits,test_count,_ = eval_model(model, images_test_np, labels_test_np, forward_fn="linear", layers=layer_i)
            # print("Loaded linear model, test acc = ", test_hits/test_count)
        else:
            print(f"============== Finetuning linear model {layer_i} ==============")
            linear_model = train_model(
                copy.deepcopy(model),
                images_train_np[:],
                labels_train_np[:],
                epochs=args.epochs_all,
                lr=args.lr_all,
                optimizer_in = optim.Adam,
                batch_size=64,
                mode="train",
                subset_only = True, 
                use_adversarial_training=False,
                adversarial_epsilon=None,
                images_test_np=images_test_np,
                labels_test_np=labels_test_np,
                forward_fn="linear",
                layers=layer_i,
                )
            torch.save(linear_model.linear_layers[layer_i].state_dict(), f"{save_path}/linear_model_{layer_i}.pth")

            model.linear_layers[layer_i] = copy.deepcopy(linear_model.linear_layers[layer_i])


    def test_per_layer():
        test_acc_by_layer = []
        for layer_i in args.layers_to_use:
            test_hits,test_count,_ = eval_model(model, images_test_np, labels_test_np, forward_fn="linear", layers=layer_i)
            test_acc_by_layer.append(test_hits/test_count)
            print(f"layer={layer_i} test={test_hits}/{test_count}={test_hits/test_count}")
            

        plt.figure(figsize=(7,5), dpi=100)
        plt.title("Accuracy at intermediate layers",fontsize=14)
        plt.plot(args.layers_to_use,test_acc_by_layer,marker="o",color="navy",label="Test")
        plt.legend(fontsize=16)
        plt.xlabel("Layer",fontsize=14)
        plt.ylabel("Accuracy",fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f"{args.img_path}/acc_by_layer.png")

    def test_ensemble():
        test_hits,test_count,_ = eval_model(model, images_test_np, labels_test_np, forward_fn="ensemble")
        print(f"test={test_hits}/{test_count}={test_hits/test_count}")
        self_ensemble_test_acc = test_hits / test_count
        print(f"Self-ensemble test acc = {self_ensemble_test_acc}")
        print("\n---------------------------------------------\n")

    test_per_layer()
    test_ensemble()
    
    # 5. Evaluate the robustness of the model under PGD attack, non-adaptive attack
    # non_adaptive_attack(model, args=args, targetd_attack=False)

    # 6. Evaluate the robustness of the model under adaptive attack
    adaptive_attack(model, args=args)
    
    # save the whole model, not the parameters
    # torch.save(model, "model_whole.pth")
    # import pdb; pdb.set_trace()
    
    # 7. Evaluate the robustness of the model under AutoAttack
    with isolated_environment():
        time_start = time.time()
        benchmark(
                model.eval(),
                dataset=args.dataset,
                threat_model='Linf',
                device=torch.device("cuda"),
                eps=8/255,
                n_examples=args.num_test, 
                version='rand',
                batch_size=32,
                args=args
            )
        time_end = time.time()
        print(f"Time cost:  {(time_end-time_start)/60} mins")
    
 
    
    


if __name__ == '__main__':
    setup_seed(1)
    main()

"""
CUDA_VISIBLE_DEVICES=7 python main.py --bs=8 --steps=200 --eot=8 --dataset cifar10 --save_path=/data/projects/ensem_adv/ckpts_cifar10


"""