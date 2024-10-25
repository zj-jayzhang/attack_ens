import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from helpers.networks import SourceModel, TargetModel, get_network
from helpers.utils import fgsm_attack, get_dataset, make_multichannel_input, plot_images, setup_seed
from torch.autograd import Variable
import random
import time
import copy
import torchvision
import matplotlib.pyplot as plt
from contextlib import contextmanager
from tqdm import tqdm

import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import random
from autoattack import AutoAttack
from autoattack.state import EvaluationState
from torch import nn
from tqdm import tqdm

from robustbench.data import CORRUPTIONS_DICT, get_preprocessing, load_clean_dataset
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import clean_accuracy, update_json


from collections import Counter




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
        # model.linear_layers[layer_i].parameters(),
        train_optimizer = optimizer_in(model_in.linear_layers[layers].parameters(), lr=lr)
        
        # train_optimizer = optimizer_in(subset_only, lr=lr)   # doesn't work for some reason

    for epoch in range(epochs):

        randomized_ids = np.random.permutation(range(len(images_in)))


        # Ensure the model is in the correct eval/train mode every epoch
        # if mode == "train":
        #     model_in.train()
        # elif mode == "eval":
        #     model_in.eval()
        # else:
        #     assert False

        its = int(np.ceil(float(len(images_in)) / float(batch_size)))
        pbar = tqdm(range(its), desc='Training', ncols=100)

        all_hits = []

        for it in pbar:
            # import pdb; pdb.set_trace()
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
            # outputs = model_in(inputs_used)
            if forward_fn == "original":
                outputs = model_in.forward_original(inputs_used)
            elif forward_fn == "linear":
                outputs = model_in.predict_from_layer(inputs_used, layers)
            loss = criterion(outputs, labels)
            loss.backward()
            # print(model_in.linear_layers[1][1].weight.grad)
            train_optimizer.step()

            # On-the-fly eval for the train set batches
            preds = torch.argmax(outputs, axis=-1)
            acc = torch.mean((preds == labels).to(torch.float), axis=-1)
            all_hits.append((preds == labels).to(torch.float).detach().cpu().numpy())
            train_accs.append(acc.detach().cpu().numpy())

            pbar.set_description(f"train acc={acc.detach().cpu().numpy()} loss={loss.item()}")

        # import pdb; pdb.set_trace()
        # model_in.linear_layers[1][1].weight.sum() = tensor(-0.4745, device='cuda:0', grad_fn=<SumBackward0>)
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









    


def benchmark(
    model: Union[nn.Module, Sequence[nn.Module]],
    n_examples: int = 10000,
    dataset: Union[str, BenchmarkDataset] = BenchmarkDataset.cifar_10,
    threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
    to_disk: bool = False,
    model_name: Optional[str] = None,
    data_dir: str = "./data",
    corruptions_data_dir: Optional[str] = None,
    device: Optional[Union[torch.device, Sequence[torch.device]]] = None,
    batch_size: int = 32,
    eps: Optional[float] = None,
    log_path: Optional[str] = None,

    version='custom', #'rand', # 'rand' for models using stochasticity

    preprocessing: Optional[Union[str, Callable]] = None,
    aa_state_path: Optional[Path] = None,

    ) -> Tuple[float, float]:

    if isinstance(model, Sequence) or isinstance(device, Sequence):
        # Multiple models evaluation in parallel not yet implemented
        raise NotImplementedError

    try:
        if model.training:
            warnings.warn(Warning("The given model is *not* in eval mode."))
    except AttributeError:
        warnings.warn(
            Warning(
                "It is not possible to asses if the model is in eval mode"))

    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    threat_model_: ThreatModel = ThreatModel(threat_model)
    model = model.to(device)

    prepr = get_preprocessing(dataset_, threat_model_, model_name,
                            preprocessing)

    clean_x_test, clean_y_test = load_clean_dataset(dataset_, n_examples,
                                                    data_dir, prepr)


    accuracy = clean_accuracy(model,
                            clean_x_test,
                            clean_y_test,
                            batch_size=batch_size,
                            device=device)
    print(f'With {len(clean_x_test)} clean examples, clean accuracy: {accuracy:.2%}')

    adversary = AutoAttack(
                        model,  # at default, this will call aggreegated model
                        norm=threat_model_.value,
                        eps=eps,
                        version=version,
                        device=device,
                        log_path=log_path,
                        attacks_to_run=['apgd-ce', 'apgd-t'] if version == "custom" else [], #Stan's addition
                        )
    x_adv = adversary.run_standard_evaluation(
                                            clean_x_test,
                                            clean_y_test,
                                            bs=batch_size,
                                            state_path=aa_state_path)
    adv_accuracy = clean_accuracy(
                                model,
                                x_adv,
                                clean_y_test,
                                batch_size=batch_size,
                                device=device)
    print(f'With {len(clean_x_test)} clean examples, robust accuracy: {adv_accuracy:.2%}')


def _pgd_blackbox(
                model_target,
                model_source,
                X,
                y,
                epsilon=0.031,
                num_steps=20,
                step_size=0.003,
                random=True,) -> Tuple[int, int]:
    # [20,30,35,40,45,50,52]
    layer = 50
    #! make sure that for the target model, all defenses are turned on
    # model_source.fix_seed = True
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            # loss = nn.CrossEntropyLoss()(model_source.get_logits_from_layer(X_pgd, layer), y)
            loss = nn.CrossEntropyLoss()(model_source.get_logits_from_several_layers(X_pgd, layer), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd_on_source = (model_source.get_logits_from_several_layers(X_pgd, layer).data.max(1)[1] != y.data).float().sum()
    # err_pgd_on_source = (model_source.get_logits_from_layer(X_pgd, layer).data.max(1)[1] != y.data).float().sum()
    
    err_pgd_on_target = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    print(f'clean_err={err.item()}, adv_err_on_source={err_pgd_on_source.item()}, adv_err_on_target={err_pgd_on_target.item()}')
    return err, err_pgd_on_source, err_pgd_on_target


def _pgd_whitebox(
                model,
                X,
                y,
                epsilon=0.031,
                num_steps=20,
                step_size=0.003,
                random=True,
                ):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    
    print(f'clean_err={err.item()}, adv_err={err_pgd.item()}')
    return err, err_pgd


def eval_adv_test_whitebox(model, device, data_dir):
    """
    evaluate model by white-box attack
    """
    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)


    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
        total += data.size(0)
        if total >= 1000:
            break
    print(f"clean accuracy: {(1 - natural_err_total / total):.2%}, robust accuracy: {(1 - robust_err_total / total):.2%}")

def eval_adv_test_blackbox(model_target, device, data_dir, images_test_np, labels_test_np, layer_i=40):
    """
    evaluate model by black-box attack
    """
    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    

    model_source = SourceModel(copy.deepcopy(model_target.imported_model), model_target.multichannel_fn, classes=100).to("cuda")
    model_source.layer_operations = copy.deepcopy(model_target.layer_operations)
    model_source.linear_layers = copy.deepcopy(model_target.linear_layers)
    model_source.fix_seed = False
    
    model_target.eval()
    model_source.eval()
    
    # test_hits,test_count,_ = eval_model(model_source, images_test_np, labels_test_np, forward_fn="ensemble")
    # print(f"source model test={test_hits}/{test_count}={test_hits/test_count}")
    # test_hits,test_count,_ = eval_model(model_target, images_test_np, labels_test_np, forward_fn="ensemble")
    # print(f"target model test={test_hits}/{test_count}={test_hits/test_count}")
    # import pdb; pdb.set_trace()
    
    # test the accuracy of the target model and source model
    

    robust_err_total_source = 0
    robust_err_total_target = 0
    natural_err_total = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust_on_source, err_robust_on_target = _pgd_blackbox(model_target, model_source, X, y)
        robust_err_total_source += err_robust_on_source
        robust_err_total_target += err_robust_on_target
        natural_err_total += err_natural
        total += data.size(0)
        if total >= 1000:
            break
        
    print(f"clean accuracy: {(1 - natural_err_total / total):.2%}, robust accuracy on source: {(1 - robust_err_total_source / total):.2%}, robust accuracy on target: {(1 - robust_err_total_target / total):.2%}")


#! This is an example on CIFAR-100.
def main():
    # 1. Set up the environment and the model
    save_path = "/data/projects/ensem_adv/ckpts_3"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_dir = "/local/home/jiezha/data/"
    setup_seed(2024)
    resolutions = [32,16,8,4] # pretty arbitrary
    classes = 100
    in_planes = 3
    planes = 64
    stride = 2
    N = len(resolutions)  # input channels multiplier due to multi-res input
    lr_cls = 3.3e-5
    epochs_cls = 6
    lr_all=3.3e-5 # random stuff again
    epochs_all = 1
    batch_size = 64 # for CUDA RAM reasons
    layers_to_use = [20,30,35,40,45,50,52]
    
    images_train_np, images_test_np, labels_train_np, labels_test_np = get_dataset(data_dir=data_dir, classes=classes)
    # plot_images(images_test_np, resolutions=resolutions)
    network = get_network()
    # mannuallly setting the first conv layer to be multi-res
    network.conv1 = nn.Conv2d(N * in_planes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
    network.fc = nn.Linear(2048, classes)
    
    model = TargetModel(network, make_multichannel_input, classes=classes, resolutions=resolutions).to("cuda")
    

    # 2. Train the model
    if os.path.exists(f"{save_path}/model.pth"):
        model.load_state_dict(torch.load(f"{save_path}/model.pth"))
        # test_hits,test_count,_ = eval_model(model, images_test_np, labels_test_np)
        # print(f"loading model >>>,  test={test_hits}/{test_count}={test_hits/test_count}")
    else:
        model = train_model(
            model,
            images_train_np,
            labels_train_np,
            epochs=epochs_cls,
            lr=lr_cls,
            optimizer_in = optim.Adam,
            batch_size=128,
            mode="train",
            images_test_np=images_test_np,
            labels_test_np=labels_test_np,
            )
        torch.save(model.state_dict(), f"{save_path}/model.pth")
    
    
    # 3. Train the linear layers
    #! only train several specific linear layers
    model._layer_operations(model.imported_model)
    for layer_i in reversed(layers_to_use):
        print(f"============== layer={layer_i} ==============")
        if os.path.exists(f"{save_path}/linear_model_{layer_i}.pth"):
            model.linear_layers[layer_i].load_state_dict(torch.load(f"{save_path}/linear_model_{layer_i}.pth"))
            # test_hits,test_count,_ = eval_model(model, images_test_np, labels_test_np, forward_fn="linear", layers=layer_i)
            # print("Loaded linear model, test acc = ", test_hits/test_count)
        else:
            linear_model = train_model(
                copy.deepcopy(model),
                images_train_np[:],
                labels_train_np[:],
                epochs=epochs_all,
                lr=lr_all,
                optimizer_in = optim.Adam,
                batch_size=batch_size,
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
        for layer_i in layers_to_use:
            test_hits,test_count,_ = eval_model(model, images_test_np, labels_test_np, forward_fn="linear", layers=layer_i)
            test_acc_by_layer.append(test_hits/test_count)
            print(f"layer={layer_i} test={test_hits}/{test_count}={test_hits/test_count}")
            

        plt.figure(figsize=(7,5), dpi=100)
        plt.title("Accuracy at intermediate layers",fontsize=14)
        plt.plot(layers_to_use,test_acc_by_layer,marker="o",color="navy",label="Test")
        plt.legend(fontsize=16)
        plt.xlabel("Layer",fontsize=14)
        plt.ylabel("Accuracy",fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig("layer_accuracies.png")

    # test_per_layer()
    def test_ensemble():
        test_hits,test_count,_ = eval_model(model, images_test_np, labels_test_np, forward_fn="ensemble")
        print(f"test={test_hits}/{test_count}={test_hits/test_count}")
        self_ensemble_test_acc = test_hits / test_count
        print(f"Self-ensemble test acc = {self_ensemble_test_acc}")
        print("\n---------------------------------------------\n")

    # eval_adv_test_whitebox(model, device="cuda", data_dir=data_dir)
    eval_adv_test_blackbox(model, device="cuda", data_dir=data_dir, images_test_np=images_test_np, labels_test_np=labels_test_np)
    
    if False:
    
        t1 = time.time()

        attack_samples = 32
        with isolated_environment():
            benchmark(
                model.eval(),
                dataset=f'cifar{classes}',
                threat_model='Linf',
                device=torch.device("cuda"),
                eps=8/255,
                n_examples=attack_samples, 
                version='rand',
                batch_size=32,
                data_dir=data_dir,
                )

        t2 = time.time()
        print(f"Time taken = {int(t2-t1)} seconds")
    

    
    
    

    
if __name__ == '__main__':
    main()