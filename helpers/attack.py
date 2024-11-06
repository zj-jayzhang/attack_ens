
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from helpers.networks import SourceModel
from helpers.utils import eval_model, get_dataset
from torch.autograd import Variable
import copy
from tqdm import tqdm
from typing import Tuple
import numpy as np
from robustbench.data import CORRUPTIONS_DICT, get_preprocessing, load_clean_dataset
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import clean_accuracy, update_json
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union
import argparse
from autoattack import AutoAttack
from autoattack.state import EvaluationState

def benchmark(
    model: Union[nn.Module, Sequence[nn.Module]],
    n_examples: int = 10000,
    dataset: Union[str, BenchmarkDataset] = BenchmarkDataset.cifar_10,
    threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
    model_name: Optional[str] = None,
    data_dir: str = "./data",
    device: Optional[Union[torch.device, Sequence[torch.device]]] = None,
    batch_size: int = 32,
    eps: Optional[float] = None,
    log_path: Optional[str] = None,

    version='custom', #'rand', # 'rand' for models using stochasticity

    preprocessing: Optional[Union[str, Callable]] = None,
    aa_state_path: Optional[Path] = None,
    args: Optional[argparse.Namespace] = None,

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
    model = model.cuda()

    prepr = get_preprocessing(dataset_, threat_model_, model_name,
                            preprocessing)

    # clean_x_test, clean_y_test = load_clean_dataset(dataset_, n_examples,
    #                                                 args.data_dir, prepr)
    _, images_test_np, _, labels_test_np, _ = get_dataset(args.data_dir, classes=args.classes, batch_size=args.bs)
    
    images_test_np = images_test_np[:n_examples]
    labels_test_np = labels_test_np[:n_examples]
    # transpose the images to (N, C, H, W), then to torch tensor
    images_test_np = np.transpose(images_test_np, (0, 3, 1, 2))
    clean_x_test = torch.tensor(images_test_np, dtype=torch.float32)
    clean_y_test = torch.tensor(labels_test_np, dtype=torch.long)
    
    
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
    # save the adversarial examples
    # import pdb; pdb.set_trace()
    np.save(f"aa_adv_images_{args.dataset}.npy", x_adv.cpu().detach().numpy())
    # import pdb; pdb.set_trace()
    x_adv = np.load(f"aa_adv_images_{args.dataset}.npy")
    x_adv = torch.tensor(x_adv, dtype=torch.float32)
    adv_accuracy = clean_accuracy(
                                model,
                                x_adv,
                                clean_y_test,
                                batch_size=batch_size,
                                device=device)
    print(f'With {len(clean_x_test)} clean examples, robust accuracy: {adv_accuracy:.2%}')



def _pgd_targeted_attack(
                model,
                X,
                target_y,  # Target label for the adversarial attack
                epsilon=20/255,  # 0.031
                num_steps=20,
                step_size=1,
                random=True,
                ):
    # Start by making predictions for clean images
    initial_out = model(X)
    initial_err = (initial_out.data.max(1)[1] != target_y.data).float().sum()
    target_y = torch.zeros_like(target_y)
    # import pdb; pdb.set_trace()
    # Initialize the adversarial example with the input data
    X_pgd = Variable(X.data, requires_grad=True)
    
    # Add random noise within [-epsilon, epsilon] if specified
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    # Run PGD for the specified number of steps
    for _ in range(num_steps):
        # Set up the optimizer
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        # Calculate loss for targeted attack (maximize probability of target class)
        with torch.enable_grad():
            loss = -nn.CrossEntropyLoss()(model.forward_original(X_pgd), target_y)
            # loss = -nn.CrossEntropyLoss()(model(X_pgd), target_y)
        loss.backward()
        
        # Compute the perturbation and update X_pgd
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        # Project the perturbation to maintain it within epsilon bounds
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)

        # Ensure the adversarial image remains in the valid range [0, 1]
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    # Calculate the error on the targeted adversarial examples
    adv_out = model(X_pgd)
    # import pdb; pdb.set_trace()
    adv_err = (adv_out.data.max(1)[1] != target_y.data).float().sum()
    print("predicition:", adv_out.data.max(1)[1], "target:", target_y.data)
    return initial_err, adv_err, X_pgd



def _pgd_adaptive_attack(
                model_target,
                model_source,
                X,
                y,
                epsilon=0.031,   # 0.031
                num_steps=20,
                step_size=0.003,  # 0.003
                random=True,
                num_eot=1,) -> Tuple[int, int]:
    model_source.eval()
    model_target.eval()
    layer = 50
    #! make sure that for the target model, all defenses are turned on    
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
            loss = 0
            for i in range(num_eot):
                x_transformed = X_pgd
                loss += nn.CrossEntropyLoss()(model_source.get_logits_from_several_layers(x_transformed, layer), y)
                # loss += nn.CrossEntropyLoss()(model_source.forward_original(x_transformed), y)
            loss /= num_eot
                
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        
    err_pgd_on_source = (model_source.get_logits_from_several_layers(X_pgd, layer).data.max(1)[1] != y.data).float().sum()
    # err_pgd_on_source = (model_source.forward_original(X_pgd).data.max(1)[1] != y.data).float().sum()

    err_pgd_on_target = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd_on_source, err_pgd_on_target, X_pgd


def _pgd_attack(
                model,
                X,
                y,
                epsilon=0.031,
                num_steps=20,
                step_size=0.003, # 0.003
                random=True,
                num_eot=1,
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
        loss = 0
        for i in range(num_eot):
            x_transformed = X_pgd
            loss += nn.CrossEntropyLoss()(model(x_transformed), y)
        # with torch.enable_grad():
        #     loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    
    return err, err_pgd, X_pgd



def non_adaptive_attack(model, args, targetd_attack=False):
    """
    evaluate model by non-adaptive attack
    """
    test_loader = get_dataset(args.data_dir, classes=args.classes, batch_size=args.bs)[-1]
    model.eval()
    
    robust_err_total = 0
    natural_err_total = 0
    total = 0
    
    pgd_steps = args.steps
    
    pbar = tqdm(test_loader, ncols=100)
    for data, target in pbar:
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data, requires_grad=True), Variable(target)
        if targetd_attack:
            err_natural, err_robust, X_pgd = _pgd_targeted_attack(model, X, y, num_steps=pgd_steps)
        else:
            err_natural, err_robust, X_pgd = _pgd_attack(model, X, y, num_steps=pgd_steps)
        robust_err_total += err_robust
        natural_err_total += err_natural
        total += data.size(0)
        pbar.set_description(f'#steps:{pgd_steps}, #bs:{args.bs} | '
                            f'Nat Err: {natural_err_total} | '
                            f'Rob Err: {robust_err_total} | '
                            f'Total: {total} \n' )
        
        if total >= args.num_test:
            break
    # visualize the last batch of images
    # visualize_adv_images(X, y, X_pgd, model, description="non-adpt")
    
    print(f"clean accuracy: {(1 - natural_err_total / total):.2%}, robust accuracy: {(1 - robust_err_total / total):.2%}")

def adaptive_attack_old(model_target, args):
    """
    evaluate model by our adaptive attack
    """
    _, images_test_np, _, labels_test_np, test_loader = get_dataset(args.data_dir, classes=args.classes, batch_size=args.bs)

    model_source = SourceModel(copy.deepcopy(model_target.imported_model), model_target.multichannel_fn, classes=args.classes).to("cuda")
    model_source.layer_operations = copy.deepcopy(model_target.layer_operations)
    model_source.linear_layers = copy.deepcopy(model_target.linear_layers)
    

        
    model_target.eval()
    model_source.eval()
    
    # note: source model should have a similar test accuracy as the target model
    if False:
        test_hits,test_count,_ = eval_model(model_source, images_test_np, labels_test_np, forward_fn="ensemble")
        print(f"source model test={test_hits}/{test_count}={test_hits/test_count}")
        test_hits,test_count,_ = eval_model(model_target, images_test_np, labels_test_np, forward_fn="ensemble")
        print(f"target model test={test_hits}/{test_count}={test_hits/test_count}")
    
    robust_err_total_source = 0
    robust_err_total_target = 0
    natural_err_total = 0
    total = 0
    num_eot = args.eot
    pgd_steps = args.steps
    pbar = tqdm(test_loader, ncols=150)
    # description = "fix" if only_attack_cross_max else "non-fix" 
    description = args.dataset
    saved_adv_images = []
    for data, target in pbar:
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data, requires_grad=True), Variable(target)
        #! we do transfer attack on the source model
        err_natural, err_robust_on_source, err_robust_on_target, X_pgd = _pgd_adaptive_attack(model_target, model_source, X, y, num_eot=num_eot, num_steps=pgd_steps)
        robust_err_total_source += err_robust_on_source
        robust_err_total_target += err_robust_on_target
        natural_err_total += err_natural
        total += data.size(0)
        pbar.set_description(f'{description}, #steps:{pgd_steps}, #eot:{num_eot}, #bs:{args.bs} | #Err:'
                            f'Nat: {natural_err_total} | '
                            f'Rob_S: {robust_err_total_source} | '
                            f'Rob_T: {robust_err_total_target} | '
                            f'Total: {total} \n' )
        saved_adv_images.append(X_pgd.cpu().detach().numpy())
        if total >= args.num_test:
            break
    saved_adv_images_np = np.concatenate(saved_adv_images, axis=0)
    np.save(f"saved_adv_images_{args.dataset}.npy", saved_adv_images_np)    
    # visualize the last batch of images
    # visualize_adv_images(data, target, X_pgd, model_target, description="adpt")
    
    print(f"clean accuracy: {(1 - natural_err_total / total):.2%}, robust accuracy on source: {(1 - robust_err_total_source / total):.2%}, robust accuracy on target: {(1 - robust_err_total_target / total):.2%}")

    # double check, load the saved images, then evaluate the model
    saved_adv_images_np = np.load(f"saved_adv_images_{args.dataset}.npy")
    saved_adv_images = np.transpose(saved_adv_images_np, (0, 2, 3, 1))
    # evaluate the model for 10 times, then report the average accuracy and std
    mean_acc = []
    for i in range(10):
        test_hits, test_count, _ = eval_model(model_target, saved_adv_images, labels_test_np[:saved_adv_images.shape[0]], forward_fn="ensemble")
        acc = test_hits/test_count
        mean_acc.append(acc)
    print(f"robust accuracy on target: {np.mean(mean_acc):.2%}+-{np.std(mean_acc):.2%}")
    
    
def adaptive_attack(model_target, args):
    """
    evaluate model by our adaptive attack
    """
    _, images_test_np, _, labels_test_np, test_loader = get_dataset(args.data_dir, classes=args.classes, batch_size=args.bs)

    model_source = SourceModel(copy.deepcopy(model_target.imported_model), model_target.multichannel_fn, classes=args.classes).to("cuda")
    model_source.layer_operations = copy.deepcopy(model_target.layer_operations)
    model_source.linear_layers = copy.deepcopy(model_target.linear_layers)
    

        
    model_target.eval()
    model_source.eval()
    
    # note: source model should have a similar test accuracy as the target model
    if False:
        test_hits,test_count,_ = eval_model(model_source, images_test_np, labels_test_np, forward_fn="ensemble")
        print(f"source model test={test_hits}/{test_count}={test_hits/test_count}")
        test_hits,test_count,_ = eval_model(model_target, images_test_np, labels_test_np, forward_fn="ensemble")
        print(f"target model test={test_hits}/{test_count}={test_hits/test_count}")
    
    robust_err_total_source = 0
    robust_err_total_target = 0
    natural_err_total = 0
    total = 0
    num_eot = args.eot
    pgd_steps = args.steps
    pbar = tqdm(test_loader, ncols=150)
    # description = "fix" if only_attack_cross_max else "non-fix" 
    description = args.dataset
    saved_adv_images = []
    true_labels = []
    for data, target in pbar:
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data, requires_grad=True), Variable(target)
        #! we do transfer attack on the source model
        err_natural, err_robust_on_source, err_robust_on_target, X_pgd = _pgd_adaptive_attack(model_target, model_source, X, y, num_eot=num_eot, num_steps=pgd_steps)
        robust_err_total_source += err_robust_on_source
        robust_err_total_target += err_robust_on_target
        natural_err_total += err_natural
        total += data.size(0)
        pbar.set_description(f'{description}, #steps:{pgd_steps}, #eot:{num_eot}, #bs:{args.bs} | #Err:'
                            f'Nat: {natural_err_total} | '
                            f'Rob_S: {robust_err_total_source} | '
                            f'Rob_T: {robust_err_total_target} | '
                            f'Total: {total} \n' )
        saved_adv_images.append(X_pgd.cpu().detach().numpy())
        true_labels.append(y.cpu().detach().numpy())
        if total >= args.num_test:
            break
        

    saved_adv_images_np = np.concatenate(saved_adv_images, axis=0)
    saved_adv_images_np_cp = copy.deepcopy(saved_adv_images_np)
    true_labels = np.concatenate(true_labels, axis=0)
    true_labels_cp = copy.deepcopy(true_labels)
    #! we pick the samples that are correctly classified by the target model, then do non-adaptive attack
    # calculate the robust accuracy of the target model 5 times, pick samples that are correctly classified
    picked_indices = []
    for i in range(5):
        predict = eval_model(model_target, np.transpose(saved_adv_images_np, (0, 2, 3, 1)), true_labels, forward_fn="ensemble", return_pred=True)
        picked_indices.append(np.where(predict == true_labels)[0])
    picked_indices = np.concatenate(picked_indices, axis=0)
    indices = np.unique(picked_indices)
    saved_adv_images_np = saved_adv_images_np[indices]
    true_labels = true_labels[indices]
    
    saved_adv_images = torch.tensor(saved_adv_images_np, dtype=torch.float32).cuda()
    y = torch.tensor(true_labels, dtype=torch.long).cuda()
    # set batch size to 16, then do non-adaptive attack
    bs_ = 4
    for i in range(0, len(saved_adv_images), bs_):
        X = saved_adv_images[i:i+bs_]
        y_batch = y[i:i+bs_]
        err_natural, err_robust, X_pgd = _pgd_attack(model_target, X, y_batch, num_steps=pgd_steps, num_eot=4)
        saved_adv_images_np_cp[indices[i:i+bs_]] = X_pgd.cpu().detach().numpy()
        print(f"Nat Err: {err_natural} | Rob Err: {err_robust} | Total: {bs_} \n")

    # evaluate the model for 10 times, then report the average accuracy and std
    mean_acc = []
    for i in range(10):
        test_hits, test_count, _ = eval_model(model_target, np.transpose(saved_adv_images_np_cp, (0, 2, 3, 1)), true_labels_cp, forward_fn="ensemble")
        acc = test_hits/test_count
        mean_acc.append(acc)
    print(f"robust accuracy on target: {np.mean(mean_acc):.2%}+-{np.std(mean_acc):.2%}")
    
