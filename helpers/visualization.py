from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
import torch
from tqdm import tqdm

from helpers.utils import cifar100_class_to_description, cifar10_class_to_description

def random_class_without_list(excluded_lists, classes=100):
	available_classes = [[c for c in range(classes) if c not in excluded_lists[i]] for i in range(len(excluded_lists))]
	return [np.random.choice(cs) for cs in available_classes]

def hard_to_soft_targets(hard_targets, num_classes=100):
	target_labels_hard = torch.Tensor(hard_targets).to(torch.long)
	target_labels_soft = torch.nn.functional.one_hot(target_labels_hard, num_classes=num_classes).to(torch.float).to("cuda")
	return target_labels_soft.detach().cpu().numpy()


def get_complex_specification_adversaries(
	attack_specifications,
	batch_size=32,
	lr=1e-1,
	optimizer_in=optim.SGD,
	attack_resolution=32,
	clip_images=True,
	steps=100,
	attack_Linfty_limit=None,
	stop_at_loss=None,
	verbose=True,
	flip_loss_sign=False,
):
	collect_images = []
	# import pdb; pdb.set_trace()
	attack_specifications_prepared = []
	for (model, images, perturbation_ids, soft_targets) in attack_specifications:
		attack_specifications_prepared.append(
			(model.to("cuda"), torch.Tensor(images).to("cuda"), perturbation_ids, torch.Tensor(soft_targets).to("cuda"))
		)

	# count how many perturbations are needed
	perturbation_count = max([max(specification[2]) for specification in attack_specifications]) + 1

	# perturbations
	image_perturbations = torch.Tensor(np.zeros((perturbation_count, 3, attack_resolution, attack_resolution))).to("cuda")
	image_perturbations.requires_grad = True

	optimizer = optimizer_in([image_perturbations], lr=lr)

	if verbose:
		steps_bar = tqdm(range(steps), desc='Adversary progress', ncols=100)
	else:
		steps_bar = range(steps)

	for step in steps_bar:
		losses = []

		for (model, images, perturbation_ids, soft_targets) in attack_specifications_prepared:
			perturbations_to_use = image_perturbations[perturbation_ids]

			if attack_Linfty_limit is None:
				attacked_images = images + perturbations_to_use
			else:
				attacked_images = images + attack_Linfty_limit * torch.tanh(perturbations_to_use)

			if clip_images:
				attacked_images = torch.clip(attacked_images, 0, 1)

			# batching for the model
			batched_losses = []
			iterations = int(np.ceil(attacked_images.shape[0] / batch_size))
			for it in range(iterations):
				i1 = it * batch_size
				i2 = min([(it + 1) * batch_size, attacked_images.shape[0]])
				# import pdb; pdb.set_trace()
				logits = model.forward_original(attacked_images[i1:i2])
				# logits = model(attacked_images[i1:i2])  # much worse images
				loss = torch.nn.functional.cross_entropy(logits, soft_targets[i1:i2], reduction="none")
				batched_losses.append(loss)

			if flip_loss_sign is False:
				torch.mean(torch.concatenate(batched_losses, axis=0)).backward()
			else:
				torch.mean(-torch.concatenate(batched_losses, axis=0)).backward()

			losses.append(torch.concatenate(batched_losses, axis=0).detach().cpu().numpy())

		overall_loss = np.mean(np.stack(losses))

		if verbose:
			steps_bar.set_description(f"loss = {overall_loss}")
		
		if stop_at_loss is not None and ((overall_loss <= stop_at_loss and flip_loss_sign is False) or (overall_loss >= stop_at_loss and flip_loss_sign is True)):
			# import pdb; pdb.set_trace()
			# getting the resulting images
			if attack_Linfty_limit is None:
				return image_perturbations.detach().cpu().numpy()
			else:
				return attack_Linfty_limit * torch.tanh(image_perturbations).detach().cpu().numpy()

		optimizer.step()
		optimizer.zero_grad()

		collect_images.append(attacked_images)

	# getting the resulting images
	if attack_Linfty_limit is None:
		return image_perturbations.detach().cpu().numpy(), collect_images
	else:
		return attack_Linfty_limit * torch.tanh(image_perturbations).detach().cpu().numpy(), collect_images


def visualize_adv_imgs_from_random(model_to_use, images_test_np, labels_test_np, args):
	per_class_reps = 5
	classes_at_once = 4
	all_perturbed_images = []
	num_classes = np.max(labels_test_np) + 1
	desc_func = cifar100_class_to_description if num_classes == 100 else cifar10_class_to_description
	for i in range(1):
		target_labels = []
		for c in range(i * classes_at_once, (i + 1) * classes_at_once):
			target_labels += [c] * per_class_reps
		# classes=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
		print(f"---- classes={target_labels} ----")

		size = 128
		j = 0

		count = 20
		batch_size = 10
		images_selected = images_test_np[j * count:(j + 1) * count]
		images_selected = np.ones(images_selected.shape) * 0.5  

		model_to_use = model_to_use.eval()

		attack_specifications = [
			(
				model_to_use.eval(),
				images_selected.transpose([0, 3, 1, 2]),  # all of the train set
				range(len(images_selected)),  # different perturbation id for each image = an attack per image
				hard_to_soft_targets(target_labels, num_classes=num_classes),  # randomly selected target labels exluding the original label of the image
			),  # a single model to attack with its specification
		]

		perturbations, _ = get_complex_specification_adversaries(
			attack_specifications,
			steps=400,
			batch_size=batch_size,  # if you have cude problems, you can decrease this => less memory but slower
			attack_Linfty_limit=size / 255,
			lr=10e-1,  # 2e-1,#*6,#1e0 if layer < 5 else 1e-1,
			# stop_at_loss=1e-2,
		)

		perturbed_images = np.clip(images_selected + perturbations.transpose([0, 2, 3, 1]), 0, 1)
		all_perturbed_images.append(perturbed_images)
	# import pdb; pdb.set_trace()
	# all_perturbed_images: 2 x (20, 32, 32, 3)
	all_composite_images = []
	for i in range(1):
		for j in range(4):
			cut = all_perturbed_images[i][j * 5:j * 5 + 4]  # (4, 32, 32, 3)
			composite_image = np.zeros((2 * 32, 2 * 32, 3))  # (64, 64, 3)
			composite_image[:32, :32, :] = cut[0]   
			composite_image[:32, 32:64, :] = cut[1]
			composite_image[32:64, :32, :] = cut[2]
			composite_image[32:64, 32:64, :] = cut[3]

			composite_image[:, 32, :] = 1
			composite_image[32, :, :] = 1

			all_composite_images.append(composite_image)

	# all_composite_images: 8 x (64, 64, 3)
	plt.figure(figsize=(4 * 9, 4 * 12), dpi=125)
	for i in range(4):
		plt.subplot(4, 1, i + 1)
		plt.title(f"c={i} {desc_func(i)}", fontsize=24)
		plt.imshow(all_composite_images[i])  # (64, 64, 3)
		plt.xticks([], [])
		plt.yticks([], [])
	plt.tight_layout()
	plt.savefig(f"{args.img_path}/composite_images_{args.dataset}.png")



def visualize_adv_imgs_from_real(model_to_use, images_test_np, labels_test_np, args):
	per_class_reps = 5
	classes_at_once = 4
	all_perturbed_images = []
	num_classes = np.max(labels_test_np) + 1
	desc_func = cifar100_class_to_description if num_classes == 100 else cifar10_class_to_description

	for i in range(1):
		target_labels = []
		for c in range(i * classes_at_once, (i + 1) * classes_at_once):
			target_labels += [c] * per_class_reps

		print(f"---- classes={target_labels} ----")

		size = 128
		j = 0

		count = 20
		batch_size = 10

		images_selected = images_test_np[j * count:(j + 1) * count]
		labels_selected = labels_test_np[j * count:(j + 1) * count]
		# plot all images in images_selected and corresponding labels
		plt.figure(figsize=(20, 20))
		for i in range(len(images_selected)):
			plt.subplot(4, 5, i + 1)
			plt.imshow(images_selected[i])
			plt.title(f"c={i} {desc_func(labels_selected[i])}")
			# plt.show()
		plt.savefig(f"{args.img_path}/images_selected_{args.dataset}.png")

		model_to_use = model_to_use.eval()

		attack_specifications = [
			(
				model_to_use.eval(),
				images_selected.transpose([0, 3, 1, 2]),  # all of the train set
				range(len(images_selected)),  # different perturbation id for each image = an attack per image
				hard_to_soft_targets(target_labels, num_classes=num_classes),  # randomly selected target labels exluding the original label of the image
			),  # a single model to attack with its specification
		]

		perturbations, _ = get_complex_specification_adversaries(
			attack_specifications,
			steps=400,
			batch_size=batch_size,  # if you have cude problems, you can decrease this => less memory but slower
			attack_Linfty_limit=size / 255,
			lr=10e-1,  # 2e-1,#*6,#1e0 if layer < 5 else 1e-1,
			# stop_at_loss=1e-2,
		)

		perturbed_images = np.clip(images_selected + perturbations.transpose([0, 2, 3, 1]), 0, 1)
		all_perturbed_images.append(perturbed_images)
	
	all_composite_images = []
	for i in range(1):
		for j in range(4):
			cut = all_perturbed_images[i][j * 5:j * 5 + 4]  # (4, 32, 32, 3)
			composite_image = np.zeros((2 * 32, 2 * 32, 3))  # (64, 64, 3)
			composite_image[:32, :32, :] = cut[0]   
			composite_image[:32, 32:64, :] = cut[1]
			composite_image[32:64, :32, :] = cut[2]
			composite_image[32:64, 32:64, :] = cut[3]

			composite_image[:, 32, :] = 1
			composite_image[32, :, :] = 1

			all_composite_images.append(composite_image)

	
	plt.figure(figsize=(4 * 9, 4 * 12), dpi=125)
	for i in range(4):
		plt.subplot(4, 1, i + 1)
		plt.title(f"c={i} {desc_func(i)}", fontsize=24)
		plt.imshow(all_composite_images[i])  # (64, 64, 3)
		plt.xticks([], [])
		plt.yticks([], [])
	plt.tight_layout()
	plt.savefig(f"{args.img_path}/adv_real_{args.dataset}.png")
