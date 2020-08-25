import logging

import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import fmin as scipy_optimizer
from scipy.stats import weibull_min


def save_grad(save):
    def hook(grad):
        # note: clear gradient after saving it each time
        save['grads'] = grad.detach().clone()
        grad.data.zero_()
    return hook


def clever(model, x, gt, nb_batches, batch_size, radius, c_init=1.0, targeted=None):
    """
    Compute CLEVER score for an untargeted attack.
    | Paper link: https://arxiv.org/abs/1801.10578

    # note compute clever on correctly classified images
    # note this only support l2 norm (can be extended to linf if needed)
    # note this assumes the the range of values of x are [0.0, 1.0]
    
    :param model: pytorch mode
    :param x: tensor of shape batch_size x c x h x w (image batch to compute clever scores)
    :param gt: ground truth for the x shape batch_size x num_classes
    :param nb_batches: number of batches to run the estimation of max gradients
    :param batch_size: number of samples per batch used for estimation of max gradients
    :param radius: radius of the max l2 pertubation
    :param c_init: init for  Weibull distribution
    :param targeted: np.array of targets for each image shape batch_size x target_classes
                     'top-2': compute clever on top-2 class
    :return: list of clever score for each image in x
    """
    _, c, h, w = x.shape
    dims = c * h * w
    device = x.device
    
    model.eval()
    y = model(x)
    pred_class = torch.argmax(y, dim=1).cpu().numpy()
    x = x.cpu()
    gt = gt.cpu()
    # skip computing clever scores for samples misclassified
    skip_x_idx = set([idx for idx in range(x.shape[0]) if pred_class[idx] != gt[idx]])
    
    clever_scores = list()
    input_grads = {'grads': None}

    # targeted classes for clever computation
    if targeted is None:
        targeted_classes = torch.topk(y, k=y.shape[1], dim=1).indices[:, 1:].cpu().numpy()
    elif isinstance(targeted, str) and targeted == 'top-2':
        targeted_classes = torch.topk(y, k=2, dim=1).indices[:, 1].unsqueeze(1).cpu().numpy()
    elif targeted.shape[0] == y.shape[0] and len(targeted.shape) == 2:
        targeted_classes =  targeted
    else:
        raise RuntimeError('[CLEVER] unsupported shape for targets')
    
    for idx, img in tqdm(enumerate(x), desc="CLEVER"):
        if idx in skip_x_idx:
            logging.warning("misclassified sample skipping clever computation for idx: %d" % idx)
            continue
        img_pred_class = pred_class[idx]
        targeted_classes_img = targeted_classes[idx]
        untargeted_clever_scores_img = list()
        img = torch.unsqueeze(img, 0).to(device)
        max_grad_norms = np.zeros((y.shape[1], nb_batches))

        for bidx in range(nb_batches):
            # sample pertubation from uniform l2 ball and add to image
            rand_pert = torch.randn(batch_size, (dims + 1) + 1, device=device)
            r_pert = torch.norm(rand_pert, 2, dim=1, keepdim=True)
            random_pert_images = torch.clamp(((rand_pert / r_pert)[:, :dims] * radius).view(batch_size, c, h, w) + img, 0.0, 1.0)
            random_pert_images.requires_grad = True
            
            preds = model(random_pert_images)
            model.zero_grad()
            hook = random_pert_images.register_hook(save_grad(input_grads))
            for target_class in targeted_classes_img:
                torch.autograd.backward(preds[:, img_pred_class] - preds[:, target_class],
                                        torch.tensor([1.0] * len(preds[:, 0])).to(next(model.parameters()).device),
                                        retain_graph=True)
                max_grad_norm = torch.max(
                    torch.norm(
                        (input_grads['grads']).view(batch_size, -1), p=2, dim=1)
                )
                max_grad_norms[target_class][bidx] = max_grad_norm.item()
            
        # Maximum likelihood estimation for max gradient norms
        for target_class in targeted_classes_img:
            [_, loc, _] = weibull_min.fit(-max_grad_norms[target_class], c_init, optimizer=scipy_optimizer)
            value = (y[idx][img_pred_class] - y[idx][target_class]).item()
            untargeted_clever_scores_img.append(min([-value/loc, radius]))
            logging.info("clever for img:%d target_class: %d = %f" % (idx, target_class, untargeted_clever_scores_img[-1]))

        clever_scores.append(min(untargeted_clever_scores_img))

    x = x.to(device)
    gt = gt.to(device)
    return clever_scores
