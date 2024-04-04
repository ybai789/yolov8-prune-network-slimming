import re
import os
import sys
from pathlib import Path

import yaml
import argparse

import torch
import torch.nn as nn
from ultralytics.utils import colorstr
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules import Conv, Concat

from ultralytics.nn.modules.block_pruned import C2fPruned, SPPFPruned
from ultralytics.nn.modules.head_pruned import DetectPruned
from ultralytics.nn.tasks_pruned import DetectionModelPruned

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv8 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def prepare_prune_data(model):
    """
    Prepare data for determining the pruning threshold.
    
    Args:
        model: The original model.
        
    Returns:
        bn_dict: Dictionary storing all bn layers, key is the bn layer name, value is the corresponding bn layer.
        ignore_bn_list: List of bn layer names to be ignored for pruning.
        chunk_bn_list: List of bn layer names that need to ensure the number of channels is even.
        sorted_bn: Tensor sorted by the absolute value of gamma.
    """
    bn_dict = {}  # Storing all bn layers
    ignore_bn_list = []  # Storing bn layers to be ignored for pruning
    chunk_bn_list = []  # Storing bn layers that need to ensure the number of channels is even
    for name, module in model.model.named_modules():
        # For Bottleneck modules with residual connections, the number of channels of the two branches to be added must be consistent, and cannot be pruned.
        if isinstance(module, Bottleneck):
            if module.add:  
                ignore_bn_list.append(f"{name[:-4]}.cv1.bn")
                ignore_bn_list.append(f"{name}.cv2.bn")
            else:
                # For the C2f module without the add operation, since there is a chunk operation during forward, the number of channels before chunk must be even.
                chunk_bn_list.append(f"{name[:-4]}.cv1.bn")
        if isinstance(module, nn.BatchNorm2d):
            bn_dict[name] = module

    # Check for naming errors in the bn layers to be ignored.      
    for ignore_bn_name in ignore_bn_list:
        assert ignore_bn_name in bn_dict.keys()

    # Filter the stored bn layers dictionary according to the ignore list.  
    bn_dict = {k: v for k, v in bn_dict.items() if k not in ignore_bn_list}

    # Gather the absolute values of gamma of the bn layers to be pruned.
    bn_weights = []
    for name, module in bn_dict.items():
        bn_weights.extend(module.weight.data.abs().clone().cpu().tolist())
        
    # Sort the absolute values of gamma by magnitude.
    sorted_bn = torch.sort(torch.tensor(bn_weights))[0]
    
    return bn_dict, ignore_bn_list, chunk_bn_list, sorted_bn

def compute_prune_threshold(bn_dict, sorted_bn, prune_ratio):
    """
    Compute the pruning threshold.
    
    Args:
        bn_dict: Dictionary storing all bn layers, key is the bn layer name, value is the corresponding bn layer.
        sorted_bn: Tensor sorted by the absolute value of gamma.
        prune_ratio: The pruning ratio specified by the user.
        
    Returns:
        highest_thre: The maximum threshold for pruning, exceeding this value may cause an entire layer to be pruned away.
        percent_limit: The maximum percentage for pruning.
        thre: The pruning threshold under the specified ratio by the user.
    """
    highest_thre = [] 
    for name, module in bn_dict.items():
        # Calculate the maximum absolute value of gamma of each bn layer.
        highest_thre.append(module.weight.data.abs().clone().cpu().max())
    # The minimum of all maximum values is the maximum threshold for pruning, exceeding this value may cause an entire layer to be pruned away.
    highest_thre = min(highest_thre) 
    # Calculate the maximum ratio of pruning.
    percent_limit = (sorted_bn == highest_thre).nonzero()[0, 0].item() / len(sorted_bn)
    # Calculate the pruning threshold under the specified ratio by the user.
    thre = sorted_bn[int(len(sorted_bn) * prune_ratio)]
    print(f'Suggested Gamma threshold should be less than {colorstr(f"{highest_thre:.4f}")}, yours is {colorstr(f"{thre:.4f}")}')
    print(f'The corresponding prune ratio should be less than {colorstr(f"{percent_limit:.3f}")}, yours is {colorstr(f"{prune_ratio:.3f}")}')
    return highest_thre, percent_limit, thre

def rewrite_model_config(model, cfg, model_size):
    """
    Rewrite the model configuration file as a dictionary (rewrote the C2f, SPPF, and Detect modules).
    
    Args:
        model: The original model.
        cfg: The path to the original model configuration file.
        model_size: The size of the model (n/s/m/l/x).
        
    Returns:
        pruned_yaml: The rewritten model configuration dictionary.
    """
    pruned_yaml = {}
    with open(cfg, encoding='ascii', errors='ignore') as f:
        model_yamls = yaml.safe_load(f)  # model dict
        
    nc = model.model.nc
    pruned_yaml["nc"] = nc
    pruned_yaml["scales"] = model_yamls["scales"] 
    pruned_yaml["scale"] = model_size
    
    # Define the pruned backbone
    pruned_yaml["backbone"] = [
        [-1, 1, Conv, [64, 3, 2]],  # 0-P1/2
        [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4  
        [-1, 3, C2fPruned, [128, True]],
        [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
        [-1, 6, C2fPruned, [256, True]],  
        [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
        [-1, 6, C2fPruned, [512, True]], 
        [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
        [-1, 3, C2fPruned, [1024, True]], 
        [-1, 1, SPPFPruned, [1024, 5]],  # 9
    ]
    
    # Define the pruned head
    pruned_yaml["head"] = [
        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        [-1, 3, C2fPruned, [512]],  # 12
        
        [-1, 1, nn.Upsample, [None, 2, 'nearest']], 
        [[-1, 4], 1, Concat, [1]],  # cat backbone P3
        [-1, 3, C2fPruned, [256]],  # 15 (P3/8-small)
        
        [-1, 1, Conv, [256, 3, 2]], 
        [[-1, 12], 1, Concat, [1]],  # cat head P4
        [-1, 3, C2fPruned, [512]],  # 18 (P4/16-medium)
        
        [-1, 1, Conv, [512, 3, 2]],
        [[-1, 9], 1, Concat, [1]],  # cat head P5
        [-1, 3, C2fPruned, [1024]],  # 21 (P5/32-large)
        
        [[15, 18, 21], 1, DetectPruned, [nc]], # Detect(P3, P4, P5)  
    ]
    
    return pruned_yaml

def compute_bn_mask(model, ignore_bn_list, chunk_bn_list, thre):
    """
    Calculate the mask for bn layers, multiply gamma and beta by the mask, and count the changes in the number of channels before and after pruning.
    
    Args:
        model: The original model.
        ignore_bn_list: List of bn layer names to be ignored for pruning.
        chunk_bn_list: List of bn layer names that need to ensure the number of channels is even.
        thre: The pruning threshold.
        
    Returns:
        maskbndict: Dictionary storing the mask of bn layers, key is the bn layer name, value is the corresponding mask.
    """
    maskbndict = {}
    print("=" * 80)
    print(f"|\t{'layer name':<25}{'|':<10}{'origin channels':<20}{'|':<10}{'remaining channels':<20}|")
    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            origin_channels = module.weight.data.size()[0]
            remaining_channels = origin_channels 
            mask = torch.ones(origin_channels)
            if name not in ignore_bn_list:
                mask = module.weight.data.abs().gt(thre).float()
                # If the number of remaining channels is odd after pruning and it's a C2f structure, adjust the remaining channels to be even.
                if name in chunk_bn_list and mask.sum() % 2 == 1:  
                    flattened_sorted_weight = torch.sort(module.weight.data.abs().view(-1))[0]
                    idx = torch.min(torch.nonzero(flattened_sorted_weight.gt(thre))).item()
                    thre_ = flattened_sorted_weight[idx - 1] - 1e-6
                    mask = module.weight.data.abs().gt(thre_).float()
                assert mask.sum() > 0, f"bn {name} has no active elements"  
                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
                remaining_channels = mask.sum().int()    
            maskbndict[name] = mask
            print(f"|\t{name:<25}{'|':<10}{origin_channels:<20}{'|':<10}{remaining_channels:<20}|")
    print("=" * 80)
    return maskbndict

def assign_pruned_params(model, pruned_model, maskbndict):
    """
    Assign the parameters of the channels retained in the original model to the pruned model.
    
    Args:
        model: The original model.
        pruned_model: The pruned model.
        maskbndict: Dictionary storing the mask of bn layers, key is the bn layer name, value is the corresponding mask.
        
    Returns:
        pruned_model: The pruned model with assigned parameters.
    """
    current_to_prev = pruned_model.current_to_prev
    # Verify that both the keys and values in current_to_prev are in maskbndict.
    for xks, xvs in current_to_prev.items(): 
        xvs = [xvs] if not isinstance(xvs, list) else xvs
        for xk, xv in zip(xks, xvs):
            assert all([xk, xv in maskbndict.keys()]), f"{xk, xv} from 'current_to_prev' not in maskbndict" 

    changed = []
    # Match the first convolution layer in the C2f module Bottleneck.
    pattern_c2f = re.compile(r"model.\d+.m.0.cv1.bn")  
    # Match the last convolution layer in the Detect module (without BN).
    pattern_detect = re.compile(r"model.\d+.cv\d.\d.2") 
    for (name_org, module_org), (name_pruned, module_pruned) in \
        zip(model.model.named_modules(), pruned_model.named_modules()): 

        assert name_org == name_pruned

        # If it's a dfl layer, it means the process is finished.
        if 'dfl' in name_org:
            break

        # Handle the last convolution layer in the Detect module (without BN).
        if pattern_detect.fullmatch(name_org) is not None:
            current_conv_layer_name = name_org
            prev_bn_layer_name = current_to_prev[current_conv_layer_name] 
            in_channels_mask = maskbndict[prev_bn_layer_name].to(torch.bool)
            module_pruned.weight.data = module_org.weight.data[:, in_channels_mask, :, :]
            if module_org.bias is not None:
                assert module_pruned.bias.data is not None
                module_pruned.bias.data = module_org.bias.data
            continue

        # Handle ordinary convolution layers    
        if isinstance(module_org, nn.Conv2d):
            currnet_bn_layer_name = name_org[:-4] + 'bn'
            out_channels_mask = maskbndict[currnet_bn_layer_name].to(torch.bool)
            prev_bn_layer_name = current_to_prev.get(currnet_bn_layer_name, None)
            if isinstance(prev_bn_layer_name, list):
                in_channels_masks = [maskbndict[ni] for ni in prev_bn_layer_name] 
                in_channels_mask = torch.cat(in_channels_masks, dim=0).to(torch.bool)
            elif prev_bn_layer_name is not None:
                in_channels_mask = maskbndict[prev_bn_layer_name].to(torch.bool)
                # The first convolution in the C2f structure Bottleneck.
                if pattern_c2f.fullmatch(currnet_bn_layer_name) is not None:
                    in_channels_mask = in_channels_mask.chunk(2, 0)[1]
                # The second convolution in the SPPF structure.    
                if name_org == "model.9.cv2.conv":
                    in_channels_mask = torch.cat([in_channels_mask for _ in range(4)], dim=0)  
            else:
                in_channels_mask = torch.ones(module_org.weight.data.shape[1], dtype=torch.bool)

            state_dict_org = module_org.weight.data[out_channels_mask, :, :, :]
            state_dict_org = state_dict_org[:, in_channels_mask, :, :]
            module_pruned.weight.data = state_dict_org

            assert module_pruned.in_channels == state_dict_org.shape[1]  
            assert module_pruned.out_channels == state_dict_org.shape[0]

            if module_org.bias is not None: 
                assert module_pruned.bias.data is not None
                module_pruned.bias.data = module_org.bias.data[out_channels_mask]
            changed.append(currnet_bn_layer_name)

        # Handle BN layers    
        if isinstance(module_org, nn.BatchNorm2d):
            out_channels_mask = maskbndict[name_org].to(torch.bool)
            module_pruned.weight.data = module_org.weight.data[out_channels_mask]
            module_pruned.bias.data = module_org.bias.data[out_channels_mask]
            module_pruned.running_mean = module_org.running_mean[out_channels_mask] 
            module_pruned.running_var = module_org.running_var[out_channels_mask]

    missing = [name for name in maskbndict.keys() if name not in changed]        
    assert not missing, f"missing: {missing}"
    return pruned_model

def main(opt):
    weights, prune_ratio, cfg, model_size, save_dir = opt.weights, opt.prune_ratio, opt.cfg, opt.model_size, opt.save_dir
    model = AutoBackend(weights, fuse=False)
    model.eval()

    # Step 1: Prepare data for determining the pruning threshold.
    bn_dict, ignore_bn_list, chunk_bn_list, sorted_bn = prepare_prune_data(model)
    
    # Step 2: Determine the pruning threshold.
    highest_thre, percent_limit, thre = compute_prune_threshold(bn_dict, sorted_bn, prune_ratio)
    
    # Step 3: Rewrite the model configuration file as a dictionary (rewrote the C2f, SPPF, and Detect modules).
    pruned_yaml = rewrite_model_config(model, cfg, model_size)
    
    # Step 4: Calculate the mask for bn layers, multiply gamma and beta by the mask, and count the changes in the number of channels before and after pruning.
    maskbndict = compute_bn_mask(model, ignore_bn_list, chunk_bn_list, thre)

    # Step 5: Build the pruned model based on maskbndict.
    pruned_model = DetectionModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).cuda()  
    pruned_model.eval()
        
    # Step 6: Assign the parameters of the channels retained in the original model to the pruned model.
    pruned_model = assign_pruned_params(model, pruned_model, maskbndict)

    # Step 7: Save the pruned model.
    pruned_model.eval()
    save_path = os.path.join(save_dir, "pruned.pt")
    torch.save(
        {
            "model": pruned_model,
            "maskbndict": maskbndict  
        },
        save_path
    )
    model = torch.load(save_path)["model"]
    model = model.cuda()
    dummies = torch.randn([1, 3, 640, 640], dtype=torch.float32).cuda()
    model(dummies)
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'ultralytics/cfg/datasets/VOC-ball.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/detect/train-sparse/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default=ROOT / 'ultralytics/cfg/models/v8/yolov8.yaml', help='model.yaml path')
    parser.add_argument('--model-size', type=str, default='s', help='(yolov8)n, s, m, l or x?')
    parser.add_argument('--prune-ratio', type=float, default=0.3, help='prune ratio')
    parser.add_argument('--save-dir', type=str, default=ROOT / 'ultralytics/weights', help='pruned model weight save dir')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
