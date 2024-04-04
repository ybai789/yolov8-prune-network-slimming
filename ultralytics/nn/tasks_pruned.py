import contextlib
from copy import deepcopy

import torch
import torch.nn as nn

from ultralytics.nn.tasks import BaseModel
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.head_pruned import DetectPruned
from ultralytics.nn.modules.block_pruned import C2fPruned, SPPFPruned

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import initialize_weights,scale_img


class DetectionModelPruned(BaseModel):
    """
        forward继承BaseModel
    """
    def __init__(self, maskbndict, cfg, ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg
        # 注意这里一定要deepcopy一下cfg, 因为当模型剪枝过程全部结束时要把模型保存下来, 包括模型配置信息
        # 所以我们并不想改变模型配置信息, 如果改变, 当再次用配置信息去构建网络的时候就会出错
        self.model, self.save, self.current_to_prev = parse_model_pruned(maskbndict, deepcopy(cfg), ch)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (DetectPruned)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')
    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v8DetectionLoss(self)
    
def parse_model_pruned(maskbndict, d, ch, verbose=True):  # model_dict, input_channels(3)
    """    
    根据剪枝的结果重构网络结构
    这里要重写C2f模块、SPPF模块和Detect模块
    """
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    #  宽度系数没有用上, 因为每一层的输出通道数都是由mask来定的
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<50}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # current_to_prev字典按照 {某一BN层的名字: 该BN层连接的上一层(可能有多个)的BN层的名字} 的格式记录了
    # 网络中每一层与其前置层之间的连接关系。
    current_to_prev = {}
    # 按照{索引: 本层最后一个bn层的名字}记录
    idx_to_bn_layer_name = {}
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        
        base_name = f'model.{i}'
        if m in [Conv]:
            # Conv层的f只有-1
            c1 = ch[f]
            bn_layer_name = base_name + '.bn'
            mask = maskbndict[bn_layer_name]
            c2 = torch.sum(mask).int().item()
            args[0] = c2
            args.insert(0, c1)
            # =============================================
            if i == 0:
                prev_bn_layer_name = bn_layer_name
            else:
                current_to_prev[bn_layer_name] = prev_bn_layer_name
                prev_bn_layer_name = bn_layer_name
            idx_to_bn_layer_name[i] = bn_layer_name
            # =============================================
        elif m in [C2fPruned]:
            # 这里重点是搞清楚每一个Conv的in_channels和out_channels如何计算
            cv1in = ch[f]
            cv1_bn_layer_name = base_name + '.cv1.bn'
            inner_cv1_bn_layer_names = [base_name + f'.m.{i}.cv1.bn' for i in range(n)]
            inner_cv2_bn_layer_names = [base_name + f'.m.{i}.cv2.bn' for i in range(n)]
            cv2_bn_layer_name = base_name + '.cv2.bn'
            cv1_mask = maskbndict[cv1_bn_layer_name]
            inner_cv1_masks = [maskbndict[inner_cv1_bn_layer_name] for inner_cv1_bn_layer_name in inner_cv1_bn_layer_names]
            inner_cv2_masks = [maskbndict[inner_cv2_bn_layer_name] for inner_cv2_bn_layer_name in inner_cv2_bn_layer_names]
            cv2_mask = maskbndict[cv2_bn_layer_name]
            cv1out = torch.sum(cv1_mask).int().item()
            inner_cv1outs = [torch.sum(inner_cv1_mask).int().item() for inner_cv1_mask in inner_cv1_masks]
            # head部分的C2f层没有shortcut残差结构, 因此C2f结构中的第一个cv1层是可以被剪枝的
            # 但是剪完以后是不一定对称的, 因此要重新计算比例
            cv1_split_sections = [torch.sum(cv1_mask.chunk(2, 0)[0]).int().item(), torch.sum(cv1_mask.chunk(2, 0)[1]).int().item()] 
            inner_cv2outs = [torch.sum(inner_cv2_mask).int().item() for inner_cv2_mask in inner_cv2_masks]
            cv2out = torch.sum(cv2_mask).int().item()
            args = [cv1in, cv1out, cv1_split_sections, inner_cv1outs, inner_cv2outs, cv2out, n, *args[1:]]
            c2 = cv2out
            # =============================================
            current_to_prev[cv1_bn_layer_name] = prev_bn_layer_name
            # 如果当前C2f的前一层是Concat层, 就要记录Concat层是由哪几个支路来的
            if prev_module in [Concat]:
                current_to_prev[cv1_bn_layer_name] = [idx_to_bn_layer_name[ix] for ix in idx_to_bn_layer_name[i + f]]
            prev_bn_layer_name = cv1_bn_layer_name
            # C2f中的最后一个Conv层的输入通道数, 是由前面很多个分支共同决定的
            prev_bn_layer_names_for_last_cv2 = [cv1_bn_layer_name]
            for i_inner in range(n):
                inner_cv1_bn_layer_name = base_name + f'.m.{i_inner}.cv1.bn'
                inner_cv2_bn_layer_name = base_name + f'.m.{i_inner}.cv2.bn'
                current_to_prev[inner_cv1_bn_layer_name] = prev_bn_layer_name
                prev_bn_layer_name = inner_cv1_bn_layer_name
                current_to_prev[inner_cv2_bn_layer_name] = prev_bn_layer_name
                prev_bn_layer_name = inner_cv2_bn_layer_name
                prev_bn_layer_names_for_last_cv2.append(inner_cv2_bn_layer_name)
            current_to_prev[cv2_bn_layer_name] = prev_bn_layer_names_for_last_cv2
            prev_bn_layer_name = cv2_bn_layer_name
            idx_to_bn_layer_name[i] = cv2_bn_layer_name
            # =============================================
            n = 1 
        elif m in [SPPFPruned]:
            # SPPF层中不止是maxpool层, 还有两个卷积层, 所以要重写
            cv1in = ch[f]
            cv1_bn_layer_name = base_name + '.cv1.bn'
            cv1_mask = maskbndict[cv1_bn_layer_name]
            cv1out = torch.sum(cv1_mask).int().item()
            cv2_bn_layer_name = base_name + '.cv2.bn'
            cv2_mask = maskbndict[cv2_bn_layer_name]
            cv2out = torch.sum(cv2_mask).int().item()
            args = [cv1in, cv1out, cv2out, *args[1:]]
            c2 = cv2out
            # =============================================
            current_to_prev[cv1_bn_layer_name] = prev_bn_layer_name
            prev_bn_layer_name = cv1_bn_layer_name
            current_to_prev[cv2_bn_layer_name] = prev_bn_layer_name
            idx_to_bn_layer_name[i] = cv2_bn_layer_name
            # =============================================
        elif m in [nn.Upsample]:
            c2 = ch[f]
            idx_to_bn_layer_name[i] = idx_to_bn_layer_name[i-1]
        elif m in [Concat]:
            c2 = sum(ch[x] for x in f)
            # =============================================
            fx = []
            for ix in f:
                if ix == -1:
                    fx.append(i + ix)
                else:
                    fx.append(ix)
            idx_to_bn_layer_name[i] = fx
            # =============================================
        elif m in [DetectPruned]:
            # cv2x?是有所有的左侧分支, cv3x?是所有的右侧分支
            args.append([ch[x] for x in f])
            cv2x0_out_bn_layer_names = [base_name + f'.cv2.{i}.0.bn' for i in range(3)]
            cv2x1_out_bn_layer_names = [base_name + f'.cv2.{i}.1.bn' for i in range(3)]
            cv2x2_out_conv_layer_names = [base_name + f'.cv2.{i}.2' for i in range(3)]
            cv3x0_out_bn_layer_names = [base_name + f'.cv3.{i}.0.bn' for i in range(3)]
            cv3x1_out_bn_layer_names = [base_name + f'.cv3.{i}.1.bn' for i in range(3)]
            cv3x2_out_conv_layer_names = [base_name + f'.cv3.{i}.2' for i in range(3)]
            cv2x0_mask = [maskbndict[x] for x in cv2x0_out_bn_layer_names]
            cv2x1_mask = [maskbndict[x] for x in cv2x1_out_bn_layer_names]
            cv3x0_mask = [maskbndict[x] for x in cv3x0_out_bn_layer_names]
            cv3x1_mask = [maskbndict[x] for x in cv3x1_out_bn_layer_names]
            cv2x0_outs = [torch.sum(x).int().item() for x in cv2x0_mask]
            cv2x1_outs = [torch.sum(x).int().item() for x in cv2x1_mask]
            cv3x0_outs = [torch.sum(x).int().item() for x in cv3x0_mask]
            cv3x1_outs = [torch.sum(x).int().item() for x in cv3x1_mask]
            args = [cv2x0_outs, cv2x1_outs, cv3x0_outs, cv3x1_outs, *args]
            # =============================================
            for ix, (cv2x0_out_bn_layer_name, cv3x0_out_bn_layer_name) in \
                enumerate(zip(cv2x0_out_bn_layer_names, cv3x0_out_bn_layer_names)):
                current_to_prev[cv2x0_out_bn_layer_name] = idx_to_bn_layer_name[f[ix]]
                current_to_prev[cv3x0_out_bn_layer_name] = idx_to_bn_layer_name[f[ix]]
            for ix in range(3):
                current_to_prev[cv2x1_out_bn_layer_names[ix]] = cv2x0_out_bn_layer_names[ix]
                current_to_prev[cv3x1_out_bn_layer_names[ix]] = cv3x0_out_bn_layer_names[ix]
            for ix in range(3):
                current_to_prev[cv2x2_out_conv_layer_names[ix]] = cv2x1_out_bn_layer_names[ix]
                current_to_prev[cv3x2_out_conv_layer_names[ix]] = cv3x1_out_bn_layer_names[ix]
            # =============================================
        else:
            raise ValueError(f"ERROR ❌ module {m} not supported in parse_model.")
        prev_module = m
            
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<50}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), current_to_prev


if __name__ == "__main__":
    import prune
    dummies = torch.randn([4, 3, 640, 640], dtype=torch.float32).cuda()
    opt = prune.parse_opt()
    maskbndict, pruned_yaml = prune.main(opt)
    model = DetectionModelPruned(maskbndict, pruned_yaml, ch=3)
    model.train().cuda()
    out = model(dummies)