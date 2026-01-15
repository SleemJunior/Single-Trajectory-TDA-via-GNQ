"""
Standalone GNQ attribution DURING training on Mixed MNIST + CIFAR, using ResNet9.

IMPORTANT SEMANTICS (FIXED):
Rows 1–5 now answer the SAME question:

    "Which training points causally help/hurt THIS SINGLE query image?"

Concretely:
- For each class c, we pick ONE query image q_c = queries[c][0][query_index] (default query_index=0).
- GNQ / TraceIn / TraceIn-Norm are computed against ONLY that q_c (not mean over queries_per_class).
- DataModel (if enabled) evaluates reward ONLY on that same q_c per class.

So the displayed query in the PDF is exactly the query being used everywhere.

HOW TO RUN:
python GNQ_mnist_cifar.py   --layers all --steps 2000 --ckpt_every 200   --n_mnist 20000 --n_cifar 20000 --batch_size 128 --lr 1e-3   --queries_per_class 32 --query_index 0   --reward margin_mean_other   --lambda_reg 1e-2 --alpha_center   --query_bs 8 --query_cache_device cuda --q_chunk 64   --out_dir ./out_ggnq_vs_TraceIn_vs_NormTraceIn_vs_DataModel --save_npz   --dm_runs 64 --dm_steps 300 --dm_p 0.3 --dm_ridge 1e-2   --topk 10   --cf_topk 0 --cf_max_total 1000 --cf_random_per_class 0 --cf_rank abs
"""
from __future__ import annotations
import os
import gc
import time
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Conv overlap helper
# -----------------------------

def conv2d_overlap_sqrt_map(
    H_in, W_in,
    H_out, W_out,
    kernel_size, stride, padding, dilation,
    device, dtype=torch.float32,
):
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):      stride      = (stride, stride)
    if isinstance(padding, int):     padding     = (padding, padding)
    if isinstance(dilation, int):    dilation    = (dilation, dilation)

    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    H_nom = (H_out - 1) * sh - 2 * ph + dh * (kh - 1) + 1
    W_nom = (W_out - 1) * sw - 2 * pw + dw * (kw - 1) + 1

    op_h = H_in - H_nom
    op_w = W_in - W_nom

    if not (0 <= op_h < sh and 0 <= op_w < sw):
        raise ValueError(
            f"Bad output_padding computed: op_h={op_h}, op_w={op_w} "
            f"(H_in={H_in}, W_in={W_in}, H_nom={H_nom}, W_nom={W_nom}, stride={stride})"
        )

    ones_out = torch.ones((1, 1, H_out, W_out), device=device, dtype=dtype)
    ones_k   = torch.ones((1, 1, kh, kw), device=device, dtype=dtype)

    counts = F.conv_transpose2d(
        ones_out, ones_k,
        stride=stride, padding=padding, dilation=dilation,
        output_padding=(op_h, op_w),
    )[0, 0]

    return counts.clamp_min(0).sqrt()


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_antialias_resize():
    return {"antialias": True}

def tensor_to_img(x: torch.Tensor):
    x = x.detach().cpu()
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()

def sync_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def mem_report(device: torch.device):
    if device.type != "cuda":
        return ""
    alloc = torch.cuda.memory_allocated(device) / (1024**3)
    reserv = torch.cuda.memory_reserved(device) / (1024**3)
    return f"(cuda mem alloc={alloc:.2f}GB reserved={reserv:.2f}GB)"


# -----------------------------
# ResNet9 (small)
# -----------------------------

def conv_bn_relu(in_ch, out_ch, ks=3, stride=1, pad=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class Residual(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
    def forward(self, x):
        return x + self.block(x)

class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = conv_bn_relu(3, 64)
        self.conv2 = conv_bn_relu(64, 128, stride=2)
        self.res1  = Residual(nn.Sequential(conv_bn_relu(128, 128), conv_bn_relu(128, 128)))

        self.conv3 = conv_bn_relu(128, 256, stride=2)
        self.conv4 = conv_bn_relu(256, 512, stride=2)
        self.res2  = Residual(nn.Sequential(conv_bn_relu(512, 512), conv_bn_relu(512, 512)))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes, bias=True)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)
        x = self.pool(x).flatten(1)
        logits = self.fc(x)
        if return_feat:
            return logits, x
        return logits


# -----------------------------
# Data
# -----------------------------

class MixedMNISTCIFAR(Dataset):
    def __init__(self, mnist_train, cifar_train, n_mnist: int, n_cifar: int,
                 tf_mnist, tf_cifar):
        super().__init__()
        self.mnist = mnist_train
        self.cifar = cifar_train
        self.n_mnist = min(n_mnist, len(self.mnist))
        self.n_cifar = min(n_cifar, len(self.cifar))
        self.N = self.n_mnist + self.n_cifar
        self.tf_mnist = tf_mnist
        self.tf_cifar = tf_cifar

    def __len__(self):
        return self.N

    def __getitem__(self, gid: int):
        if gid < self.n_mnist:
            img, y = self.mnist[gid]
            x = self.tf_mnist(img)
            src = 0
        else:
            j = gid - self.n_mnist
            img, y = self.cifar[j]
            x = self.tf_cifar(img)
            src = 1
        return x, int(y), int(gid), int(src)

    def get_raw_for_plot(self, gid: int):
        if gid < self.n_mnist:
            img, y = self.mnist[gid]
            x = self.tf_mnist(img)
            src = "MNIST"
        else:
            j = gid - self.n_mnist
            img, y = self.cifar[j]
            x = self.tf_cifar(img)
            src = "CIFAR"
        return x, int(y), src

def build_datasets(data_root: str, n_mnist: int, n_cifar: int):
    tf_mnist = transforms.Compose([
        transforms.Resize((32, 32), **safe_antialias_resize()),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
    ])
    tf_cifar = transforms.Compose([transforms.ToTensor()])

    mnist_train = torchvision.datasets.MNIST(root=data_root, train=True, download=True)
    mnist_test_wrapped = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=tf_mnist
    )
    cifar_train = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)

    mixed_train = MixedMNISTCIFAR(
        mnist_train=mnist_train,
        cifar_train=cifar_train,
        n_mnist=n_mnist,
        n_cifar=n_cifar,
        tf_mnist=tf_mnist,
        tf_cifar=tf_cifar,
    )
    return mixed_train, mnist_test_wrapped

def sample_queries_per_class(mnist_test_ds, queries_per_class: int, seed: int):
    rng = np.random.RandomState(seed)
    by_c = {c: [] for c in range(10)}
    for i in range(len(mnist_test_ds)):
        x, y = mnist_test_ds[i]
        by_c[int(y)].append(i)

    queries = {}
    for c in range(10):
        idxs = by_c[c]
        rng.shuffle(idxs)
        pick = idxs[:queries_per_class]
        xs, ys = [], []
        for j in pick:
            x, y = mnist_test_ds[j]
            xs.append(x)
            ys.append(int(y))
        queries[c] = (torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long))
    return queries


# -----------------------------
# Reward + query delta
# -----------------------------

def reward_from_logits(logits: torch.Tensor, y: torch.Tensor, reward: str) -> torch.Tensor:
    if reward == "neg_ce":
        return -F.cross_entropy(logits, y, reduction="none")
    if reward == "logp":
        lp = F.log_softmax(logits, dim=1)
        return lp.gather(1, y.view(-1, 1)).squeeze(1)
    if reward == "prob":
        p = F.softmax(logits, dim=1)
        return p.gather(1, y.view(-1, 1)).squeeze(1)
    if reward == "margin":
        true = logits.gather(1, y.view(-1, 1)).squeeze(1)
        tmp = logits.clone()
        tmp[torch.arange(logits.size(0), device=logits.device), y] = -1e9
        other = tmp.max(dim=1).values
        return true - other
    if reward == "margin_mean_other":
        true = logits.gather(1, y.view(-1, 1)).squeeze(1)
        sum_all = logits.sum(dim=1)
        other_sum = sum_all - true
        mean_other = other_sum / (logits.size(1) - 1)
        return true - mean_other
    raise ValueError(reward)

def query_delta_from_logits(logits: torch.Tensor, y: torch.Tensor, reward: str) -> torch.Tensor:
    B, C = logits.shape
    if reward in ["neg_ce", "logp"]:
        p = F.softmax(logits, dim=1)
        oh = F.one_hot(y, num_classes=C).to(dtype=p.dtype)
        return p - oh
    if reward == "margin_mean_other":
        delta = torch.full_like(logits, fill_value=1.0/(C-1))
        delta[torch.arange(B, device=logits.device), y] = -1.0
        return delta

    logits2 = logits.detach().requires_grad_(True)
    r = reward_from_logits(logits2, y, reward)
    L = (-r).sum()
    (dlogits,) = torch.autograd.grad(L, logits2, create_graph=False, retain_graph=False)
    return dlogits.detach()

@torch.no_grad()
def eval_selected_query_rewards(
    model: nn.Module,
    queries: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    reward: str,
    query_index: int,
) -> torch.Tensor:
    """
    Returns (10,) where entry c is reward on the SINGLE selected query image for class c.
    This is the canonical evaluand for ALL methods (Rows 1–5).
    """
    model.eval()
    Rs = []
    for c in range(10):
        xq, yq = queries[c]
        if not (0 <= query_index < xq.size(0)):
            raise ValueError(f"query_index={query_index} out of range for class {c} with Qc={xq.size(0)}")
        xb = xq[query_index:query_index+1].to(device)
        yb = yq[query_index:query_index+1].to(device)
        logits = model(xb)
        r = reward_from_logits(logits, yb, reward).mean()
        Rs.append(r.detach().cpu())
    return torch.stack(Rs, dim=0)


# -----------------------------
# LAST-layer cache (SINGLE query per class)
# -----------------------------

@dataclass
class QueryCacheLast:
    feat_q: torch.Tensor
    delta_q: torch.Tensor
    cls_index: torch.Tensor

def build_query_cache_lastlayer_single_query_per_class(
    model: nn.Module,
    queries: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    reward: str,
    query_index: int,
    cache_device: torch.device,
) -> QueryCacheLast:
    """
    Builds a cache with EXACTLY ONE query per class (10 total queries).
    This enforces semantic alignment: rows 1–5 are about the same query image.
    """
    model.eval()
    feats_all, deltas_all, cls_all = [], [], []

    with torch.no_grad():
        for c in range(10):
            xq, yq = queries[c]
            if not (0 <= query_index < xq.size(0)):
                raise ValueError(f"query_index={query_index} out of range for class {c} with Qc={xq.size(0)}")
            xb = xq[query_index:query_index+1].to(device)
            yb = yq[query_index:query_index+1].to(device)

            logits, feat = model(xb, return_feat=True)
            delta_q = query_delta_from_logits(logits, yb, reward)

            feats_all.append(feat.detach().to(cache_device))
            deltas_all.append(delta_q.detach().to(cache_device))
            cls_all.append(torch.full((feat.size(0),), c, dtype=torch.long, device=cache_device))

            del xb, yb, logits, feat, delta_q

    feat_q = torch.cat(feats_all, dim=0)     # (10, D)
    delta_q = torch.cat(deltas_all, dim=0)   # (10, C)
    cls_index = torch.cat(cls_all, dim=0)    # (10,)
    return QueryCacheLast(feat_q=feat_q, delta_q=delta_q, cls_index=cls_index)

@torch.no_grad()
def batch_alphas_lastlayer_gnq_trace_tracenorm(
    feat_tr: torch.Tensor,
    logits_tr: torch.Tensor,
    y_tr: torch.Tensor,
    qcache: QueryCacheLast,
    lambda_reg: float,
    alpha_center: bool,
    cache_device: torch.device,
    device: torch.device,
    eps_norm: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (alpha_gnq, alpha_trace, alpha_trace_norm) on CPU, each (B,10).

    Each class column is computed using EXACTLY ONE query (the selected query image for that class).
    """
    B, C = logits_tr.shape

    p = F.softmax(logits_tr, dim=1)
    oh = F.one_hot(y_tr, num_classes=C).to(dtype=p.dtype)
    delta_tr = p - oh

    DD = (delta_tr @ delta_tr.t()).float()
    FF = (feat_tr @ feat_tr.t()).float()
    K = DD * (FF + 1.0)

    diagK_tr = K.diag().clamp_min(0.0)  # (B,)

    Kreg = K + float(lambda_reg) * torch.eye(B, device=device, dtype=K.dtype)

    alpha_gnq = torch.zeros((B, 10), dtype=torch.float32, device="cpu")
    alpha_tr  = torch.zeros((B, 10), dtype=torch.float32, device="cpu")
    alpha_tn  = torch.zeros((B, 10), dtype=torch.float32, device="cpu")

    # qcache has exactly one query per class, so idx.numel()==1
    for c in range(10):
        idx = (qcache.cls_index == c).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue

        feat_q = qcache.feat_q[idx]
        delta_q = qcache.delta_q[idx]
        if cache_device != device:
            feat_q = feat_q.to(device)
            delta_q = delta_q.to(device)

        DDq = (delta_tr @ delta_q.t()).float()
        FFq = (feat_tr @ feat_q.t()).float()
        Kq = DDq * (FFq + 1.0)  # (B,1)

        dq2 = (delta_q.float().pow(2).sum(dim=1))               # (1,)
        fq2 = (feat_q.float().pow(2).sum(dim=1) + 1.0)          # (1,)
        diagK_q = (dq2 * fq2).clamp_min(0.0)                    # (1,)

        denom = (diagK_tr.view(B, 1) * diagK_q.view(1, -1)).sqrt().add_(eps_norm)
        Kq_norm = Kq / denom

        A = torch.linalg.solve(Kreg, Kq)        # (B,1)
        a_g  = A.mean(dim=1)
        a_tr = Kq.mean(dim=1)
        a_tn = Kq_norm.mean(dim=1)

        if alpha_center:
            a_g  = a_g  - a_g.mean()
            a_tr = a_tr - a_tr.mean()
            a_tn = a_tn - a_tn.mean()

        alpha_gnq[:, c] = a_g.detach().cpu()
        alpha_tr[:,  c] = a_tr.detach().cpu()
        alpha_tn[:,  c] = a_tn.detach().cpu()

        del feat_q, delta_q, DDq, FFq, Kq, dq2, fq2, diagK_q, denom, Kq_norm, A, a_g, a_tr, a_tn

    return alpha_gnq, alpha_tr, alpha_tn


# -----------------------------
# ALL-layer Ghost hooks + cache (SINGLE query per class)
# -----------------------------

@dataclass
class LayerSpec:
    name: str
    kind: str
    has_bias: bool
    kernel_size: Optional[Tuple[int,int]] = None
    stride: Optional[Tuple[int,int]] = None
    padding: Optional[Tuple[int,int]] = None
    dilation: Optional[Tuple[int,int]] = None

class GhostHooks:
    def __init__(self, model: nn.Module):
        self.model = model
        self.specs: List[LayerSpec] = []
        self.modules: Dict[str, nn.Module] = {}
        self.handles = []

        self.a: Dict[str, torch.Tensor] = {}
        self.y: Dict[str, torch.Tensor] = {}

        self._mult_cache: Dict[Tuple[str,int,int,str,str], torch.Tensor] = {}

        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                ks = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
                st = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
                pd = m.padding if isinstance(m.padding, tuple) else (m.padding, m.padding)
                dl = m.dilation if isinstance(m.dilation, tuple) else (m.dilation, m.dilation)
                self.specs.append(LayerSpec(
                    name=name, kind="conv", has_bias=(m.bias is not None),
                    kernel_size=ks, stride=st, padding=pd, dilation=dl
                ))
                self.modules[name] = m
            elif isinstance(m, nn.Linear):
                self.specs.append(LayerSpec(
                    name=name, kind="linear", has_bias=(m.bias is not None)
                ))
                self.modules[name] = m

        for spec in self.specs:
            m = self.modules[spec.name]
            self.handles.append(m.register_forward_hook(self._fwd_hook(spec.name)))

    def _fwd_hook(self, name: str):
        def hook(mod, inp, out):
            self.a[name] = inp[0]
            self.y[name] = out
            if isinstance(out, torch.Tensor) and out.requires_grad:
                out.retain_grad()
        return hook

    def clear(self):
        self.a.clear()
        self.y.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @torch.no_grad()
    def conv_mult_map(self, spec: LayerSpec, Hin: int, Win: int, device: torch.device, dtype: torch.dtype):
        key = (spec.name, Hin, Win, str(device), str(dtype))
        if key in self._mult_cache:
            return self._mult_cache[key]

        kH, kW = spec.kernel_size
        sH, sW = spec.stride
        pH, pW = spec.padding
        dH, dW = spec.dilation

        def out_dim(L, K, S, P, D):
            return (L + 2*P - D*(K-1) - 1)//S + 1

        Hout = out_dim(Hin, kH, sH, pH, dH)
        Wout = out_dim(Win, kW, sW, pW, dW)

        ones_out = torch.ones((1,1,Hout,Wout), device=device, dtype=dtype)
        ones_k   = torch.ones((1,1,kH,kW), device=device, dtype=dtype)

        counts_in = F.conv_transpose2d(
            ones_out, ones_k,
            stride=(sH,sW), padding=(pH,pW), dilation=(dH,dW)
        )
        self._mult_cache[key] = counts_in
        return counts_in

@dataclass
class QueryCacheAll:
    cls_index: torch.Tensor
    per_layer_a: Dict[str, torch.Tensor]
    per_layer_d: Dict[str, torch.Tensor]

def _flatten_dict_ordered(specs: List[LayerSpec]) -> List[str]:
    return [s.name for s in specs]

def build_query_cache_alllayers_single_query_per_class(
    model: nn.Module,
    hooks: GhostHooks,
    queries: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    reward: str,
    query_index: int,
    cache_device: torch.device,
) -> QueryCacheAll:
    """
    Builds an ALL-layer cache with EXACTLY ONE query per class (10 total).
    """
    model.eval()

    per_layer_a = {name: [] for name in _flatten_dict_ordered(hooks.specs)}
    per_layer_d = {name: [] for name in _flatten_dict_ordered(hooks.specs)}
    cls_all = []

    for c in range(10):
        xq, yq = queries[c]
        if not (0 <= query_index < xq.size(0)):
            raise ValueError(f"query_index={query_index} out of range for class {c} with Qc={xq.size(0)}")

        xb = xq[query_index:query_index+1].to(device)
        yb = yq[query_index:query_index+1].to(device)

        hooks.clear()
        model.zero_grad(set_to_none=True)

        logits = model(xb)
        dlogits = query_delta_from_logits(logits, yb, reward)
        logits.backward(dlogits)

        bsz = xb.size(0)  # 1
        cls_all.append(torch.full((bsz,), c, dtype=torch.long, device=cache_device))

        for spec in hooks.specs:
            name = spec.name
            a = hooks.a[name].detach().to(cache_device)
            d = hooks.y[name].grad.detach().to(cache_device)
            per_layer_a[name].append(a)
            per_layer_d[name].append(d)

        del xb, yb, logits, dlogits

    for name in per_layer_a.keys():
        per_layer_a[name] = torch.cat(per_layer_a[name], dim=0)  # (10, ...)
        per_layer_d[name] = torch.cat(per_layer_d[name], dim=0)

    cls_index = torch.cat(cls_all, dim=0)  # (10,)
    return QueryCacheAll(cls_index=cls_index, per_layer_a=per_layer_a, per_layer_d=per_layer_d)

@torch.no_grad()
def batch_alphas_alllayers_gnq_trace_tracenorm(
    hooks: GhostHooks,
    qcache: QueryCacheAll,
    lambda_reg: float,
    alpha_center: bool,
    cache_device: torch.device,
    device: torch.device,
    q_chunk: int = 64,
    ggnq_buf=None,
    eps_norm: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (alpha_gnq, alpha_trace, alpha_trace_norm) on CPU, each (B,10).

    qcache contains EXACTLY ONE query per class (Q=10).
    Thus each class column corresponds to THAT single query image.
    """
    first = hooks.specs[0].name
    B = hooks.a[first].shape[0]

    any_name = hooks.specs[0].name
    Q = qcache.per_layer_a[any_name].shape[0]  # should be 10

    n_chunks = (Q + q_chunk - 1) // q_chunk

    if ggnq_buf is None:
        K_total  = torch.zeros((B,B), device=device, dtype=torch.float32)
        Kq_total = torch.zeros((B,Q), device=device, dtype=torch.float32)
        I = torch.eye(B, device=device, dtype=torch.float32)
    else:
        K_total  = ggnq_buf["K_total"];  K_total.zero_()
        Kq_total = ggnq_buf["Kq_total"]; Kq_total.zero_()
        I = ggnq_buf["I"]

    diagK_tr_unscaled = torch.zeros((B,), device=device, dtype=torch.float32)
    diagK_q_total     = torch.zeros((Q,), device=device, dtype=torch.float32)

    for spec in hooks.specs:
        name = spec.name

        a_tr = hooks.a[name].detach()
        d_tr = hooks.y[name].grad.detach()

        a_q_all = qcache.per_layer_a[name]
        d_q_all = qcache.per_layer_d[name]

        if spec.kind == "linear":
            A_tr = a_tr.flatten(1).float()
            D_tr = d_tr.flatten(1).float()

            DD = D_tr @ D_tr.t()
            AA = A_tr @ A_tr.t()
            K_layer = DD * AA
            if spec.has_bias:
                K_layer = K_layer + DD

            K_total += float(n_chunks) * K_layer
            diagK_tr_unscaled += K_layer.diag()

            for qs in range(0, Q, q_chunk):
                a_q = a_q_all[qs:qs+q_chunk]
                d_q = d_q_all[qs:qs+q_chunk]
                if cache_device != device:
                    a_q = a_q.to(device, non_blocking=True)
                    d_q = d_q.to(device, non_blocking=True)

                A_q = a_q.flatten(1).float()
                D_q = d_q.flatten(1).float()

                DDq = D_tr @ D_q.t()
                AAq = A_tr @ A_q.t()
                Kq = DDq * AAq
                if spec.has_bias:
                    Kq = Kq + DDq

                Kq_total[:, qs:qs+q_chunk] += Kq

                dq2 = D_q.pow(2).sum(dim=1)
                aq2 = A_q.pow(2).sum(dim=1)
                diag_layer_q = dq2 * aq2
                if spec.has_bias:
                    diag_layer_q = diag_layer_q + dq2
                diagK_q_total[qs:qs+q_chunk] += diag_layer_q

        else:
            x_tr = a_tr
            B2, Cin, H_in, W_in = x_tr.shape
            assert B2 == B
            D_tr = d_tr.flatten(1).float()

            mult = hooks.conv_mult_map(spec, H_in, W_in, device=device, dtype=a_tr.dtype)
            w2d = mult

            if (w2d is None) or (w2d.shape[-2] != H_in) or (w2d.shape[-1] != W_in):
                _, _, H_out, W_out = d_tr.shape
                w2d = conv2d_overlap_sqrt_map(
                    H_in, W_in, H_out, W_out,
                    kernel_size=spec.kernel_size,
                    stride=spec.stride,
                    padding=spec.padding,
                    dilation=spec.dilation,
                    device=device,
                    dtype=x_tr.dtype,
                )
            else:
                w2d = w2d.to(device=device, dtype=x_tr.dtype)

            X_tr = (x_tr * w2d.view(1, 1, H_in, W_in)).flatten(1).float()

            DD = D_tr @ D_tr.t()
            XX = X_tr @ X_tr.t()
            K_layer = DD * (XX + (1.0 if spec.has_bias else 0.0))

            K_total += float(n_chunks) * K_layer
            diagK_tr_unscaled += K_layer.diag()

            for qs in range(0, Q, q_chunk):
                a_q = a_q_all[qs:qs+q_chunk]
                d_q = d_q_all[qs:qs+q_chunk]
                if cache_device != device:
                    a_q = a_q.to(device, non_blocking=True)
                    d_q = d_q.to(device, non_blocking=True)

                x_q = a_q
                D_q = d_q.flatten(1).float()

                _, _, H_in_q, W_in_q = x_q.shape
                _, _, H_out_q, W_out_q = d_q.shape

                w2d_q = w2d
                if (H_in_q != H_in) or (W_in_q != W_in) or (H_out_q != d_tr.shape[-2]) or (W_out_q != d_tr.shape[-1]):
                    w2d_q = conv2d_overlap_sqrt_map(
                        H_in_q, W_in_q, H_out_q, W_out_q,
                        kernel_size=spec.kernel_size,
                        stride=spec.stride,
                        padding=spec.padding,
                        dilation=spec.dilation,
                        device=device,
                        dtype=x_q.dtype,
                    )
                else:
                    w2d_q = w2d_q.to(device=device, dtype=x_q.dtype)

                X_q = (x_q * w2d_q.view(1, 1, H_in_q, W_in_q)).flatten(1).float()

                DDq = D_tr @ D_q.t()
                XXq = X_tr @ X_q.t()
                Kq = DDq * (XXq + (1.0 if spec.has_bias else 0.0))
                Kq_total[:, qs:qs+q_chunk] += Kq

                dq2 = D_q.pow(2).sum(dim=1)
                xq2 = X_q.pow(2).sum(dim=1)
                diag_layer_q = dq2 * (xq2 + (1.0 if spec.has_bias else 0.0))
                diagK_q_total[qs:qs+q_chunk] += diag_layer_q

    Kreg = K_total + float(lambda_reg) * I
    A_gnq = torch.linalg.solve(Kreg, Kq_total)

    diagK_tr_unscaled = diagK_tr_unscaled.clamp_min(0.0)
    diagK_q_total     = diagK_q_total.clamp_min(0.0)
    denom = (diagK_tr_unscaled.view(B, 1) * diagK_q_total.view(1, Q)).sqrt().add_(eps_norm)
    Kq_norm = Kq_total / denom

    alpha_gnq = torch.zeros((B, 10), dtype=torch.float32, device="cpu")
    alpha_tr  = torch.zeros((B, 10), dtype=torch.float32, device="cpu")
    alpha_tn  = torch.zeros((B, 10), dtype=torch.float32, device="cpu")

    # qcache has exactly one query per class, so idx.numel()==1
    for c in range(10):
        idx = (qcache.cls_index == c).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue

        a_g  = A_gnq[:, idx].mean(dim=1)
        a_tr_c = Kq_total[:, idx].mean(dim=1)
        a_tn_c = Kq_norm[:, idx].mean(dim=1)

        if alpha_center:
            a_g    = a_g    - a_g.mean()
            a_tr_c = a_tr_c - a_tr_c.mean()
            a_tn_c = a_tn_c - a_tn_c.mean()

        alpha_gnq[:, c] = a_g.detach().cpu()
        alpha_tr[:,  c] = a_tr_c.detach().cpu()
        alpha_tn[:,  c] = a_tn_c.detach().cpu()

    return alpha_gnq, alpha_tr, alpha_tn


# -----------------------------
# Plotting (3 rows)
# -----------------------------

def make_class_pdf_compare3(
    out_path: str,
    cls: int,
    mixed_ds: MixedMNISTCIFAR,
    query_img: torch.Tensor,
    query_pred: int,
    query_conf: float,
    top_gnq: List[int],
    top_tr: List[int],
    top_tn: List[int],
    scores_gnq_c: torch.Tensor,
    scores_tr_c: torch.Tensor,
    scores_tn_c: torch.Tensor,
    seen_counts: torch.Tensor,
    title_suffix: str,
    layers_mode: str,
):
    k = len(top_gnq)
    fig = plt.figure(figsize=(16, 10))
    # fig.suptitle(
    #     f"Attribution comparison ({layers_mode}): GNQ vs TraceIn vs TraceIn-Norm\n"
    #     f"MNIST SINGLE query (class={cls}) vs Mixed MNIST + CIFAR training\n{title_suffix}",
    #     fontsize=15,
    #     y=0.98,
    # )

    axq = fig.add_axes([0.05, 0.35, 0.18, 0.32])
    axq.imshow(tensor_to_img(query_img))
    axq.axis("off")
    axq.set_title(f"Query (MNIST test)\npred={query_pred} conf={query_conf:.3f}", fontsize=12)

    # fig.text(0.70, 0.90, f"Row 1: Top-{k} by |score| — GNQ", ha="center", fontsize=12)
    # fig.text(0.70, 0.62, f"Row 2: Top-{k} by |score| — TraceIn (raw dot)", ha="center", fontsize=12)
    # fig.text(0.70, 0.34, f"Row 3: Top-{k} by |score| — TraceIn-Norm", ha="center", fontsize=12)

    def draw_row(gids: List[int], y0: float, scores_c: torch.Tensor):
        w = 0.11
        h = 0.18
        x0 = 0.28
        for j, gid in enumerate(gids):
            ax = fig.add_axes([x0 + j * w, y0, w - 0.01, h])
            x, y, src = mixed_ds.get_raw_for_plot(gid)
            ax.imshow(tensor_to_img(x))
            ax.axis("off")
            s = float(scores_c[gid].item())
            cnt = int(seen_counts[gid].item())
            ax.set_title(f"{src} id={gid}\ny={y} cnt={cnt}\ns={s:+.2e}", fontsize=8)

    draw_row(top_gnq, y0=0.68, scores_c=scores_gnq_c)
    draw_row(top_tr,  y0=0.40, scores_c=scores_tr_c)
    draw_row(top_tn,  y0=0.12, scores_c=scores_tn_c)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Plotting (4 rows: add DataModel)
# -----------------------------

def make_class_pdf_compare4(
    out_path: str,
    cls: int,
    mixed_ds: MixedMNISTCIFAR,
    query_img: torch.Tensor,
    query_pred: int,
    query_conf: float,
    top_gnq: List[int],
    top_tr: List[int],
    top_tn: List[int],
    top_dm: List[int],
    scores_gnq_c: torch.Tensor,
    scores_tr_c: torch.Tensor,
    scores_tn_c: torch.Tensor,
    scores_dm_c: torch.Tensor,
    seen_counts: torch.Tensor,
    title_suffix: str,
    layers_mode: str,
):
    k = len(top_gnq)
    fig = plt.figure(figsize=(16, 12))
    # fig.suptitle(
    #     f"Attribution + causal proxy ({layers_mode}): GNQ vs TraceIn vs TraceIn-Norm vs DataModel\n"
    #     f"MNIST SINGLE query (class={cls}) vs Mixed MNIST + CIFAR training\n{title_suffix}",
    #     fontsize=14,
    #     y=0.98,
    # )

    axq = fig.add_axes([0.05, 0.34, 0.18, 0.30])
    axq.imshow(tensor_to_img(query_img))
    axq.axis("off")
    axq.set_title(f"Query (MNIST test)\npred={query_pred} conf={query_conf:.3f}", fontsize=12)

    # fig.text(0.70, 0.90, f"Row 1: Top-{k} by |score| — GNQ", ha="center", fontsize=12)
    # fig.text(0.70, 0.69, f"Row 2: Top-{k} by |score| — TraceIn (raw dot)", ha="center", fontsize=12)
    # fig.text(0.70, 0.48, f"Row 3: Top-{k} by |score| — TraceIn-Norm", ha="center", fontsize=12)
    # fig.text(0.70, 0.27, f"Row 4: Top-{k} by |score| — DataModel (proxy)", ha="center", fontsize=12)

    def draw_row(gids: List[int], y0: float, scores_c: torch.Tensor):
        w = 0.11
        h = 0.16
        x0 = 0.28
        for j, gid in enumerate(gids):
            ax = fig.add_axes([x0 + j * w, y0, w - 0.01, h])
            x, y, src = mixed_ds.get_raw_for_plot(gid)
            ax.imshow(tensor_to_img(x))
            ax.axis("off")
            s = float(scores_c[gid].item())
            cnt = int(seen_counts[gid].item())
            ax.set_title(f"{src} id={gid}\ny={y} cnt={cnt}\ns={s:+.2e}", fontsize=8)

    draw_row(top_dm,  y0=0.76, scores_c=scores_dm_c)
    draw_row(top_tr,  y0=0.55, scores_c=scores_tr_c)
    draw_row(top_tn,  y0=0.34, scores_c=scores_tn_c)
    draw_row(top_gnq, y0=0.13, scores_c=scores_gnq_c)
    

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Plotting (5 rows: add TrueCF)
# -----------------------------

def make_class_pdf_compare5(
    out_path: str,
    cls: int,
    mixed_ds: MixedMNISTCIFAR,
    query_img: torch.Tensor,
    query_pred: int,
    query_conf: float,
    top_gnq: List[int],
    top_tr: List[int],
    top_tn: List[int],
    top_dm: List[int],
    top_cf: List[int],
    scores_gnq_c: torch.Tensor,
    scores_tr_c: torch.Tensor,
    scores_tn_c: torch.Tensor,
    scores_dm_c: torch.Tensor,
    scores_cf_c: torch.Tensor,
    seen_counts: torch.Tensor,
    title_suffix: str,
    layers_mode: str,
):
    k = len(top_gnq)
    fig = plt.figure(figsize=(16, 14))
    # fig.suptitle(
    #     f"Attribution + causal ({layers_mode}): GNQ vs TraceIn vs TraceIn-Norm vs DataModel vs TrueCF\n"
    #     f"MNIST SINGLE query (class={cls}) vs Mixed MNIST + CIFAR training\n{title_suffix}",
    #     fontsize=13,
    #     y=0.98,
    # )

    axq = fig.add_axes([0.05, 0.34, 0.18, 0.28])
    axq.imshow(tensor_to_img(query_img))
    axq.axis("off")
    axq.set_title(f"Query (MNIST test)\npred={query_pred} conf={query_conf:.3f}", fontsize=12)

    # fig.text(0.70, 0.91, f"Row 1: Top-{k} by |score| — GNQ", ha="center", fontsize=11)
    # fig.text(0.70, 0.74, f"Row 2: Top-{k} by |score| — TraceIn (raw dot)", ha="center", fontsize=11)
    # fig.text(0.70, 0.57, f"Row 3: Top-{k} by |score| — TraceIn-Norm", ha="center", fontsize=11)
    # fig.text(0.70, 0.40, f"Row 4: Top-{k} by |score| — DataModel (proxy)", ha="center", fontsize=11)
    # fig.text(0.70, 0.23, f"Row 5: Top-{k} within C_c — TrueCF (single-run replay)", ha="center", fontsize=11)

    def draw_row(gids: List[int], y0: float, scores_c: torch.Tensor):
        w = 0.11
        h = 0.13
        x0 = 0.28
        for j, gid in enumerate(gids):
            ax = fig.add_axes([x0 + j * w, y0, w - 0.01, h])
            x, y, src = mixed_ds.get_raw_for_plot(gid)
            ax.imshow(tensor_to_img(x))
            ax.axis("off")
            s = float(scores_c[gid].item())
            cnt = int(seen_counts[gid].item())
            ax.set_title(f"{src} id={gid}\ny={y} cnt={cnt}\ns={s:+.2e}", fontsize=8)

    draw_row(top_dm,  y0=0.80, scores_c=scores_dm_c)
    draw_row(top_tr,  y0=0.63, scores_c=scores_tr_c)
    draw_row(top_tn,  y0=0.46, scores_c=scores_tn_c)
    draw_row(top_gnq, y0=0.29, scores_c=scores_gnq_c)
    draw_row(top_cf,  y0=0.12, scores_c=scores_cf_c)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Plotting (4 rows: GNQ/Trace/TraceNorm + TrueCF) when DataModel is OFF
# -----------------------------

def make_class_pdf_compare4_truecf(
    out_path: str,
    cls: int,
    mixed_ds: MixedMNISTCIFAR,
    query_img: torch.Tensor,
    query_pred: int,
    query_conf: float,
    top_gnq: List[int],
    top_tr: List[int],
    top_tn: List[int],
    top_cf: List[int],
    scores_gnq_c: torch.Tensor,
    scores_tr_c: torch.Tensor,
    scores_tn_c: torch.Tensor,
    scores_cf_c: torch.Tensor,
    seen_counts: torch.Tensor,
    title_suffix: str,
    layers_mode: str,
):
    k = len(top_gnq)
    fig = plt.figure(figsize=(16, 12))
    # fig.suptitle(
    #     f"Attribution + causal ({layers_mode}): GNQ vs TraceIn vs TraceIn-Norm vs TrueCF\n"
    #     f"MNIST SINGLE query (class={cls}) vs Mixed MNIST + CIFAR training\n{title_suffix}",
    #     fontsize=14,
    #     y=0.98,
    # )

    axq = fig.add_axes([0.05, 0.34, 0.18, 0.30])
    axq.imshow(tensor_to_img(query_img))
    axq.axis("off")
    axq.set_title(f"Query (MNIST test)\npred={query_pred} conf={query_conf:.3f}", fontsize=12)

    # fig.text(0.70, 0.90, f"Row 1: Top-{k} by |score| — GNQ", ha="center", fontsize=12)
    # fig.text(0.70, 0.69, f"Row 2: Top-{k} by |score| — TraceIn (raw dot)", ha="center", fontsize=12)
    # fig.text(0.70, 0.48, f"Row 3: Top-{k} by |score| — TraceIn-Norm", ha="center", fontsize=12)
    # fig.text(0.70, 0.27, f"Row 4: Top-{k} within C_c — TrueCF (single-run replay)", ha="center", fontsize=12)

    def draw_row(gids: List[int], y0: float, scores_c: torch.Tensor):
        w = 0.11
        h = 0.16
        x0 = 0.28
        for j, gid in enumerate(gids):
            ax = fig.add_axes([x0 + j * w, y0, w - 0.01, h])
            x, y, src = mixed_ds.get_raw_for_plot(gid)
            ax.imshow(tensor_to_img(x))
            ax.axis("off")
            s = float(scores_c[gid].item())
            cnt = int(seen_counts[gid].item())
            ax.set_title(f"{src} id={gid}\ny={y} cnt={cnt}\ns={s:+.2e}", fontsize=8)

    draw_row(top_gnq, y0=0.76, scores_c=scores_gnq_c)
    draw_row(top_tr,  y0=0.55, scores_c=scores_tr_c)
    draw_row(top_tn,  y0=0.34, scores_c=scores_tn_c)
    draw_row(top_cf,  y0=0.13, scores_c=scores_cf_c)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# DataModel (paper-style) helpers
# -----------------------------

def fit_ridge_centered_dual(Z: torch.Tensor, y: torch.Tensor, ridge: float):
    """
    y ≈ beta0 + Z beta, solved via centering + dual ridge.
    Z: (M,N), y: (M,)
    """
    Z = Z.float()
    y = y.float()

    mu_z = Z.mean(dim=0)          # (N,)
    mu_y = y.mean()               # scalar

    Zc = Z - mu_z                 # (M,N)
    yc = y - mu_y                 # (M,)

    M = Zc.shape[0]
    I = torch.eye(M, device=Z.device, dtype=Z.dtype)

    G = Zc @ Zc.t()               # (M,M)
    a = torch.linalg.solve(G + ridge * I, yc)  # (M,)
    beta = Zc.t() @ a             # (N,)

    beta0 = mu_y - (mu_z @ beta)
    return beta0, beta

def run_datamodel_p_fraction(
    mixed_train: Dataset,
    queries: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    reward: str,
    device: torch.device,
    seed: int,
    runs: int,
    steps: int,
    lr: float,
    batch_size: int,
    p: float,
    ridge: float,
    dm_cache_device: torch.device,
    query_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random p-fraction subsets of full training set, fresh model per run.
    IMPORTANT: y_m[c] is reward on the SINGLE selected query image for class c.

    Returns:
      beta_full: (N,10) on CPU
      beta0:     (10,) on CPU
    """
    set_seed(seed)
    N = len(mixed_train)
    M = runs
    if not (0.0 < p <= 1.0):
        raise ValueError("--dm_p must be in (0,1].")

    k = max(1, int(round(p * N)))  # subset size

    Z = torch.zeros((M, N), dtype=torch.float32, device=dm_cache_device)
    Y = torch.zeros((M, 10), dtype=torch.float32, device=dm_cache_device)

    rng = np.random.RandomState(seed + 12345)

    for m in range(M):
        set_seed(seed + 10_000 + m)

        idx = rng.choice(N, size=k, replace=False)
        idx = np.array(idx, dtype=np.int64)

        Z[m, idx] = 1.0

        subset = torch.utils.data.Subset(mixed_train, idx.tolist())
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        it = iter(loader)

        model = ResNet9(num_classes=10).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            x, y, *_ = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

        Rm = eval_selected_query_rewards(model, queries, device, reward, query_index=query_index)  # (10,)
        Y[m] = Rm.to(dm_cache_device)

        del model, opt, subset, loader, it
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        print(f"[DataModel-p] run {m+1:3d}/{M} done. subset_size={k}")

    beta0 = torch.zeros((10,), dtype=torch.float32, device=dm_cache_device)
    beta  = torch.zeros((N, 10), dtype=torch.float32, device=dm_cache_device)

    for c in range(10):
        b0, bc = fit_ridge_centered_dual(Z, Y[:, c], ridge=ridge)
        beta0[c] = b0
        beta[:, c] = bc

    return beta.to("cpu"), beta0.to("cpu")


# -----------------------------
# True single-run counterfactual (run-conditioned)
# -----------------------------

class BatchSamplerFromList(torch.utils.data.Sampler):
    """Yields pre-recorded batches of indices (list[int]) in order."""
    def __init__(self, batches: List[List[int]]):
        self.batches = batches
    def __iter__(self):
        for b in self.batches:
            yield b
    def __len__(self):
        return len(self.batches)

def build_replay_loader(
    mixed_train: Dataset,
    recorded_batches: List[List[int]],
    device: torch.device,
    num_workers: int = 2
):
    batch_sampler = BatchSamplerFromList(recorded_batches)
    return DataLoader(
        mixed_train,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

@torch.no_grad()
def _set_deterministic_if_needed(flag: bool):
    if not flag:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def run_single_run_counterfactual_replay(
    mixed_train: Dataset,
    recorded_batches: List[List[int]],
    queries: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    reward: str,
    device: torch.device,
    seed: int,
    steps: int,
    lr: float,
    target_gid: int,
    init_state_dict_cpu: Dict[str, torch.Tensor],
    cf_deterministic: bool,
    cf_eps: float,
    query_index: int,
) -> torch.Tensor:
    """
    Replays SAME batch sequence as the original run, from SAME init weights,
    but zeros the loss contribution for `target_gid` whenever it appears.

    Returns: R_cf (10,) on CPU, where each class entry is reward on the SINGLE selected query image.
    """
    set_seed(seed)
    _set_deterministic_if_needed(cf_deterministic)

    model = ResNet9(num_classes=10).to(device)
    model.load_state_dict(init_state_dict_cpu, strict=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    replay_loader = build_replay_loader(mixed_train, recorded_batches, device=device, num_workers=2)
    it = iter(replay_loader)

    model.train()
    for _ in range(steps):
        try:
            x, y, gid, _src = next(it)
        except StopIteration:
            it = iter(replay_loader)
            x, y, gid, _src = next(it)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        gid = gid.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits = model(x)

        loss_vec = F.cross_entropy(logits, y, reduction="none")  # (B,)
        w = (gid != int(target_gid)).to(loss_vec.dtype)          # (B,)
        loss = (w * loss_vec).mean()  # exact 1/B scaling; just zero that example's gradient

        if cf_eps > 0.0:
            loss = (w * loss_vec).sum() / (w.sum() + cf_eps)
        else:
            # exact 1/B scaling; just zero that example's gradient
            loss = (w * loss_vec).mean()

        loss.backward()
        opt.step()

    R_cf = eval_selected_query_rewards(model, queries, device, reward, query_index=query_index)  # (10,) CPU
    del model, opt, replay_loader, it
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return R_cf


# def run_single_run_counterfactual_replay(
#     mixed_train: Dataset,
#     recorded_batches: List[List[int]],
#     queries: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
#     reward: str,
#     device: torch.device,
#     seed: int,
#     steps: int,
#     lr: float,
#     target_gid: int,
#     init_state_dict_cpu: Dict[str, torch.Tensor],
#     cf_deterministic: bool,
#     cf_eps: float,          # kept for API compatibility (unused in the fixed TrueCF)
#     query_index: int,
# ) -> torch.Tensor:
#     """
#     Replays SAME batch sequence as the original run, from SAME init weights,
#     but REPLACES `target_gid` whenever it appears (keeps batch size + mean scaling identical).

#     Returns: R_cf (10,) on CPU, where each class entry is reward on the SINGLE selected query image.
#     """
#     set_seed(seed)
#     _set_deterministic_if_needed(cf_deterministic)

#     # ---------------------------------------------------------
#     # Choose a deterministic replacement example (prefer same label)
#     # ---------------------------------------------------------
#     rng = np.random.RandomState(int(seed) + 7777)

#     _x_t, y_t, _gid_t, _src_t = mixed_train[int(target_gid)]
#     y_t = int(y_t)

#     N = len(mixed_train)
#     rep_gid = None
#     for _ in range(2000):  # cheap probing; only done once per TrueCF run
#         g = int(rng.randint(0, N))
#         if g == int(target_gid):
#             continue
#         _x_r, y_r, _gid_r, _src_r = mixed_train[g]
#         if int(y_r) == y_t:
#             rep_gid = int(_gid_r)
#             x_rep0 = _x_r
#             y_rep0 = int(y_r)
#             break

#     if rep_gid is None:
#         # fallback: any other gid
#         rep_gid = int((int(target_gid) + 1) % N)
#         x_rep0, y_rep0, _gid_r, _src_r = mixed_train[rep_gid]
#         y_rep0 = int(y_rep0)

#     # prepare replacement tensors (single example)
#     x_rep = x_rep0.to(device).unsqueeze(0)                         # (1,C,H,W)
#     y_rep = torch.tensor([y_rep0], dtype=torch.long, device=device) # (1,)

#     # ---------------------------------------------------------
#     # Normal replay
#     # ---------------------------------------------------------
#     model = ResNet9(num_classes=10).to(device)
#     model.load_state_dict(init_state_dict_cpu, strict=True)
#     opt = torch.optim.Adam(model.parameters(), lr=lr)

#     replay_loader = build_replay_loader(mixed_train, recorded_batches, device=device, num_workers=2)
#     it = iter(replay_loader)

#     model.train()
#     for _ in range(steps):
#         try:
#             x, y, gid, _src = next(it)
#         except StopIteration:
#             it = iter(replay_loader)
#             x, y, gid, _src = next(it)

#         x = x.to(device, non_blocking=True)
#         y = y.to(device, non_blocking=True)
#         gid = gid.to(device, non_blocking=True)

#         # --- counterfactual intervention: replace target example(s) in-batch ---
#         mask = (gid == int(target_gid))
#         if mask.any():
#             m = int(mask.sum().item())
#             x[mask] = x_rep.expand(m, -1, -1, -1)
#             y[mask] = y_rep.expand(m)
#             gid[mask] = int(rep_gid)

#         opt.zero_grad(set_to_none=True)
#         logits = model(x)
#         loss = F.cross_entropy(logits, y)   # mean over B (unchanged)
#         loss.backward()
#         opt.step()

#     R_cf = eval_selected_query_rewards(model, queries, device, reward, query_index=query_index)  # (10,) CPU
#     del model, opt, replay_loader, it
#     if device.type == "cuda":
#         torch.cuda.empty_cache()
#     gc.collect()
#     return R_cf

# -----------------------------
# TrueCF candidate pool builder: EXACTLY your requested logic
# -----------------------------

def build_truecf_pools_from_pdf_topk(
    scores_gnq: torch.Tensor,
    scores_tr: torch.Tensor,
    scores_tn: torch.Tensor,
    labels: torch.Tensor,
    topk_pdf: int,
    max_total: int,
    seed: int,
    beta_full: Optional[torch.Tensor] = None,
    cf_random_per_class: int = 0,
    cf_random_seed: int = 12345,
) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, List[int]], Dict[int, set], List[int]]:
    """
    Builds per-class candidate pools C_c from the SAME top-k lists used in the PDFs (topk_pdf),
    plus optional random outsiders per class (label-based), excluding estimator union for that class.

    Returns:
      pdf_top[c][method] : list of gids (exact PDF top-k per class per method)
      random_pool[c]     : list of random gids for class c (outside estimator union for that class)
      cand_pool[c]       : set of gids in C_c AFTER budget truncation update
      candidates         : global list of gids (union of all C_c) AFTER budget truncation
    """
    N = scores_gnq.shape[0]
    k = int(min(max(1, topk_pdf), N))
    budget = int(max(1, min(max_total, N)))

    # 1) exact PDF top-k lists per class per method
    pdf_top = {c: {} for c in range(10)}
    for c in range(10):
        pdf_top[c]["gnq"] = torch.topk(scores_gnq[:, c].abs(), k=k, largest=True).indices.tolist()
        pdf_top[c]["trace"] = torch.topk(scores_tr[:, c].abs(), k=k, largest=True).indices.tolist()
        pdf_top[c]["tracenorm"] = torch.topk(scores_tn[:, c].abs(), k=k, largest=True).indices.tolist()
        if beta_full is not None:
            pdf_top[c]["datamodel"] = torch.topk(beta_full[:, c].abs(), k=k, largest=True).indices.tolist()
        else:
            pdf_top[c]["datamodel"] = []

    # 2) random outsiders per class (label-based), excluding estimator union for that class
    rng = np.random.RandomState(int(seed) + int(cf_random_seed))
    random_pool = {c: [] for c in range(10)}

    if int(cf_random_per_class) > 0:
        for c in range(10):
            pool = (labels == c).nonzero(as_tuple=False).squeeze(1).cpu().numpy()
            if pool.size == 0:
                continue

            excl = set()
            for m in ["gnq", "trace", "tracenorm", "datamodel"]:
                excl |= set(pdf_top[c][m])

            if len(excl) > 0:
                pool = pool[~np.isin(pool, np.array(list(excl), dtype=np.int64))]

            if pool.size == 0:
                continue

            take = min(int(cf_random_per_class), int(pool.size))
            pick = rng.choice(pool, size=take, replace=False)
            random_pool[c] = [int(x) for x in pick.tolist()]

    # 3) per-class pools
    cand_pool = {c: set() for c in range(10)}
    for c in range(10):
        for m in ["gnq", "trace", "tracenorm", "datamodel"]:
            cand_pool[c] |= set(pdf_top[c][m])
        cand_pool[c] |= set(random_pool[c])

    # 4) budget with MUST-include = union of all estimator top-k across all classes/methods
    must = set()
    for c in range(10):
        for m in ["gnq","trace","tracenorm","datamodel"]:
            must |= set(pdf_top[c][m])

    must = sorted(must)

    if len(must) > budget:
        raise RuntimeError(
            f"Budget {budget} is smaller than MUST-include {len(must)} "
            f"(pdf_topk union). Increase --cf_max_total or reduce --topk/DM."
        )

    rest = []
    rest_set = set(must)
    for c in range(10):
        for gid in random_pool[c]:
            if gid not in rest_set:
                rest.append(gid); rest_set.add(gid)

    candidates = must + rest[: max(0, budget - len(must))]

    kept = set(candidates)
    for c in range(10):
        cand_pool[c] = cand_pool[c] & kept

    return pdf_top, random_pool, cand_pool, candidates


# -----------------------------
# Args
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./out_ggnq")

    p.add_argument("--n_mnist", type=int, default=20000)
    p.add_argument("--n_cifar", type=int, default=20000)

    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ckpt_every", type=int, default=200)

    p.add_argument("--queries_per_class", type=int, default=32)
    p.add_argument("--query_index", type=int, default=0,
                   help="Which query within each class to use as the SINGLE query for Rows 1–5.")
    p.add_argument("--reward", type=str, default="margin_mean_other",
                   choices=["neg_ce", "logp", "prob", "margin", "margin_mean_other"])

    p.add_argument("--lambda_reg", type=float, default=1e-2)
    p.add_argument("--alpha_center", action="store_true")

    p.add_argument("--topk", type=int, default=12)

    p.add_argument("--query_bs", type=int, default=8)  # kept for compatibility
    p.add_argument("--query_cache_device", type=str, default="cpu",
                   choices=["cpu", "cuda"])

    p.add_argument("--layers", type=str, default="all", choices=["last", "all"])
    p.add_argument("--q_chunk", type=int, default=64)

    p.add_argument("--measure_overhead_steps", type=int, default=200)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_npz", action="store_true")

    # --- DataModel (paper-style) ---
    p.add_argument("--dm_runs", type=int, default=0)
    p.add_argument("--dm_steps", type=int, default=300)
    p.add_argument("--dm_lr", type=float, default=1e-3)
    p.add_argument("--dm_batch_size", type=int, default=128)
    p.add_argument("--dm_p", type=float, default=0.3)
    p.add_argument("--dm_ridge", type=float, default=1e-2)
    p.add_argument("--dm_cache_device", type=str, default="cpu", choices=["cpu", "cuda"])

    # --- TrueCF (single-run counterfactual) ---
    p.add_argument("--cf_topk", type=int, default=0,
                   help="If >0, compute TrueCF for a candidate pool built from the SAME PDF top-k (args.topk) + random outsiders.")
    p.add_argument("--cf_max_total", type=int, default=200,
                   help="Hard cap on total counterfactual examples to replay (across all classes).")
    p.add_argument("--cf_deterministic", action="store_true",
                   help="Try to enforce deterministic ops for more faithful replay.")
    p.add_argument("--cf_eps", type=float, default=0.0,
                   help="If >0, use sum/(w.sum()+eps). If 0, use mean() to keep exact 1/B scaling.")
    p.add_argument("--cf_random_per_class", type=int, default=0,
                   help="Add this many RANDOM candidates per class (outside estimator union), if budget allows.")
    p.add_argument("--cf_random_seed", type=int, default=12345,
                   help="Seed for random exploration candidates (independent of --seed).")
    p.add_argument("--cf_rank", type=str, default="abs",
                   choices=["help", "hurt", "abs"],
                   help="How to rank TrueCF within the candidate pool: "
                        "help=largest positive (helping), hurt=most negative (hurting), abs=largest magnitude.")

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------

def main(args):
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device(args.device)
    cache_device = torch.device(args.query_cache_device)

    mixed_train, mnist_test = build_datasets(args.data_root, args.n_mnist, args.n_cifar)
    N = len(mixed_train)

    labels = torch.empty(N, dtype=torch.long)
    srcs = torch.empty(N, dtype=torch.long)
    for gid in range(N):
        _, y, _, src = mixed_train[gid]
        labels[gid] = y
        srcs[gid] = src

    queries = sample_queries_per_class(mnist_test, args.queries_per_class, seed=args.seed)

    # hard-check query_index
    for c in range(10):
        qx, _qy = queries[c]
        if not (0 <= args.query_index < qx.size(0)):
            raise ValueError(f"--query_index {args.query_index} out of range for class {c} with queries_per_class={qx.size(0)}")

    model = ResNet9(num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Save init weights for TrueCF replay (store on CPU to save GPU memory)
    init_state_dict_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    loader = DataLoader(
        mixed_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    it = iter(loader)

    hooks = GhostHooks(model)

    # Will be set if TrueCF runs (needed so Row-5 ranks within C_c)
    cand_pool_by_class = None
    pdf_top_by_class = None
    random_pool_by_class = None

    # -------------------------
    # Build query cache (SINGLE query per class)
    # -------------------------
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    if args.layers == "last":
        print("\nBuilding query cache (LAST layer) for SINGLE query per class...")
        t0 = time.perf_counter()
        qcache_last = build_query_cache_lastlayer_single_query_per_class(
            model=model,
            queries=queries,
            device=device,
            reward=args.reward,
            query_index=args.query_index,
            cache_device=cache_device,
        )
        t1 = time.perf_counter()
        print(f"Query cache built: Q={qcache_last.feat_q.size(0)} (should be 10) "
              f"D={qcache_last.feat_q.size(1)} on {cache_device} in {t1-t0:.2f}s")
        qcache_all = None
        ggnq_buf = None
    else:
        print("\nBuilding query cache (ALL layers) for SINGLE query per class...")
        t0 = time.perf_counter()
        qcache_all = build_query_cache_alllayers_single_query_per_class(
            model=model,
            hooks=hooks,
            queries=queries,
            device=device,
            reward=args.reward,
            query_index=args.query_index,
            cache_device=cache_device,
        )
        t1 = time.perf_counter()
        some = hooks.specs[-1].name
        print(f"Query cache built: Q={qcache_all.cls_index.numel()} (should be 10) layers={len(hooks.specs)} "
              f"(e.g., {some}: a{tuple(qcache_all.per_layer_a[some].shape)} d{tuple(qcache_all.per_layer_d[some].shape)}) "
              f"on {cache_device} in {t1-t0:.2f}s")
        qcache_last = None

        B = args.batch_size
        Q = int(qcache_all.cls_index.numel())
        ggnq_buf = {
            "K_total":  torch.empty((B, B), device=device, dtype=torch.float32),
            "Kq_total": torch.empty((B, Q), device=device, dtype=torch.float32),
            "I":        torch.eye(B, device=device, dtype=torch.float32),
        }

    # -------------------------
    # Overhead measurement (optional)
    # -------------------------
    if args.measure_overhead_steps > 0:
        print("\n" + "="*100)
        print(f"Overhead measurement: no attribution vs with attribution (GNQ + TraceIn + TraceIn-Norm) ({args.layers} mode)")
        print("="*100)

        model.train()
        warm = 10
        for _ in range(warm):
            try:
                x, y, gid, _ = next(it)
            except StopIteration:
                it = iter(loader)
                x, y, gid, _ = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            hooks.clear()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

        times0 = []
        for _ in range(args.measure_overhead_steps):
            try:
                x, y, gid, _ = next(it)
            except StopIteration:
                it = iter(loader)
                x, y, gid, _ = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            sync_cuda(device)
            ta = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            hooks.clear()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            sync_cuda(device)
            tb = time.perf_counter()
            times0.append(tb - ta)

        times1 = []
        for _ in range(args.measure_overhead_steps):
            try:
                x, y, gid, _ = next(it)
            except StopIteration:
                it = iter(loader)
                x, y, gid, _ = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            sync_cuda(device)
            ta = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            hooks.clear()
            if args.layers == "last":
                logits, feat = model(x, return_feat=True)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                _ag, _at, _an = batch_alphas_lastlayer_gnq_trace_tracenorm(
                    feat_tr=feat.detach(),
                    logits_tr=logits.detach(),
                    y_tr=y,
                    qcache=qcache_last,
                    lambda_reg=args.lambda_reg,
                    alpha_center=args.alpha_center,
                    cache_device=cache_device,
                    device=device,
                )
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                _ag, _at, _an = batch_alphas_alllayers_gnq_trace_tracenorm(
                    hooks=hooks,
                    qcache=qcache_all,
                    lambda_reg=args.lambda_reg,
                    alpha_center=args.alpha_center,
                    cache_device=cache_device,
                    device=device,
                    q_chunk=args.q_chunk,
                    ggnq_buf=ggnq_buf,
                )
            opt.step()

            sync_cuda(device)
            tb = time.perf_counter()
            times1.append(tb - ta)

        m0 = float(np.mean(times0))
        m1 = float(np.mean(times1))
        print(f"avg iter time (no attribution):            {m0:.6f} s")
        print(f"avg iter time (with 3-way attribution):   {m1:.6f} s")
        print(f"overhead factor:                           {m1/m0:.3f}x   {mem_report(device)}")
        print("="*100 + "\n")

    # -------------------------
    # Training + attribution
    # -------------------------
    scores_gnq = torch.zeros((N, 10), dtype=torch.float32)   # CPU
    scores_tr  = torch.zeros((N, 10), dtype=torch.float32)   # CPU
    scores_tn  = torch.zeros((N, 10), dtype=torch.float32)   # CPU
    seen_counts = torch.zeros((N,), dtype=torch.long)        # CPU

    # Record realized batch sequence (for TrueCF replay)
    recorded_batches: List[List[int]] = []

    R_start = eval_selected_query_rewards(model, queries, device, args.reward, query_index=args.query_index)
    print(f"[ckpt     0] rewards (SINGLE query per class; query_index={args.query_index}) reward={args.reward}")
    print("  " + "  ".join([f"R(class{c})={R_start[c].item():+.4e}" for c in [0,1,8]]))

    step = 0
    while step < args.steps:
        step += 1
        try:
            x, y, gid, _src = next(it)
        except StopIteration:
            it = iter(loader)
            x, y, gid, _src = next(it)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        gid_cpu = gid.to("cpu", non_blocking=True)

        # record this exact batch’s gids
        recorded_batches.append(gid_cpu.tolist())

        model.train()
        opt.zero_grad(set_to_none=True)
        hooks.clear()

        if args.layers == "last":
            logits, feat = model(x, return_feat=True)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            alpha_gnq, alpha_tr, alpha_tn = batch_alphas_lastlayer_gnq_trace_tracenorm(
                feat_tr=feat.detach(),
                logits_tr=logits.detach(),
                y_tr=y,
                qcache=qcache_last,
                lambda_reg=args.lambda_reg,
                alpha_center=args.alpha_center,
                cache_device=cache_device,
                device=device,
            )
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            alpha_gnq, alpha_tr, alpha_tn = batch_alphas_alllayers_gnq_trace_tracenorm(
                hooks=hooks,
                qcache=qcache_all,
                lambda_reg=args.lambda_reg,
                alpha_center=args.alpha_center,
                cache_device=cache_device,
                device=device,
                q_chunk=args.q_chunk,
                ggnq_buf=ggnq_buf,
            )

        scores_gnq.index_add_(0, gid_cpu, alpha_gnq)
        scores_tr.index_add_(0,  gid_cpu, alpha_tr)
        scores_tn.index_add_(0,  gid_cpu, alpha_tn)
        seen_counts.index_add_(0, gid_cpu, torch.ones_like(gid_cpu, dtype=torch.long))

        opt.step()

        if (step % args.ckpt_every) == 0:
            R_cur = eval_selected_query_rewards(model, queries, device, args.reward, query_index=args.query_index)
            print(f"[ckpt {step:5d}] " + "  ".join([f"R(class{c})={R_cur[c].item():+.4e}" for c in [0,1,8]]))

    R_final = eval_selected_query_rewards(model, queries, device, args.reward, query_index=args.query_index)
    dR_total = R_final - R_start
    print("\nNet reward change ΔR_total (classes 0,1,8): " +
          "  ".join([f"{c}:{dR_total[c].item():+.3e}" for c in [0,1,8]]))

    # -------------------------
    # DataModel (paper-style, optional)
    # -------------------------
    beta_full = None  # (N,10) CPU
    beta0 = None      # (10,) CPU
    if args.dm_runs > 0:
        print("\n" + "="*100)
        print("Running DataModel (paper p-fraction subsets + dual ridge) [SINGLE query per class]")
        print("="*100)

        dm_cache_device = torch.device(args.dm_cache_device)

        beta_full, beta0 = run_datamodel_p_fraction(
            mixed_train=mixed_train,
            queries=queries,
            reward=args.reward,
            device=device,
            seed=args.seed,
            runs=args.dm_runs,
            steps=args.dm_steps,
            lr=args.dm_lr,
            batch_size=args.dm_batch_size,
            p=args.dm_p,
            ridge=args.dm_ridge,
            dm_cache_device=dm_cache_device,
            query_index=args.query_index,
        )

        print("="*100 + "\n")

    # -------------------------
    # True single-run counterfactual (optional)
    # -------------------------
    causal_full = None  # (N,10) CPU (NaN for non-evaluated)
    truecf_candidates: Optional[List[int]] = None

    if args.cf_topk > 0:
        print("\n" + "="*100)
        print("Running TRUE single-run counterfactual replay (run-conditioned causality) [SINGLE query per class]")
        print("="*100)

        # Build C_c from EXACT PDF top-k lists (args.topk) plus random outsiders
        pdf_top, random_pool, cand_pool, candidates = build_truecf_pools_from_pdf_topk(
            scores_gnq=scores_gnq,
            scores_tr=scores_tr,
            scores_tn=scores_tn,
            beta_full=beta_full,
            labels=labels,
            topk_pdf=int(args.topk),
            max_total=int(args.cf_max_total),
            seed=int(args.seed),
            cf_random_per_class=int(args.cf_random_per_class),
            cf_random_seed=int(args.cf_random_seed),
        )
        truecf_candidates = candidates
        cand_pool_by_class = cand_pool
        pdf_top_by_class = pdf_top
        random_pool_by_class = random_pool

        # Sanity check: every PDF top-k id MUST be in TrueCF candidates
        cand_set = set(truecf_candidates)
        ok_all = True
        for c in range(10):
            for m in ["gnq", "trace", "tracenorm", "datamodel"]:
                miss = [gid for gid in pdf_top_by_class[c][m] if gid not in cand_set]
                if miss:
                    ok_all = False
                    print(f"MISSING class {c} {m} count {len(miss)} examples {miss[:5]}")
        print("ALL_IN_CANDIDATES =", ok_all)

        print(f"[TrueCF] candidates_kept={len(candidates)} | budget={int(args.cf_max_total)} | pdf_topk={int(args.topk)} | random_per_class={int(args.cf_random_per_class)}")
        for c_show in [0, 1, 8]:
            print(f"[TrueCF] |C_{c_show}|={len(cand_pool_by_class[c_show])}  random={len(random_pool_by_class[c_show])}")

        # Store NaN for non-evaluated points so later top-k doesn't get polluted by zeros
        causal_full = torch.full((N, 10), float("nan"), dtype=torch.float32)

        # effect positive means "helping":
        #   effect = R_final - R_cf  (positive => removing point makes reward worse => point was helping)
        for idx, gid_i in enumerate(candidates):
            R_cf = run_single_run_counterfactual_replay(
                mixed_train=mixed_train,
                recorded_batches=recorded_batches,
                queries=queries,
                reward=args.reward,
                device=device,
                seed=args.seed + 555,
                steps=args.steps,
                lr=args.lr,
                target_gid=int(gid_i),
                init_state_dict_cpu=init_state_dict_cpu,
                cf_deterministic=args.cf_deterministic,
                cf_eps=args.cf_eps,
                query_index=args.query_index,
            )
            effect = (R_final - R_cf).to(torch.float32).cpu()
            causal_full[int(gid_i)] = effect

            if (idx + 1) % 10 == 0 or (idx + 1) == len(candidates):
                print(f"[TrueCF] {idx+1:4d}/{len(candidates)} done. last_gid={gid_i}  |effect|={float(effect.norm().item()):.3e}")

        print("="*100 + "\n")

    # -------------------------
    # PDFs
    # -------------------------
    model.eval()
    for c in range(10):
        qx, qy = queries[c]
        qx0 = qx[args.query_index].to(device)
        with torch.no_grad():
            log0 = model(qx0.unsqueeze(0))
            p_ = F.softmax(log0, dim=1)[0]
            pred = int(p_.argmax().item())
            conf = float(p_.max().item())

        sc_g = scores_gnq[:, c]
        sc_t = scores_tr[:, c]
        sc_n = scores_tn[:, c]
        k = min(args.topk, N)

        top_gnq = torch.topk(sc_g.abs(), k=k, largest=True).indices.tolist()
        top_tr  = torch.topk(sc_t.abs(), k=k, largest=True).indices.tolist()
        top_tn  = torch.topk(sc_n.abs(), k=k, largest=True).indices.tolist()

        out_pdf = os.path.join(args.out_dir, f"class_{c}.pdf")
        title_suffix = (
            f"query_index={args.query_index} | layers={args.layers} | reward={args.reward} | steps={args.steps} ckpt_every={args.ckpt_every} "
            f"bs={args.batch_size} | lambda_reg={args.lambda_reg} | alpha_center={args.alpha_center}"
        )

        if (beta_full is not None) and (causal_full is not None):
            sc_dm = beta_full[:, c]
            top_dm = torch.topk(sc_dm.abs(), k=k, largest=True).indices.tolist()

            # --- Row-5: Top-k TrueCF STRICTLY within C_c ---
            sc_cf = causal_full[:, c]
            pool_c = []
            if cand_pool_by_class is not None:
                pool_c = sorted(list(cand_pool_by_class[c]))

            if len(pool_c) == 0:
                top_cf = []
            else:
                vals = sc_cf[pool_c]
                ok = ~torch.isnan(vals)
                ok_idx = ok.nonzero(as_tuple=False).squeeze(1)
                pool_ok = [pool_c[i] for i in ok_idx.tolist()]
                vals_ok = vals[ok]

                kk = min(k, len(pool_ok))
                if kk == 0:
                    top_cf = []
                else:
                    if args.cf_rank == "help":
                        idx = torch.topk(vals_ok, k=kk, largest=True).indices
                    elif args.cf_rank == "hurt":
                        idx = torch.topk(vals_ok, k=kk, largest=False).indices
                    else:  # abs
                        idx = torch.topk(vals_ok.abs(), k=kk, largest=True).indices
                    top_cf = [pool_ok[i] for i in idx.tolist()]

            if (random_pool_by_class is not None) and len(top_cf) > 0:
                cf_set = set(top_cf)
                rand_hit = len(cf_set & set(random_pool_by_class[c]))
                print(f"[Class {c}] TrueCF-top{len(top_cf)} random_outside_estimators: {rand_hit}/{len(top_cf)}")

            make_class_pdf_compare5(
                out_path=out_pdf,
                cls=c,
                mixed_ds=mixed_train,
                query_img=qx0.detach().cpu(),
                query_pred=pred,
                query_conf=conf,
                top_gnq=top_gnq,
                top_tr=top_tr,
                top_tn=top_tn,
                top_dm=top_dm,
                top_cf=top_cf,
                scores_gnq_c=sc_g,
                scores_tr_c=sc_t,
                scores_tn_c=sc_n,
                scores_dm_c=sc_dm,
                scores_cf_c=sc_cf,
                seen_counts=seen_counts,
                title_suffix=title_suffix + f" | DataModel(runs={args.dm_runs}, steps={args.dm_steps}) | TrueCF(rank={args.cf_rank})",
                layers_mode=args.layers,
            )

        elif (beta_full is None) and (causal_full is not None):
            sc_cf = causal_full[:, c]

            pool_c = []
            if cand_pool_by_class is not None:
                pool_c = sorted(list(cand_pool_by_class[c]))

            if len(pool_c) == 0:
                top_cf = []
            else:
                vals = sc_cf[pool_c]
                ok = ~torch.isnan(vals)
                ok_idx = ok.nonzero(as_tuple=False).squeeze(1)
                pool_ok = [pool_c[i] for i in ok_idx.tolist()]
                vals_ok = vals[ok]

                kk = min(k, len(pool_ok))
                if kk == 0:
                    top_cf = []
                else:
                    if args.cf_rank == "help":
                        idx = torch.topk(vals_ok, k=kk, largest=True).indices
                    elif args.cf_rank == "hurt":
                        idx = torch.topk(vals_ok, k=kk, largest=False).indices
                    else:
                        idx = torch.topk(vals_ok.abs(), k=kk, largest=True).indices
                    top_cf = [pool_ok[i] for i in idx.tolist()]

            if (random_pool_by_class is not None) and len(top_cf) > 0:
                cf_set = set(top_cf)
                rand_hit = len(cf_set & set(random_pool_by_class[c]))
                print(f"[Class {c}] TrueCF-top{len(top_cf)} random_outside_estimators: {rand_hit}/{len(top_cf)}")

            make_class_pdf_compare4_truecf(
                out_path=out_pdf,
                cls=c,
                mixed_ds=mixed_train,
                query_img=qx0.detach().cpu(),
                query_pred=pred,
                query_conf=conf,
                top_gnq=top_gnq,
                top_tr=top_tr,
                top_tn=top_tn,
                top_cf=top_cf,
                scores_gnq_c=sc_g,
                scores_tr_c=sc_t,
                scores_tn_c=sc_n,
                scores_cf_c=sc_cf,
                seen_counts=seen_counts,
                title_suffix=title_suffix + f" | TrueCF(rank={args.cf_rank})",
                layers_mode=args.layers,
            )

        elif beta_full is not None:
            sc_dm = beta_full[:, c]
            top_dm = torch.topk(sc_dm.abs(), k=k, largest=True).indices.tolist()
            make_class_pdf_compare4(
                out_path=out_pdf,
                cls=c,
                mixed_ds=mixed_train,
                query_img=qx0.detach().cpu(),
                query_pred=pred,
                query_conf=conf,
                top_gnq=top_gnq,
                top_tr=top_tr,
                top_tn=top_tn,
                top_dm=top_dm,
                scores_gnq_c=sc_g,
                scores_tr_c=sc_t,
                scores_tn_c=sc_n,
                scores_dm_c=sc_dm,
                seen_counts=seen_counts,
                title_suffix=title_suffix + f" | DataModel(runs={args.dm_runs}, steps={args.dm_steps})",
                layers_mode=args.layers,
            )
        else:
            make_class_pdf_compare3(
                out_path=out_pdf,
                cls=c,
                mixed_ds=mixed_train,
                query_img=qx0.detach().cpu(),
                query_pred=pred,
                query_conf=conf,
                top_gnq=top_gnq,
                top_tr=top_tr,
                top_tn=top_tn,
                scores_gnq_c=sc_g,
                scores_tr_c=sc_t,
                scores_tn_c=sc_n,
                seen_counts=seen_counts,
                title_suffix=title_suffix,
                layers_mode=args.layers,
            )

        print(f"Saved: {out_pdf}")

    if args.save_npz:
        npz_path = os.path.join(args.out_dir, "attrib_compare_outputs.npz")
        payload = dict(
            scores_gnq=scores_gnq.numpy(),
            scores_trace=scores_tr.numpy(),
            scores_trace_norm=scores_tn.numpy(),
            labels=labels.numpy(),
            srcs=srcs.numpy(),
            seen_counts=seen_counts.numpy(),
            R_start=R_start.numpy(),
            R_final=R_final.numpy(),
            query_index=np.array([args.query_index], dtype=np.int64),
        )
        if beta_full is not None:
            payload["beta_datamodel"] = beta_full.numpy()
            if beta0 is not None:
                payload["beta0_datamodel"] = beta0.numpy()
        if causal_full is not None:
            payload["truecf_effect"] = causal_full.numpy()
        if truecf_candidates is not None:
            payload["truecf_candidates"] = np.array(truecf_candidates, dtype=np.int64)

        np.savez_compressed(npz_path, **payload)
        print(f"Saved: {npz_path}")

    hooks.close()
    print("\nDone.")
    print(f"PDFs are in: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
