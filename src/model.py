import os
import glob
import math
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

import os
import glob
import math
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

# --- Core imports ---
import os, glob, math, random, json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

# --- Device setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_float32_matmul_precision("high")

print(f"Using device: {device}")


class NPYLayerActivationStream(IterableDataset):
    """
    Iterable PyTorch dataset for per-layer GraphCast activations
    saved as .npy arrays (one timestep per file).

    Automatically handles both [N, D] and [N, 1, D] shapes.
    """

    def __init__(self, data_dirs, layer_prefix="layer0012_", d_in=512,
                 batch_size=8192, steps_per_epoch=None, seed=0):
        super().__init__()

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        # --- Collect matching files ---
        file_list = []
        for root in data_dirs:
            pattern = os.path.join(root, f"{layer_prefix}*.npy")
            file_list.extend(glob.glob(pattern))
        file_list = sorted(file_list)
        if not file_list:
            raise FileNotFoundError(f"No .npy files found matching {layer_prefix} in {data_dirs}")

        self.files = file_list
        self.d_in = d_in
        self.batch = batch_size
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch

        # --- Compute metadata ---
        self.file_meta = []
        for f in self.files:
            try:
                arr = np.load(f, mmap_mode="r")
            except ValueError as e:
                print(f"[CORRUPT] {f}: {e}")
                continue
            if arr.ndim == 3 and arr.shape[1] == 1:
                arr = arr[:, 0, :]           # ✅ fix singleton dimension
            assert arr.ndim == 2 and arr.shape[1] == d_in, f"{f} has wrong shape {arr.shape}"
            self.file_meta.append({"path": f, "n_nodes": int(arr.shape[0])})

        total_nodes = sum(m["n_nodes"] for m in self.file_meta)
        self.total_batches = math.ceil(total_nodes / self.batch)
        if self.steps_per_epoch is None:
            self.steps_per_epoch = self.total_batches

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        nw = worker.num_workers if worker else 1
        wid = worker.id if worker else 0
        rng = np.random.default_rng(self.seed + 997 * wid)

        file_shard = self.file_meta[wid::nw]
        rng.shuffle(file_shard)

        max_batches_worker = math.ceil(self.steps_per_epoch / nw)
        batches_yielded = 0

        for md in file_shard:
            if batches_yielded >= max_batches_worker:
                break

            fpath = md["path"]
            X = np.load(fpath, mmap_mode="r")
            if X.ndim == 3 and X.shape[1] == 1:
                X = X[:, 0, :]                # ✅ fix again at runtime

            n = X.shape[0]
            perm = rng.permutation(n)
            for start in range(0, n, self.batch):
                if batches_yielded >= max_batches_worker:
                    return
                sel = perm[start:start + self.batch]
                if sel.size == 0:
                    break
                xb = torch.from_numpy(X[sel, :])
                yield xb
                batches_yielded += 1
def topk(x, k: int):
    """Keep top-k per row, zero out the rest."""
    if k >= x.shape[1]:
        return x
    vals, idx = torch.topk(x, k, dim=1)
    mask = torch.zeros_like(x)
    mask.scatter_(1, idx, 1.0)
    return x * mask


class SAE(nn.Module):
    """Top-K Sparse Autoencoder with AuxK loss (OpenAI recipe)."""
    def __init__(self, d_in, latent, k_active, k_aux,
                 unit_norm_decoder=True, dead_window=3_000_000):
        super().__init__()
        # --- Core layers (no internal bias terms) ---
        self.enc = nn.Linear(d_in, latent, bias=False)
        self.dec = nn.Linear(latent, d_in, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(d_in))  # shared pre/post bias

        # --- Hyperparameters ---
        self.k = k_active
        self.k_aux = k_aux
        self.unit_norm_decoder = unit_norm_decoder
        self.dead_window = dead_window
        self.eps = 1e-8

        # --- Dead neuron tracking ---
        self.register_buffer("miss_counts", torch.zeros(latent, dtype=torch.long))
        self.register_buffer("dead_mask", torch.zeros(latent, dtype=torch.bool))

        # --- OpenAI-style initialization ---
        with torch.no_grad():
            # 1. Randomly initialize decoder (dictionary) and normalize columns
            torch.nn.init.normal_(self.dec.weight, mean=0.0, std=1.0)
            W = self.dec.weight
            W.div_(W.norm(dim=0, keepdim=True).clamp_min(1e-8))

            # 2. Set encoder = decoderᵀ
            self.enc.weight.copy_(W.t())

            # 3. Zero shared bias
            self.b_pre.zero_()

    def _renorm_decoder_columns_(self):
        """Ensure each decoder column has unit norm."""
        W = self.dec.weight.data
        norms = W.norm(dim=0, keepdim=True).clamp_min(self.eps)
        W.div_(norms)

    def forward(self, x):
        # ---- Normalize inputs (zero-mean, unit-norm per sample) ----
        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)

        # ---- Subtract shared pre-bias before encoding ----
        x_bar = x - self.b_pre

        # ---- Encode ----
        code_pre = torch.relu(self.enc(x_bar))
        code = topk(code_pre, self.k)

        # ---- Decode and add shared bias back ----
        if self.unit_norm_decoder:
            W = self.dec.weight
            norms = W.norm(dim=0, keepdim=True).clamp_min(self.eps)
            recon = torch.addmm(self.b_pre, code, (W / norms).t())
        else:
            recon = self.dec(code) + self.b_pre

        # ---- AuxK reconstruction (dead latents only) ----
        dead_mask = self.dead_mask
        if dead_mask.any():
            dead_code = code_pre * dead_mask.unsqueeze(0)
            aux_code = topk(dead_code, min(self.k_aux, dead_code.shape[1]))
            aux_recon = torch.addmm(
                torch.zeros_like(self.b_pre),
                aux_code,
                self.dec.weight.t(),
            )
        else:
            aux_recon = torch.zeros_like(x)

        return recon, code, aux_recon

    @torch.no_grad()
    def update_dead_mask(self, code, batch_size: int):
        """Update miss_counts and dead_mask each batch."""
        active = (code > 0).any(dim=0).cpu()
        self.miss_counts[active] = 0
        self.miss_counts[~active] += batch_size
        self.dead_mask = self.miss_counts >= self.dead_window


@torch.no_grad()
def _project_decoder_grads_orthogonal(model):
    """Project decoder gradients so they don't change column norms."""
    W = model.dec.weight
    G = model.dec.weight.grad
    if G is None:
        return
    dots = (G * W).sum(dim=0, keepdim=True)
    norms2 = (W * W).sum(dim=0, keepdim=True).clamp_min(1e-8)
    G.sub_((dots / norms2) * W)


@torch.no_grad()
def _renorm_decoder_columns_(model):
    """Ensure each decoder column has unit L2 norm."""
    W = model.dec.weight.data
    norms = W.norm(dim=0, keepdim=True).clamp_min(1e-8)
    W.div_(norms)

# for use with JAX model

# ---------- Convert PyTorch SAE to SAEStaticParams ----------
import torch
import numpy as np
from typing import Tuple
import sys, os

@dataclass
class SAEStaticParams:
    enc_w: jnp.ndarray      # [d_in, latent]
    dec_w: jnp.ndarray      # [latent, d_in]
    b_pre: jnp.ndarray      # [d_in]
    k_active: int
    unit_norm_decoder: bool

def load_sae_params_from_torch(ckpt_path: str, unit_norm_decoder: bool, k_active: int):
    import torch, jax.numpy as jnp
    from dataclasses import dataclass

    @dataclass
    class SAEStaticParams:
        enc_w: jnp.ndarray
        dec_w: jnp.ndarray
        b_pre: jnp.ndarray
        k_active: int
        unit_norm_decoder: bool

    # --- load and unwrap nested dict ---
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state" in state:
        state = state["model_state"]

    # --- extract weights ---
    enc_w_torch = state["enc.weight"]    # [latent, d_in]
    dec_w_torch = state["dec.weight"]    # [d_in, latent]
    b_pre_torch = state["b_pre"]         # [d_in]

    # --- transpose and convert to numpy/JAX ---
    enc_w = enc_w_torch.t().contiguous().cpu().numpy()  # [d_in, latent]
    dec_w = dec_w_torch.t().contiguous().cpu().numpy()  # [latent, d_in]
    b_pre = b_pre_torch.contiguous().cpu().numpy()      # [d_in]

    return SAEStaticParams(
        enc_w=jnp.asarray(enc_w),
        dec_w=jnp.asarray(dec_w),
        b_pre=jnp.asarray(b_pre),
        k_active=k_active,
        unit_norm_decoder=unit_norm_decoder,
    )
