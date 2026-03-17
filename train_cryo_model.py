#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Finetune the *original* Cryo_Model (protein EM heads only) on your own data.

This script trains the 3D UNets used by AutoEM:
  - BB_model: backbone voxel classification (expects 4-channel logits)
  - CA_model: Cα voxel classification (expects 4-channel logits)
  - AA_model: amino-acid type classification (expects 21-channel logits: bg + 20 AA)

It is designed to be minimally invasive:
  - keeps the existing architecture unchanged
  - loads 'cryo_model' weights from the original checkpoint
  - saves a new checkpoint with ONLY 'cryo_model' replaced (backbone/design kept)

Train list format (txt):
  Each non-empty, non-comment line:
    <map_path> <pdb_path>
  - map_path: .mrc / .map / .map.gz
  - pdb_path: protein structure in the same reference frame as the map

Example:
  python train_cryo_model.py \
    --train_list data/train_list.txt \
    --pretrained models/model_weight.pth \
    --save_path models/model_weight_finetune_cryo.pth \
    --epochs 3 --lr 1e-4 --device cuda
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import mrcfile
import numpy as np
import torch
import torch.nn as nn
from Bio.PDB import PDBParser
from torch.utils.data import DataLoader, Dataset

from cryo_module.Cryo_model import Cryo_Model
import cryo_module.utils as em_utils


@dataclass(frozen=True)
class LabelConfig:
    """
    The original ResUNet3D4EM uses 4 output channels for BB/CA.
    AutoEM inference keeps channel-0 and channels-2..end, and uses the (new) channel-2
    as the positive probability map.

    To stay compatible without changing architecture, we train *binary* BB/CA labels by
    mapping:
      - background voxels -> class 0
      - positive voxels   -> class 2
    This makes CrossEntropyLoss work with 4-channel logits.
    """

    bb_bg_class: int = 0
    bb_pos_class: int = 2
    ca_bg_class: int = 0
    ca_pos_class: int = 2

    # For AA head we keep the default convention: 0=bg, 1..20=AA type
    aa_bg_class: int = 0

    # Neighborhood (in voxels) to "thicken" labels around anchor atoms
    radius_ca: int = 1
    radius_bb: int = 1
    radius_aa: int = 1

    # Which backbone atoms to include for BB labels
    bb_atoms: tuple[str, ...] = ("N", "CA", "C", "O")


def _iter_cube(center: tuple[int, int, int], radius: int, shape: tuple[int, int, int]):
    cx, cy, cz = center
    D, H, W = shape
    x0, x1 = max(cx - radius, 0), min(cx + radius, D - 1)
    y0, y1 = max(cy - radius, 0), min(cy + radius, H - 1)
    z0, z1 = max(cz - radius, 0), min(cz + radius, W - 1)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            for z in range(z0, z1 + 1):
                yield x, y, z


def build_voxel_labels_from_pdb(
    pdb_path: str,
    grid_shape: tuple[int, int, int],
    offset: list[float],
    *,
    voxel_size: float = 1.0,
    cfg: LabelConfig | None = None,
):
    """
    Build training labels aligned to normEM (output of em_utils.processEMData).

    Returns:
      bb_label: LongTensor [D,H,W] values in {0,2}
      ca_label: LongTensor [D,H,W] values in {0,2}
      aa_label: LongTensor [D,H,W] values in {0..20}

    IMPORTANT: This is a practical "first working" labeler. For best results you will
    want to validate the coordinate mapping for your data.
    """
    if cfg is None:
        cfg = LabelConfig()

    D, H, W = grid_shape
    bb_label = torch.full((D, H, W), cfg.bb_bg_class, dtype=torch.long)
    ca_label = torch.full((D, H, W), cfg.ca_bg_class, dtype=torch.long)
    aa_label = torch.full((D, H, W), cfg.aa_bg_class, dtype=torch.long)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_path)

    AA_TYPES = em_utils.AA_types  # e.g. {"ALA":1, ... "TYR":20}

    def coord_to_grid(xyz: np.ndarray) -> tuple[int, int, int]:
        # xyz is in Å. After processEMData, the map is rescaled to ~1 Å/voxel.
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        gx = int(round(x / voxel_size)) - int(offset[0])
        gy = int(round(y / voxel_size)) - int(offset[1])
        gz = int(round(z / voxel_size)) - int(offset[2])
        return gx, gy, gz

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                aa_idx = AA_TYPES.get(resname)

                # CA label and AA label at CA
                if "CA" in residue:
                    gx, gy, gz = coord_to_grid(residue["CA"].coord)
                    if 0 <= gx < D and 0 <= gy < H and 0 <= gz < W:
                        for x, y, z in _iter_cube((gx, gy, gz), cfg.radius_ca, grid_shape):
                            ca_label[x, y, z] = cfg.ca_pos_class
                        if aa_idx is not None:
                            for x, y, z in _iter_cube((gx, gy, gz), cfg.radius_aa, grid_shape):
                                aa_label[x, y, z] = int(aa_idx)

                # Backbone label near N/CA/C/O
                for atom_name in cfg.bb_atoms:
                    if atom_name not in residue:
                        continue
                    gx, gy, gz = coord_to_grid(residue[atom_name].coord)
                    if 0 <= gx < D and 0 <= gy < H and 0 <= gz < W:
                        for x, y, z in _iter_cube((gx, gy, gz), cfg.radius_bb, grid_shape):
                            bb_label[x, y, z] = cfg.bb_pos_class

    return bb_label, ca_label, aa_label


class CryoTrainDataset(Dataset):
    """
    Each sample returns:
      em: FloatTensor [1,D,H,W]
      bb: LongTensor  [D,H,W] in {0,2}
      ca: LongTensor  [D,H,W] in {0,2}
      aa: LongTensor  [D,H,W] in {0..20}
    """

    def __init__(
        self,
        list_path: str,
        *,
        voxel_size: float = 1.0,
        cfg: LabelConfig | None = None,
        crop_size: int = 64,
        patches_per_volume: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        self.samples: list[tuple[str, str]] = []
        self.voxel_size = voxel_size
        self.cfg = cfg or LabelConfig()
        self.crop_size = int(crop_size)
        self.patches_per_volume = int(patches_per_volume)
        self.rng = np.random.default_rng(seed)

        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                self.samples.append((parts[0], parts[1]))

        if not self.samples:
            raise ValueError(f"No valid lines found in {list_path}")

    def __len__(self):
        return len(self.samples) * self.patches_per_volume

    def __getitem__(self, idx: int):
        # 使用循环来确保如果遇到坏图，可以自动尝试下一个样本
        attempt = 0
        current_idx = idx
        
        while attempt < len(self.samples):
            vol_idx = current_idx % len(self.samples)
            map_path, pdb_path = self.samples[vol_idx]

            try:
                # 1. 尝试读取和预处理地图
                with mrcfile.open(map_path, permissive=True) as EMmap:
                    # 如果这行报错，会被下面的 except 捕获
                    normEM, offset = em_utils.processEMData(EMmap)

                # 2. 尝试生成标签
                bb, ca, aa = build_voxel_labels_from_pdb(
                    pdb_path,
                    grid_shape=normEM.shape,
                    offset=offset,
                    voxel_size=self.voxel_size,
                    cfg=self.cfg,
                )

                D, H, W = normEM.shape
                cs = self.crop_size
                
                # 3. 随机采样重试逻辑（确保 patch 内部也有信号）
                max_patch_retries = 20
                for _ in range(max_patch_retries):
                    x0 = int(self.rng.integers(0, D - cs + 1))
                    y0 = int(self.rng.integers(0, H - cs + 1))
                    z0 = int(self.rng.integers(0, W - cs + 1))

                    # 检查切片是否有信号
                    if torch.any(ca[x0:x0+cs, y0:y0+cs, z0:z0+cs] > 0):
                        break
                
                # 提取切片
                em_patch = normEM[x0 : x0 + cs, y0 : y0 + cs, z0 : z0 + cs]
                em = torch.from_numpy(em_patch.astype(np.float32)).unsqueeze(0)
                bb_patch = bb[x0 : x0 + cs, y0 : y0 + cs, z0 : z0 + cs]
                ca_patch = ca[x0 : x0 + cs, y0 : y0 + cs, z0 : z0 + cs]
                aa_patch = aa[x0 : x0 + cs, y0 : y0 + cs, z0 : z0 + cs]

                return em, bb_patch, ca_patch, aa_patch

            except (ValueError, RuntimeError, Exception) as e:
                # 打印错误并尝试下一个样本
                print(f"Warning: Skipping corrupted or empty data {map_path} due to: {e}")
                current_idx = (current_idx + 1) % (len(self.samples) * self.patches_per_volume)
                attempt += 1
                continue

        # 如果所有数据都尝试一遍都失败了（极端情况）
        raise RuntimeError("All samples in the dataset are invalid or empty.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_list", type=str, required=True, help="txt file: '<map_path> <pdb_path>' per line")
    p.add_argument("--pretrained", type=str, required=True, help="original models/model_weight.pth")
    p.add_argument("--save_path", type=str, required=True, help="output checkpoint path")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--voxel_size", type=float, default=1.0, help="Å per voxel after processEMData (default 1.0)")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--crop_size", type=int, default=64, help="3D patch size (default: 64)")
    p.add_argument(
        "--patches_per_volume",
        type=int,
        default=4,
        help="number of random patches sampled per volume per epoch (default: 4)",
    )
    p.add_argument("--seed", type=int, default=0, help="random seed for patch sampling")

    # label thickness controls
    p.add_argument("--radius_ca", type=int, default=1)
    p.add_argument("--radius_bb", type=int, default=1)
    p.add_argument("--radius_aa", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    cfg = LabelConfig(
        radius_ca=args.radius_ca,
        radius_bb=args.radius_bb,
        radius_aa=args.radius_aa,
    )

    ds = CryoTrainDataset(
        args.train_list,
        voxel_size=args.voxel_size,
        cfg=cfg,
        crop_size=args.crop_size,
        patches_per_volume=args.patches_per_volume,
        seed=args.seed,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cryo = Cryo_Model().to(device)
    ckpt = torch.load(args.pretrained, map_location="cpu")
    if "cryo_model" not in ckpt:
        raise ValueError(f"{args.pretrained} missing key 'cryo_model'")
    cryo.models.load_state_dict(ckpt["cryo_model"])
    print(f"Loaded cryo_model from {args.pretrained}")

    optimizer = torch.optim.Adam(cryo.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    cryo.train()

    step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        total = 0.0
        for it, (em, bb, ca, aa) in enumerate(dl, start=1):
            em = em.to(device)  # [B,1,D,H,W]
            bb = bb.to(device)  # [B,D,H,W]
            ca = ca.to(device)
            aa = aa.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Each ResUNet3D4EM returns 3 logits; we pick the relevant branch
            bb_logits, _, _ = cryo.models["BB_model"](em)  # [B,4,D,H,W]
            _, ca_logits, _ = cryo.models["CA_model"](em)  # [B,4,D,H,W]
            _, _, aa_logits = cryo.models["AA_model"](em)  # [B,21,D,H,W]

            loss_bb = ce(bb_logits, bb)
            loss_ca = ce(ca_logits, ca)
            loss_aa = ce(aa_logits, aa)
            loss = loss_bb + loss_ca + loss_aa
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            step += 1

            if it % args.log_interval == 0:
                print(
                    f"epoch={epoch}/{args.epochs} it={it}/{len(dl)} "
                    f"loss={loss.item():.4f} bb={loss_bb.item():.4f} ca={loss_ca.item():.4f} aa={loss_aa.item():.4f}"
                )

        print(f"Epoch {epoch} avg_loss={total/len(dl):.4f} time={time.time()-t0:.1f}s")

    # Save: keep backbone/design weights, replace cryo_model
    ckpt["cryo_model"] = cryo.models.state_dict()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(ckpt, args.save_path)
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()

