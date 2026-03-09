import torch
from src.chroma.layers.linalg import eig_leading
from src.chroma.layers.structure import geometry
import numpy as np
import os
import json
from copy import deepcopy
import mrcfile
from scipy.ndimage import zoom
from Bio import PDB


alphabet = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']

AA_map = {
  "ALA": "A",
  "CYS": "C",
  "ASP": "D",
  "GLU": "E",
  "PHE": "F",
  "GLY": "G",
  "HIS": "H",
  "ILE": "I",
  "LYS": "K",
  "LEU": "L",
  "MET": "M",
  "ASN": "N",
  "PRO": "P",
  "GLN": "Q",
  "ARG": "R",
  "SER": "S",
  "THR": "T",
  "VAL": "V",
  "TRP": "W",
  "TYR": "Y"
}
DNA_map = {
    'DA' : 'A',
    'DG' : 'G',
    'DC' : 'C',
    'DT' : 'T'
}
RNA_map = {
    'A' : 'A',
    'G' : 'G',
    'C' : 'C',
    'U' : 'U'
}

parser = PDB.PDBParser()

def resize_3d_data(data, target_shape):
    zoom_factors = (
        target_shape[0] / data.shape[0],
        target_shape[1] / data.shape[1],
        target_shape[2] / data.shape[2]
    )
    resized_data = zoom(data, zoom_factors, order=3) 
    return resized_data


def get_data(dir_path):
    protein_data, seq, chain_index = None, None, None

    for file in os.listdir(dir_path):
        if file.endswith('.mrc'):
            dm_path = os.path.join(dir_path, file)
            try:
                p_map = mrcfile.open(dm_path, mode='r')
                protein_data = deepcopy(p_map.data)
                protein_data = resize_3d_data(protein_data, [360, 360, 360])
            except Exception as e:
                print(f"Error loading or processing the density map: {e}")
                raise
        elif file.endswith('.json'):
            seq_chain_path = os.path.join(dir_path, file)
            try:
                with open(seq_chain_path, 'r') as f:
                    seq_chain = json.load(f)
                    seq, chain_index = seq_chain['seq'], seq_chain['chain_index']
                    seq = np.array([alphabet.index(item) for item in seq])
                    chain_index = np.array(chain_index)
            except Exception as e:
                print(f"Error loading or processing the seq/chain: {e}")
                raise
    
    if protein_data is None:
        raise FileNotFoundError("No .mrc file found in the specified directory or failed to load.")
    if seq is None or chain_index is None:
        raise FileNotFoundError("No valid .json file found in the specified directory or failed to load.")
    return protein_data, seq, chain_index


def get_coord_from_pdb(pdb_path):
    if not os.path.exists(pdb_path):
        print(f"Error: PDB file '{pdb_path}' does not exist.")
        return None

    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("", pdb_path)
    except Exception as e:
        print(f"Error parsing PDB file '{pdb_path}': {e}")
        return None

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                coord = np.zeros((4, 3), dtype=np.float32)                
                for i, atom in enumerate(residue):
                    if i >= 4:
                        break
                    coord[i] = atom.coord
                coords.append(coord)

    if not coords:
        print("Warning: No atomic coordinates were extracted.")
        return None
    
    return np.array(coords, dtype=np.float32)

def eigen(F):
    method = 'power'
    # Compute optimal quaternion by extracting leading eigenvector
    if method == "symeig":
        L, V = torch.linalg.eigh(F)
        top_eig = L[:, 3]
        vec = V[:, :, 3]
    elif method == "power":
        top_eig, vec = eig_leading(F, num_iterations=50)
    else:
        raise NotImplementedError
    return top_eig, vec

def align(
    X_mobile,
    X_target,
    mask=None,
    _eps = 1e-5
):
    """Compute optimal RMSDs between each corresponding batch members.
    https://math.unm.edu/~vageli/papers/rmsd.pdf
    Args:
        X_mobile (Tensor): Mobile coordinates with shape
            `(..., num_atoms, 3)`.
        X_target (Tensor): Target coordinates with shape
            `(..., num_atoms, 3)`.
        mask (Tensor, optional): Binary mask tensor for missing atoms with
            shape `(..., num_atoms)`.
        compute_alignment (boolean, optional): If True, also return the
            superposed coordinates.

    Returns:
        RMSD (Tensors): Optimal RMSDs after superposition for all pairs of
            input structures with shape `(...)`.
        X_mobile_transform (Tensor, optional): Superposed coordinates with
            shape `(..., num_atoms, 3)`. Requires
            `compute_alignment` = True`.
    """
    R_to_F = np.zeros((9, 16)).astype("f")
    F_nonzero = [
    [(0,0,1.),(1,1,1.),(2,2,1.)],            [(1,2,1.),(2,1,-1.)],            [(2,0,1.),(0,2,-1.)],            [(0,1,1.),(1,0,-1.)],
            [(1,2,1.),(2,1,-1.)],  [(0,0,1.),(1,1,-1.),(2,2,-1.)],             [(0,1,1.),(1,0,1.)],             [(0,2,1.),(2,0,1.)],
            [(2,0,1.),(0,2,-1.)],             [(0,1,1.),(1,0,1.)],  [(0,0,-1.),(1,1,1.),(2,2,-1.)],             [(1,2,1.),(2,1,1.)],
            [(0,1,1.),(1,0,-1.)],             [(0,2,1.),(2,0,1.)],             [(1,2,1.),(2,1,1.)],  [(0,0,-1.),(1,1,-1.),(2,2,1.)]
    ]
    # fmt: on

    for F_ij, nonzero in enumerate(F_nonzero):
        for R_i, R_j, sign in nonzero:
            R_to_F[R_i * 3 + R_j, F_ij] = sign
    R_to_F = torch.from_numpy(R_to_F)
    # Collapse all leading batch dimensions
    num_atoms = X_mobile.size(-2)
    batch_dims = list(X_mobile.shape)[:-2]
    X_mobile = X_mobile.reshape([-1, num_atoms, 3])
    X_target = X_target.reshape([-1, num_atoms, 3])
    num_batch = X_mobile.size(0)
    if mask is not None:
        mask = mask.reshape([-1, num_atoms])

    # Center coordinates
    if mask is None:
        X_mobile_mean = X_mobile.mean(dim=1, keepdim=True)
        X_target_mean = X_target.mean(dim=1, keepdim=True)
    else:
        mask_expand = mask.unsqueeze(-1)
        X_mobile_mean = torch.sum(mask_expand * X_mobile, 1, keepdim=True) / (
            torch.sum(mask_expand, 1, keepdim=True) + _eps
        )
        X_target_mean = torch.sum(mask_expand * X_target, 1, keepdim=True) / (
            torch.sum(mask_expand, 1, keepdim=True) + _eps
        )

    X_mobile_center = X_mobile - X_mobile_mean
    X_target_center = X_target - X_target_mean

    if mask is not None:
        X_mobile_center = mask_expand * X_mobile_center
        X_target_center = mask_expand * X_target_center

    # Cross-covariance matrices contract over atoms
    R = torch.einsum("sai,saj->sij", [X_mobile_center, X_target_center])

    # F Matrix has leading eigenvector as optimal quaternion
    R_flat = R.reshape(num_batch, 9)
    R_to_F = R_to_F.type(R_flat.dtype).to(X_mobile.device)
    F = torch.matmul(R_flat, R_to_F).reshape(num_batch, 4, 4)

    top_eig, vec = eigen(F + 1e-5 * torch.randn_like(F))


    # Compute RMSD using top eigenvalue
    norms = (X_mobile_center ** 2).sum(dim=[-1, -2]) + (X_target_center ** 2).sum(
        dim=[-1, -2]
    )
    sqRMSD = torch.relu((norms - 2 * top_eig) / (num_atoms + _eps))
    RMSD = torch.sqrt(sqRMSD).mean().item()

    R = geometry.rotations_from_quaternions(vec, normalize=False)

    X_mobile_transform = torch.einsum("bxr,bir->bix", R, X_mobile_center)
    X_mobile_transform = X_mobile_transform + X_target_mean

    if mask is not None:
        X_mobile_transform = mask_expand * X_mobile_transform
    return X_mobile_transform, RMSD