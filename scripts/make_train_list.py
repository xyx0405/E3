#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a train_list.txt for train_cryo_model.py from the bundled training tree.

Expected directory layout (example):
  data/train/EMD_7/7002/
    - emd_normalized_map.mrc
    - 6asx.pdb
    - 6asx_coil.pdb / 6asx_helix.pdb / 6asx_strand.pdb  (ignored)

Output format (one sample per line):
  <map_path> <pdb_path>

By default paths are written as repo-relative paths to keep the list portable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


IGNORE_PDB_SUFFIXES = (
    "_coil.pdb",
    "_helix.pdb",
    "_strand.pdb",
    "_all_atom_model.pdb",
)


def pick_pdb(case_dir: Path) -> Path | None:
    pdbs = sorted([p for p in case_dir.glob("*.pdb") if p.is_file()])
    if not pdbs:
        return None

    # Prefer the "base" pdb like 6asx.pdb over 6asx_coil/helix/strand
    filtered = []
    for p in pdbs:
        name = p.name
        if any(name.endswith(suf) for suf in IGNORE_PDB_SUFFIXES):
            continue
        filtered.append(p)

    if filtered:
        # Often there is exactly one: <pdbid>.pdb
        return sorted(filtered, key=lambda x: (len(x.name), x.name))[0]

    # Fallback: take the first pdb
    return pdbs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="data/train/EMD_7",
        help="Training root containing case subdirectories (default: data/train/EMD_7)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="data/train_list.txt",
        help="Output train list path (default: data/train_list.txt)",
    )
    ap.add_argument(
        "--absolute",
        action="store_true",
        help="Write absolute paths instead of repo-relative paths",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    # Important: the repo uses symlinks (e.g. data/ -> /eds-storage/...).
    # For portability we prefer writing *repo-visible paths* like "data/train/..."
    # rather than resolved absolute targets under /eds-storage/.
    root_path = (repo_root / args.root) if not Path(args.root).is_absolute() else Path(args.root)
    out_path = (repo_root / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    if not root_path.exists():
        raise SystemExit(f"Root not found: {root_path}")

    lines: list[str] = []
    skipped: list[tuple[str, str]] = []

    for case_dir in sorted([p for p in root_path.iterdir() if p.is_dir()]):
        map_path = case_dir / "emd_normalized_map.mrc"
        if not map_path.exists():
            skipped.append((case_dir.name, "missing emd_normalized_map.mrc"))
            continue

        pdb_path = pick_pdb(case_dir)
        if pdb_path is None or not pdb_path.exists():
            skipped.append((case_dir.name, "missing pdb"))
            continue

        if args.absolute:
            # Absolute, resolved paths (will dereference symlinks).
            mp = str(map_path.resolve())
            pp = str(pdb_path.resolve())
        else:
            # Repo-visible relative paths. If the target is outside the repo (due to
            # symlinks), fall back to a non-resolved relative path.
            try:
                mp = str(map_path.relative_to(repo_root))
            except ValueError:
                mp = os.path.relpath(str(map_path), str(repo_root))
            try:
                pp = str(pdb_path.relative_to(repo_root))
            except ValueError:
                pp = os.path.relpath(str(pdb_path), str(repo_root))

        lines.append(f"{mp} {pp}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"Wrote {len(lines)} samples to {out_path}")
    if skipped:
        print(f"Skipped {len(skipped)} cases (showing up to 20):")
        for name, reason in skipped[:20]:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()

