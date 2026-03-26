from __future__ import annotations
import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set

import numpy as np
import SimpleITK as sitk

from sm_image import SMImage
from labels import canon_id_for, legend_rows, all_canon_names_sorted


def _sm_read_from_stdin() -> SMImage:
    data = sys.stdin.buffer.read()
    return SMImage.from_bytes(data)

def _sitk_from_sm(sm: SMImage) -> sitk.Image:
    md = getattr(sm, "metadata", {}) or {}
    spacing = tuple(map(float, (md.get("spacing") or md.get("voxel_spacing") or md.get("pixdim") or (1.0, 1.0, 1.0))))
    origin  = tuple(map(float, (md.get("origin")  or (0.0, 0.0, 0.0))))
    direction = md.get("direction")
    if not direction or len(direction) < 9:
        direction = (1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0)

    img = sitk.GetImageFromArray(sm.pixel_array)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(tuple(direction[:9]))
    return img

def _write_like(sm_in: SMImage, arr: np.ndarray, dest: Path) -> None:
    img = sitk.GetImageFromArray(arr)
    md = getattr(sm_in, "metadata", {}) or {}
    spacing = tuple(map(float, (md.get("spacing") or md.get("voxel_spacing") or md.get("pixdim") or (1.0, 1.0, 1.0))))
    origin  = tuple(map(float, (md.get("origin")  or (0.0, 0.0, 0.0))))
    direction = md.get("direction")
    if not direction or len(direction) < 9:
        direction = (1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(tuple(direction[:9]))
    sitk.WriteImage(img, str(dest))

def _arr_from_nii(p: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))

# -----------------------
# TS helpers
# -----------------------

def _seg_dir(out_dir: Path) -> Path:
    """Return directory with per-class *.nii.gz (supports nested 'segmentations/')."""
    nested = out_dir / "segmentations"
    return nested if nested.is_dir() else out_dir

def _class_files(seg_dir: Path) -> List[Path]:
    """Recursively collect per-class files; exclude a dense 'segmentations.nii.gz' if present."""
    files = [p for p in seg_dir.rglob("*.nii.gz") if p.name != "segmentations.nii.gz"]
    return sorted(files)

def _wanted_ids_from_targets(targets: List[str]) -> Set[int]:
    ids: Set[int] = set()
    for t in targets:
        cid = canon_id_for(t)
        if cid is not None:
            ids.add(int(cid))
    return ids

def _discoverable_cids(seg_dir: Path) -> Tuple[Set[int], Dict[int, Path], List[str]]:
    """
    Scan all per-class files and return:
      - set of canonical IDs found
      - map cid -> path (last one wins; files are sorted deterministically)
      - list of raw stems discovered (for debug)
    """
    files = _class_files(seg_dir)
    stems = []
    cid_to_path: Dict[int, Path] = {}
    cids: Set[int] = set()
    for p in files:
        stem = p.stem.lower().strip()
        stems.append(stem)
        cid = canon_id_for(stem)
        if cid is None:
            continue
        cid = int(cid)
        cids.add(cid)
        cid_to_path[cid] = p
    return cids, cid_to_path, stems

def _dense_from_cids(seg_dir: Path, want_ids: Set[int]) -> Optional[np.ndarray]:
    """Assemble dense map from available per-class files for a set of canonical IDs."""
    _, cid_to_path, _ = _discoverable_cids(seg_dir)
    # find a reference shape .....
    ref_shape = None
    for cid in sorted(want_ids):
        p = cid_to_path.get(cid)
        if p is not None:
            ref_shape = _arr_from_nii(p).shape
            break
    if ref_shape is None:
        return None

    dense = np.zeros(ref_shape, dtype=np.int32)
    written = 0
    # deterministic layering: increasing canonical ID
    for cid in sorted(want_ids):
        p = cid_to_path.get(cid)
        if p is None:
            continue
        m = _arr_from_nii(p) > 0.5
        if m.shape != dense.shape:
            raise RuntimeError(f"Shape mismatch for {p.name}: {m.shape} vs {dense.shape}")
        vox = int(np.count_nonzero(m))
        if vox:
            dense[m] = cid
            written += vox
    if written == 0:
        return None
    return dense

def _union_binary(seg_dir: Path, want_ids: Set[int]) -> Optional[np.ndarray]:
    _, cid_to_path, _ = _discoverable_cids(seg_dir)
    # reference shape
    ref_shape = None
    for cid in sorted(want_ids):
        p = cid_to_path.get(cid)
        if p is not None:
            ref_shape = _arr_from_nii(p).shape
            break
    if ref_shape is None:
        return None

    out = np.zeros(ref_shape, dtype=np.uint8)
    written = 0
    for cid in sorted(want_ids):
        p = cid_to_path.get(cid)
        if p is None:
            continue
        m = (_arr_from_nii(p) > 0.5).astype(np.uint8)
        out |= m
        written += int(np.count_nonzero(m))
    if written == 0:
        return None
    return out

# -----------------------------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="total")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--output_mode", choices=["labelmap_raw", "multiclass_canon", "binary"], default="labelmap_raw")
    ap.add_argument("--roi_subset", type=str, default=None, help="comma-separated class names")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample_id", required=True)
    args = ap.parse_args()

    sm_in = _sm_read_from_stdin()

    # Working dirs
    work = Path.cwd() / "ts_work" / args.sample_id
    out_dir = work / "ts_out"
    final_dir = Path(args.outdir) / "totalseg" / args.sample_id
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # Write input NIfTI for TS
    in_nii = work / "in.nii.gz"
    sitk.WriteImage(_sitk_from_sm(sm_in), str(in_nii))

    # Run TS (we rely on per-class outputs; TS may place them under ts_out/segmentations)
    cmd = ["TotalSegmentator", "-i", str(in_nii), "-o", str(out_dir), "--task", args.task, "--nr_thr_saving", "1"]
    if args.fast:
        cmd.append("--fast")

    # If user provided a subset, let TS compute only those (faster)
    if args.roi_subset:
        subset_tokens = [t.strip() for t in args.roi_subset.split(",") if t.strip()]
        if subset_tokens:
            cmd += ["--roi_subset"] + subset_tokens

    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stdout, flush=True)
        print(proc.stderr, file=sys.stderr, flush=True)
        sys.exit(proc.returncode)

    seg_dir = _seg_dir(out_dir)

    # Build desired ID set:
    if args.output_mode == "labelmap_raw":
        # treat as "multiclass over ALL canonical classes", skipping any missing
        targets = all_canon_names_sorted()
    else:
        # explicit subset (or all if user omitted it by mistake)
        if args.roi_subset:
            targets = [t.strip() for t in args.roi_subset.split(",") if t.strip()]
        else:
            targets = all_canon_names_sorted()

    want_ids = _wanted_ids_from_targets(targets)

    if args.output_mode == "binary":
        if not want_ids:
            print("ERROR: binary mode requires at least one valid class in roi_subset.", file=sys.stderr)
            sys.exit(2)
        arr = _union_binary(seg_dir, want_ids)
        if arr is None or np.count_nonzero(arr) == 0:
            _, _, stems = _discoverable_cids(seg_dir)
            print("ERROR: binary union produced empty mask — check class names and TS outputs.", file=sys.stderr)
            print(f"Checked directory: {seg_dir}", file=sys.stderr)
            print(f"Discovered stems: {sorted(set(stems))}", file=sys.stderr)
            sys.exit(2)
        result = SMImage(pixel_array=arr.astype(np.uint8), metadata=getattr(sm_in, "metadata", {}))

    else:
        dense = _dense_from_cids(seg_dir, want_ids)
        if dense is None or np.count_nonzero(dense) == 0:
            _, _, stems = _discoverable_cids(seg_dir)
            print("ERROR: canonical dense labelmap is empty — likely no per-class files found or names mismatch.", file=sys.stderr)
            print(f"Checked directory: {seg_dir}", file=sys.stderr)
            print(f"Requested IDs: {sorted(list(want_ids))}", file=sys.stderr)
            print(f"Discovered stems: {sorted(set(stems))}", file=sys.stderr)
            sys.exit(2)
        result = SMImage(pixel_array=dense.astype(np.int32), metadata=getattr(sm_in, "metadata", {}))

    # Persist primary result
    out_path = final_dir / "totalseg_result.nii.gz"
    _write_like(sm_in, result.pixel_array, out_path)

    # Write legend (id -> name)
    legend_tsv = final_dir / "labels_total.tsv"
    with legend_tsv.open("w") as f:
        if args.output_mode == "binary":
            subset_names = [t.strip() for t in (args.roi_subset or "").split(",") if t.strip()]
            label_desc = " + ".join(subset_names) if subset_names else "user-defined subset"
            f.write("value\tmeaning\n")
            f.write("0\tbackground\n")
            f.write(f"1\t{label_desc}\n")
        else:
            f.write("id\tname\n")
            for i, name in legend_rows():
                f.write(f"{i}\t{name}\n")


    # Send back to pipeline
    sys.stdout.buffer.write(result.to_bytes())

if __name__ == "__main__":
    main()
