import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.interpolate import griddata
import anndata as ad
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

SANTO_REPO = Path(os.environ.get("SANTO_REPO", "SANTO-main"))
sys.path.insert(0, str(SANTO_REPO))

from SANTO_utils import santo


DATA_DIR = Path(os.environ["SANTO_DATA_DIR"])
OUT_DIR  = Path(os.environ["SANTO_OUT_DIR"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = os.environ["SANTO_PAIRS"].split(";")

XCOL = os.environ.get("SANTO_XCOL", "center_x")
YCOL = os.environ.get("SANTO_YCOL", "center_y")

USE_SUBSAMPLE = int(os.environ.get("SANTO_USE_SUBSAMPLE", "0"))

REF_N = int(os.environ.get("SANTO_REF_N", "50000"))
MOV_N = int(os.environ.get("SANTO_MOV_N", "50000"))
GRID_BINS = int(os.environ.get("SANTO_GRID_BINS", "100"))

SEED_REF = int(os.environ.get("SANTO_SEED_REF", "0"))
SEED_MOV = int(os.environ.get("SANTO_SEED_MOV", "1"))

from types import SimpleNamespace

ARGS = SimpleNamespace(
    mode="align",
    dimension=2,
    diff_omics=False,
    alpha=float(os.environ.get("SANTO_ALPHA", 0.3)),
    k=int(os.environ.get("SANTO_K", 10)),
    lr=1e-3,
    epochs=50,
    device=os.environ.get(
        "SANTO_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
)

rng_ref = np.random.default_rng(SEED_REF)
rng_mov = np.random.default_rng(SEED_MOV)


def load_adata(meta_csv, expr_csv):
    meta = pd.read_csv(meta_csv)
    if "cell_id" not in meta.columns:
        meta.rename(columns={meta.columns[0]: "cell_id"}, inplace=True)
    meta["cell_id"] = meta["cell_id"].astype(str)
    meta[XCOL] = pd.to_numeric(meta[XCOL], errors="coerce")
    meta[YCOL] = pd.to_numeric(meta[YCOL], errors="coerce")
    meta = meta.dropna(subset=[XCOL, YCOL])
    meta.set_index("cell_id", inplace=True)

    expr = pd.read_csv(expr_csv, index_col=0)
    expr.index = expr.index.astype(str)

    if not set(expr.columns) & set(meta.index):
        expr = expr.T

    common = meta.index.intersection(expr.columns)
    meta = meta.loc[common]
    expr = expr[common]

    X = sp.csr_matrix(expr.T.values.astype(np.float32))
    adata = ad.AnnData(X)
    adata.obs_names = meta.index
    adata.var_names = expr.index
    adata.obsm["spatial"] = meta[[XCOL, YCOL]].to_numpy(float)

    return adata, meta


def stratified_subsample(coords, ids, n, n_bins):
    if n >= len(ids):
        return ids

    df = pd.DataFrame(coords, columns=["x", "y"], index=ids)
    df["xb"] = pd.qcut(df["x"], n_bins, labels=False, duplicates="drop")
    df["yb"] = pd.qcut(df["y"], n_bins, labels=False, duplicates="drop")

    per_bin = max(1, n // (n_bins * n_bins))
    chosen = []

    for _, g in df.groupby(["xb", "yb"], sort=False):
        k = min(len(g), per_bin)
        chosen.extend(rng_mov.choice(g.index, size=k, replace=False))

    if len(chosen) > n:
        chosen = rng_mov.choice(chosen, size=n, replace=False)

    return list(chosen)


def run_pair(pair_str):
    meta_ref, meta_mov, expr_ref, expr_mov = pair_str.split(",")

    tag = f"{Path(meta_ref).stem}__VS__{Path(meta_mov).stem}"
    out_dir = OUT_DIR / tag
    out_dir.mkdir(exist_ok=True)

    ref_adata, ref_meta = load_adata(DATA_DIR / meta_ref, DATA_DIR / expr_ref)
    mov_adata, mov_meta = load_adata(DATA_DIR / meta_mov, DATA_DIR / expr_mov)

    if USE_SUBSAMPLE:
        ref_ids = rng_ref.choice(ref_adata.obs_names,
                                 size=min(REF_N, ref_adata.n_obs),
                                 replace=False)
        mov_ids = stratified_subsample(
            mov_adata.obsm["spatial"],
            mov_adata.obs_names,
            MOV_N,
            GRID_BINS,
        )
        ref_sub = ref_adata[ref_ids].copy()
        mov_sub = mov_adata[mov_ids].copy()
    else:
        ref_sub = ref_adata
        mov_sub = mov_adata

    aligned_sub, _ = santo(mov_sub, ref_sub, ARGS)
    aligned_sub = np.asarray(aligned_sub)

    orig_sub = mov_sub.obsm["spatial"]
    dxy = aligned_sub - orig_sub

    full_xy = mov_adata.obsm["spatial"]
    dx = griddata(orig_sub, dxy[:, 0], full_xy, method="linear")
    dy = griddata(orig_sub, dxy[:, 1], full_xy, method="linear")

    nan = np.isnan(dx) | np.isnan(dy)
    if nan.any():
        dx[nan] = griddata(orig_sub, dxy[:, 0], full_xy[nan], method="nearest")
        dy[nan] = griddata(orig_sub, dxy[:, 1], full_xy[nan], method="nearest")

    aligned_full = full_xy + np.column_stack([dx, dy])

    out = mov_meta.copy()
    out["x_aligned"] = aligned_full[:, 0]
    out["y_aligned"] = aligned_full[:, 1]

    out.to_csv(out_dir / f"{tag}_santo_aligned.csv")
    print(f"[SANTO] Finished {tag}")


if __name__ == "__main__":
    for p in PAIRS:
        run_pair(p)
