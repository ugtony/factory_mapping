#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from retriever.netvlad_retriever import NetVLADRetriever
from localizer.local_matcher import load_local_keypoints, load_local_features
from localizer.pose_estimator import estimate_pose_from_db_image

import cv2
import h5py
import torch
import numpy as np
import traceback

# --- HLOC Imports ---
from hloc.utils.base_model import dynamic_load
from hloc import extractors
from hloc import extract_features
# --- End HLOC Imports ---

# --- Import LightGlue Directly ---
try:
    from lightglue import LightGlue
    from lightglue.utils import load_image, rbd  # image loader + dict->tensor
except ImportError:
    print("[Error] Could not import LightGlue directly. Is 'lightglue' installed?")
    print("         Try: pip install lightglue")
    exit(1)
# --- End LightGlue Import ---

OUT_ROOT = Path("outputs-hloc")

# --- Config ---
GLOBAL_CONF_NAME = "netvlad"
GLOBAL_CONF = extract_features.confs[GLOBAL_CONF_NAME]
LOCAL_CONF_NAME = "superpoint_aachen"
LOCAL_CONF = extract_features.confs[LOCAL_CONF_NAME]
# --- End Config ---

def load_rgb_tensor_for_netvlad(img_path: str | Path, max_side: int | None = 1024) -> torch.Tensor:
    """Read an RGB image and return a tensor [1,3,H,W] float32 in [0,1] for NetVLAD."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if max_side is not None:
        scale = min(1.0, float(max_side) / float(max(h, w)))
        if scale != 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # [H,W,C] -> [1,3,H,W]
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    assert tensor.ndim == 4 and tensor.shape[1] == 3, f"Bad shape for NetVLAD: {tuple(tensor.shape)}"
    return tensor

def load_gray_tensor_for_superpoint(img_path: str | Path, max_side: int | None = 1024) -> torch.Tensor:
    """Read a grayscale image and return a tensor [1,1,H,W] float32 in [0,1] for SuperPoint/LightGlue."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # [H,W], uint8
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]
    if max_side is not None:
        scale = min(1.0, float(max_side) / float(max(h, w)))
        if scale != 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # [H,W] -> [1,1,H,W] in [0,1]
    tensor = torch.from_numpy(img).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    assert tensor.ndim == 4 and tensor.shape[1] == 1, f"Bad shape for SuperPoint: {tuple(tensor.shape)}"
    return tensor


def get_query_key_and_dir(query_img_path):
    """Derive HDF5 key and block directory from a path inside data/."""
    try:
        data_root = Path("data")
        query_img_path = Path(query_img_path)
        relative_path = query_img_path.relative_to(data_root)
        block_name = relative_path.parts[0]
        image_dir = data_root / block_name
        qkey = relative_path.relative_to(block_name).as_posix()
        return qkey, image_dir
    except ValueError:
        print(f"[Warn] Query path not in 'data/' tree, using fallback key generation.")
        qkey = f"query/{query_img_path.name}"
        image_dir = query_img_path.parent.parent
        return qkey, image_dir

def main(query_img_path, block_dirs, out_pose_txt="poses.txt", per_block_topk=20, max_side=1024):
    query_img_path = Path(query_img_path)
    out_pose_txt = Path(out_pose_txt)
    print(f"Query image : {query_img_path}")
    print(f"Resize max_side: {max_side} (Note: LightGlue uses its own sizing if needed)")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    qkey, q_image_dir = get_query_key_and_dir(query_img_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1) Load Models ---
    Model_NetVLAD = dynamic_load(extractors, GLOBAL_CONF['model']['name'])
    model_netvlad = Model_NetVLAD(GLOBAL_CONF['model']).eval().to(device)

    Model_SuperPoint = dynamic_load(extractors, LOCAL_CONF['model']['name'])
    model_superpoint = Model_SuperPoint(LOCAL_CONF['model']).eval().to(device)

    try:
        model_matcher = LightGlue(features='superpoint').eval().to(device)
    except Exception as e:
        print(f"[Error] Failed to load standalone LightGlue model: {e}")
        exit(1)

    print(f"Models (NetVLAD, SuperPoint, LightGlue) loaded to {device}.")

    # --- 2) Extract Query Features (Global + Local) ---

    # 2a) Global (NetVLAD): ***MUST feed RGB image, not H5 or feature maps***
    try:
        image_rgb_for_netvlad = load_rgb_tensor_for_netvlad(query_img_path, max_side=max_side).to(device)
    except Exception as e:
        print(f"[Error] Failed to load query image {query_img_path} for NetVLAD: {e}")
        traceback.print_exc()
        exit(1)

    qfeat_path = OUT_ROOT / "query_netvlad.h5"
    print(f"Extracting NetVLAD feature for key: {qkey}")
    with torch.no_grad():
        pred_global = model_netvlad({'image': image_rgb_for_netvlad})

    # HLOC NetVLAD usually returns {'global_descriptor': Tensor[B,D]}
    if isinstance(pred_global, dict) and 'global_descriptor' in pred_global:
        gdesc = pred_global['global_descriptor'][0].detach().cpu().numpy()
    else:
        gdesc = pred_global
        if torch.is_tensor(gdesc):
            gdesc = gdesc[0].detach().cpu().numpy()

    with h5py.File(qfeat_path, 'w') as f:
        f.create_dataset(qkey, data=gdesc.astype(np.float32), dtype='f4')

    # 2b) Local (SuperPoint): HLOC models expect dict input: {'image': tensor}
    print(f"Extracting SuperPoint feature for key: {qkey}")
    try:
        image_tensor_local = load_gray_tensor_for_superpoint(query_img_path, max_side).to(device)  # [1,1,H,W]
    except Exception as e:
        print(f"[Error] Failed to load query image {query_img_path} for SuperPoint: {e}")
        traceback.print_exc()
        exit(1)


    with torch.no_grad():
        pred_local = model_superpoint({'image': image_tensor_local})

    # unify output format
    if isinstance(pred_local.get('keypoints'), list):
        kpts0 = pred_local['keypoints'][0].to(torch.float32)
        desc0 = pred_local['descriptors'][0].to(torch.float32)
    elif torch.is_tensor(pred_local.get('keypoints')):
        kpts0 = pred_local['keypoints'].squeeze(0).to(torch.float32)
        desc0 = pred_local['descriptors'].squeeze(0).to(torch.float32)
    else:
        print("[Error] Unexpected output format from SuperPoint model.")
        exit(1)

    query_kpts_lg = kpts0[None]                          # [1,N,2]
    query_desc_lg = desc0.transpose(-1, -2)[None]        # [1,N,256] (from [256,N])
    kpts_q_numpy = kpts0.cpu().numpy()
    print("Query feature extraction complete.")

    # --- 3) Validate Blocks ---
    block_paths = [Path(p) for p in block_dirs]
    valid_paths = []
    for p in block_paths:
        if not p.is_dir():
            print(f"[Warn] Provided block path not found, skipping: {p}")
            continue
        local_h5_path = p / "local-superpoint_aachen.h5"
        if not (p / "sfm").is_dir() or \
           not (p / "global-netvlad.h5").exists() or \
           not local_h5_path.exists():
            print(f"[Warn] Block path {p} seems incomplete, skipping.")
            continue
        valid_paths.append(p)
    if not valid_paths:
        raise RuntimeError("No valid block directories provided or found.")

    # --- 4) Retrieval (Global) ---
    candidates = [p.name for p in valid_paths]
    print(f"Candidate blocks: {candidates}")
    retriever = NetVLADRetriever(OUT_ROOT)
    retrieved = retriever.retrieve(qfeat_path, candidates, top_k=per_block_topk)

    # --- 5) Matching + Pose Estimation ---
    best = None
    for block, db_global_key, score in retrieved:
        image_key = Path(db_global_key).parent.as_posix()

        out_dir = OUT_ROOT / block
        local_h5 = out_dir / "local-superpoint_aachen.h5"
        sfm_dir = out_dir / "sfm"
        db_image_path = Path("data") / block / image_key

        matches_np = None
        try:
            # (A) Load DB local features
            db_feats_torch = load_local_features(local_h5, image_key, device)
            db_kpts_lg = db_feats_torch['keypoints']         # [1,M,2]
            db_desc_lg = db_feats_torch['descriptors']       # [1,M,256]

            # (B) Load DB image (for LightGlue context)
            try:
                image_tensor_db = load_gray_tensor_for_superpoint(db_image_path, max_side).to(device)  # [1,1,H,W]
            except Exception as e:
                print(f"    [Warn] Failed to load DB image {db_image_path}: {e}. Skipping match.")
                continue


            # (C) Match (LightGlue accepts kpts/desc + image tensors)
            with torch.no_grad():
                input_dict = {
                    'image0': {
                        'keypoints': query_kpts_lg,
                        'descriptors': query_desc_lg,
                        'image': image_tensor_local,   # query image tensor [1,C,H,W]
                    },
                    'image1': {
                        'keypoints': db_kpts_lg,
                        'descriptors': db_desc_lg,
                        'image': image_tensor_db,      # db image tensor [1,C,H,W]
                    }
                }
                matches_data = model_matcher(input_dict)

            # (D) Get matches
            matches0 = matches_data.get('matches0')
            if matches0 is not None:
                matches_np = matches0[:, :2].detach().cpu().numpy().astype(int)
            else:
                matches_np = np.empty((0, 2), dtype=int)

            # (E) Load DB keypoints for PnP (pixel coords)
            kpts_db_numpy = load_local_keypoints(local_h5, image_key)

        except Exception as e:
            print(f"skip {block}/{image_key}: Failed during in-memory matching/loading:")
            traceback.print_exc()
            continue

        if matches_np is None or len(matches_np) == 0:
            print(f"skip {block}/{image_key}: No matches found (in-memory).")
            continue

        # (F) Estimate Pose
        pose = estimate_pose_from_db_image(
            sfm_dir,
            image_key,
            matches_np,     # (N, 2)
            kpts_q_numpy,   # (N_q, 2)
            kpts_db_numpy,  # (N_db, 2)
            query_img_path, # original query path (for size)
            max_side
        )

        if pose is None:
            continue

        pose["retrieval_score"] = float(score)
        pose["block"] = block
        pose["db_image"] = image_key

        print(f"    > OK. {block}/{image_key}: Score={score:.3f}, Inliers={pose['inliers']}")
        if (best is None) or (pose["inliers"] > best["inliers"]):
            best = pose

    if best is None:
        print("Pose estimation failed for all candidates.")
        return

    print(f"✅ Best Pose Found! Block={best['block']} Inliers={best['inliers']} DB={best['db_image']}")
    qvec = " ".join(map(str, best["qvec"]))
    tvec = " ".join(map(str, best["tvec"]))
    out_pose_txt.write_text(f"{query_img_path.name} {qvec} {tvec}\n")
    print(f"Saved to: {out_pose_txt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Path to the query image (e.g., data/street/query/IMG_1397.jpg)")
    ap.add_argument("--block_dirs", required=True, nargs='+',
                    help="One or more block output directories (e.g., outputs-hloc/street)")
    ap.add_argument("--out", default="poses.txt", help="Output pose file.")
    ap.add_argument("--topk", type=int, default=20, help="Top-K retrieval candidates per block.")
    ap.add_argument("--max_side", type=int, default=1024,
                    help="The 'max_side' used for LOCAL (SuperPoint) feature extraction.")
    args = ap.parse_args()

    # Heads-up if LOCAL resize_max mismatches CLI
    if LOCAL_CONF['preprocessing']['resize_max'] != args.max_side:
        print(f"[Warning] --max_side ({args.max_side}) != LOCAL_CONF resize_max ({LOCAL_CONF['preprocessing']['resize_max']})")

    main(args.query, args.block_dirs, args.out, args.topk, args.max_side)
