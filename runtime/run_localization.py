#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from router.wifi_router import WifiRouter
from retriever.netvlad_retriever import NetVLADRetriever
from localizer.local_matcher import load_local_keypoints, dummy_pairing
from localizer.pose_estimator import estimate_pose_from_db_image

OUT_ROOT = Path("outputs-hloc")

def main(query_img, wifi_json, out_pose_txt="poses.txt", per_block_topk=20):
    query_img = Path(query_img)
    wifi_json = Path(wifi_json)
    out_pose_txt = Path(out_pose_txt)

    print(f"Query image : {query_img}")
    wifi_scan = json.loads(wifi_json.read_text())

    router = WifiRouter()
    candidates = [b for b, _ in router.predict_blocks(wifi_scan, top_k=3)]
    print(f"Candidate blocks: {candidates}")
    if not candidates:
        raise RuntimeError("No candidate blocks from Wi-Fi")

    retriever = NetVLADRetriever(OUT_ROOT)
    qfeat_path = OUT_ROOT / "query_netvlad.h5"  # 預先萃取
    retrieved = retriever.retrieve(qfeat_path, candidates, top_k=per_block_topk)

    best = None
    for block, db_key, score in retrieved:
        out_dir = OUT_ROOT / block
        local_h5 = out_dir / "local-superpoint_aachen.h5"
        sfm_dir = out_dir / "sfm"
        qkey = f"query/{query_img.name}"
        try:
            kpts_q = load_local_keypoints(local_h5, qkey)
            kpts_db = load_local_keypoints(local_h5, db_key)
        except Exception as e:
            print(f"skip {block}/{db_key}: {e}")
            continue

        matches = dummy_pairing(kpts_q, kpts_db, max_pairs=300)
        pose = estimate_pose_from_db_image(sfm_dir, db_key, matches, kpts_q, kpts_db)
        if pose is None:
            continue
        pose["retrieval_score"] = float(score)
        pose["block"] = block
        pose["db_image"] = db_key
        if (best is None) or (pose["inliers"] > best["inliers"]):
            best = pose

    if best is None:
        print("Pose estimation failed for all candidates.")
        return

    print(f"OK. Block={best['block']} Inliers={best['inliers']} DB={best['db_image']}")
    qvec = " ".join(map(str, best["qvec"]))
    tvec = " ".join(map(str, best["tvec"]))
    out_pose_txt.write_text(f"{query_img.name} {qvec} {tvec}\n")
    print(f"Saved to: {out_pose_txt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--wifi", required=True)
    ap.add_argument("--out", default="poses.txt")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    main(args.query, args.wifi, args.out, args.topk)
