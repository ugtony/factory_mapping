#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_pinhole_intrinsics(W, H, fov_deg):
    fov_rad = np.deg2rad(fov_deg)
    f = 0.5 * W / np.tan(0.5 * fov_rad)
    K = np.array([[f, 0, W / 2.0],
                  [0, f, H / 2.0],
                  [0, 0, 1.0]], dtype=np.float32)
    return K

def get_rotation_matrix(yaw_deg, pitch_deg=0, roll_deg=0):
    # 簡單的尤拉角轉旋轉矩陣
    yaw, pitch, roll = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg), np.deg2rad(roll_deg)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    # 注意：這裡的旋轉順序與座標系定義可能影響最終視角，
    # 對於單純水平環景 (Pitch=0, Roll=0)，此順序通常沒問題。
    return Rz @ Ry @ Rx

def build_remap_tables(W_eq, H_eq, W_pin, H_pin, K, R):
    # 建立 Pinhole 影像每個像素的座標網格
    u, v = np.meshgrid(np.arange(W_pin), np.arange(H_pin))
    ones = np.ones_like(u)
    xyz_pin = np.stack([u, v, ones], axis=-1).reshape(-1, 3) # (N, 3)
    
    # 1. 用 K_inv 轉回相機座標系的光線方向
    # 在標準針孔模型中，Y 軸是朝下的 (v 越大越下面)
    K_inv = np.linalg.inv(K)
    rays_cam = (K_inv @ xyz_pin.T).T
    
    # 2. 用 R 轉回世界(球面)座標系
    rays_world = (R @ rays_cam.T).T
    x, y, z = rays_world[:, 0], rays_world[:, 1], rays_world[:, 2]
    
    # 3. 將光線 (x,y,z) 轉換為球座標 (lon, lat)
    # [修正關鍵]：因為相機 Y 軸朝下，而球面緯度定義 Y(或Z) 朝上為正。
    # 我們在此將 y 取負號，將「朝下」轉為「朝上」，以符合一般地理定義。
    lon = np.arctan2(x, z) 
    lat = np.arcsin(np.clip(-y / (np.linalg.norm(rays_world, axis=1) + 1e-8), -1, 1))
    
    # 4. 將 (lon, lat) 轉換為 Equirectangular 影像座標 (u_eq, v_eq)
    # lon [-pi, pi] -> u [0, W_eq]
    # lat [-pi/2, pi/2] -> v [0, H_eq] (其中 +pi/2 是北極，對應 v=0)
    u_eq = (lon / (2 * np.pi) + 0.5) * W_eq
    v_eq = (-lat / np.pi + 0.5) * H_eq
    
    map_x = u_eq.reshape(H_pin, W_pin).astype(np.float32)
    map_y = v_eq.reshape(H_pin, W_pin).astype(np.float32)
    return map_x, map_y

def main():
    parser = argparse.ArgumentParser(description="Convert 360 Equirectangular to multiple pinhole views.")
    parser.add_argument("--input_dir", required=True, help="Directory containing 360 images.")
    parser.add_argument("--output_dir", required=True, help="Output directory for pinhole images.")
    parser.add_argument("--width", type=int, default=1024, help="Output pinhole width.")
    parser.add_argument("--height", type=int, default=768, help="Output pinhole height.")
    parser.add_argument("--fov", type=float, default=100.0, help="Horizontal FOV in degrees. (>90 recommended)")
    parser.add_argument("--dense", action="store_true", help="Use 8 views (every 45 deg) instead of 4 views.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 定義視角 (Yaw 角度與檔名後綴)
    if args.dense:
        views = [
            (0, "F"), (45, "FR"), (90, "R"), (135, "RB"),
            (180, "B"), (-135, "BL"), (-90, "L"), (-45, "LF")
        ]
        print("[Info] Mode: Dense (8 views)")
    else:
        views = [(0, "F"), (90, "R"), (180, "B"), (-90, "L")]
        print("[Info] Mode: Sparse (4 views)")

    # 支援常見副檔名
    images = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png")))
    if not images:
        print(f"[Error] No images found in {input_dir}")
        return

    print(f"[Info] Found {len(images)} 360 images. Starting conversion...")

    # 預讀第一張以取得尺寸
    img0 = cv2.imread(str(images[0]))
    if img0 is None:
        print(f"[Error] Could not read first image: {images[0]}")
        return
    H_eq, W_eq = img0.shape[:2]
    
    K = get_pinhole_intrinsics(args.width, args.height, args.fov)
    remap_maps = {}

    print("[Info] Pre-calculating remap tables...")
    for yaw, suffix in views:
        R = get_rotation_matrix(yaw_deg=yaw)
        remap_maps[suffix] = build_remap_tables(W_eq, H_eq, args.width, args.height, K, R)

    for img_path in tqdm(images, desc="Converting"):
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # 處理尺寸可能不一致的情況 (雖然少見)
        if img.shape[:2] != (H_eq, W_eq):
             H_eq, W_eq = img.shape[:2]
             # 重新計算 table
             for yaw, suffix in views:
                 R = get_rotation_matrix(yaw_deg=yaw)
                 remap_maps[suffix] = build_remap_tables(W_eq, H_eq, args.width, args.height, K, R)

        base_name = img_path.stem
        for suffix, (map_x, map_y) in remap_maps.items():
            out_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            out_path = output_dir / f"{base_name}_{suffix}.jpg"
            cv2.imwrite(str(out_path), out_img)

    print(f"[Success] Converted {len(images)} images to {output_dir}")

if __name__ == "__main__":
    main()