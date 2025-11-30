import pycolmap
import sys
from pathlib import Path

# 使用方式: python check_registered_images.py <sfm_aligned路徑>
# 範例: python check_registered_images.py outputs-hloc/brazil360/sfm_aligned

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_registered_images.py <path_to_sfm_dir>")
        sys.exit(1)
        
    sfm_path = Path(sys.argv[1])
    
    if not (sfm_path / "images.bin").exists():
        print(f"[Error] images.bin not found in {sfm_path}")
        sys.exit(1)

    print(f"Loading model from: {sfm_path}")
    try:
        # 載入 COLMAP 模型
        recon = pycolmap.Reconstruction(sfm_path)
        
        # 取得所有已註冊影像的名稱
        # recon.images 是一個字典 {image_id: Image object}
        registered_names = sorted([img.name for img in recon.images.values()])
        
        print(f"Total registered images: {len(registered_names)}")
        print("-" * 40)
        
        # 列印出檔名
        for name in registered_names:
            print(name)
            
        # 如果您想檢查特定檔名是否存在，可以取消下面註解並修改檔名:
        # target = "frames-000123_F.jpg"
        # if target in registered_names:
        #     print(f"\n[FOUND] {target} is registered!")
        # else:
        #     print(f"\n[MISSING] {target} was DROPPED by COLMAP.")
            
    except Exception as e:
        print(f"[Error] Failed to load reconstruction: {e}")

if __name__ == "__main__":
    main()