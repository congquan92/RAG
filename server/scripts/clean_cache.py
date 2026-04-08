import os
import shutil

def clean_pycache(root_dir="."):
    count = 0
    print(f"🚀 Đang quét dọn : {os.path.abspath(root_dir)}")
    
    for root, dirs, files in os.walk(root_dir):
        if ".venv" in dirs:
            dirs.remove(".venv")
            
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            print(f"❌ Đang xóa: {pycache_path}")
            shutil.rmtree(pycache_path)
            count += 1
            
    print(f"\n✅ Đã dọn dẹp xong {count} folder __pycache__!")

if __name__ == "__main__":
    clean_pycache()