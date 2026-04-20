import os
import shutil

def clean_project(root_dir="."):
    pycache_count = 0
    upload_path = os.path.join(root_dir, "uploads")
    data_path = os.path.join(root_dir, "data")
    
    print(f"🚀 Bắt đầu dọn dẹp tại: {os.path.abspath(root_dir)}")
    for root, dirs, files in os.walk(root_dir):
        # Bỏ qua .venv
        if ".venv" in dirs:
            dirs.remove(".venv")            
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                pycache_count += 1
                print(f"  [-] Đã xóa: {pycache_path}")
            except Exception as e:
                print(f"  [!] Lỗi xóa pycache: {e}")

    # delete upload folder
    if os.path.exists(upload_path):
        try:
            shutil.rmtree(upload_path)
            print(f"[-] Đã xóa toàn bộ thư mục: {upload_path}")
        except Exception as e:
            print(f"  [!] Lỗi xóa thư mục upload: {e}")

    if os.path.exists(data_path):
        try:
            shutil.rmtree(data_path)
            print(f"[-] Đã xóa toàn bộ thư mục: {data_path}")
        except Exception as e:
            print(f"  [!] Lỗi xóa thư mục data: {e}")    

    print(f"\n✅ Hoàn tất! Đã dọn {pycache_count}  __pycache__ , uploads , data.")

if __name__ == "__main__":
    clean_project()