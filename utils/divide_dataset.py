import os
import glob
import shutil
from tqdm import tqdm

# 原始路径
src_dir = "data/test/SDXL0829_测试集效果不佳"
# 目标路径
dst_dir = "data/test/SDXL0829_测试集效果不佳_divide"

# 匹配 jpg/jpeg 文件
files = glob.glob(os.path.join(src_dir, "*.*p*g"))
files.sort()  # 排序，保证分配稳定

# 创建目标子文件夹 0-9
os.makedirs(dst_dir, exist_ok=True)
num_folders = 10
subfolders = [os.path.join(dst_dir, str(i)) for i in range(num_folders)]
for folder in subfolders:
    os.makedirs(folder, exist_ok=True)

# 均匀分配文件
for idx, file in enumerate(tqdm(files)):
    target_folder = subfolders[idx % num_folders]  # 轮流分配
    shutil.copy(file, target_folder)  # 如果要移动用 shutil.move

    print(f"复制 {file} 到 {target_folder}")

print(f"共处理 {len(files)} 个文件，已均匀分配到 {num_folders} 个文件夹中。")
