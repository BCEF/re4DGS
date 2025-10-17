import os
import shutil
from pathlib import Path

# 定义源目录和目标目录
source_dir = Path('/home/momo/Downloads/SQ_02_first_two_frames/first_two_frames/')
output_dir = Path('/home/momo/Downloads/SQ_02_first_two_frames/first_two_frames/images')

# 创建输出目录（如果不存在）
output_dir.mkdir(parents=True, exist_ok=True)

# 遍历001到059的文件夹
for i in range(1, 60):
    # 格式化文件夹名称（001, 002, ..., 059）
    folder_name = f'{i:03d}'
    source_folder = source_dir / folder_name
    source_file = source_folder / '000001.jpg'
    
    # 检查源文件是否存在
    if source_file.exists():
        # 目标文件名
        target_file = output_dir / f'{folder_name}.jpg'
        
        # 复制文件到目标位置
        shutil.copy2(source_file, target_file)
        print(f'已复制: {source_file} -> {target_file}')
    else:
        print(f'警告: 文件不存在 - {source_file}')

print(f'\n完成！所有文件已保存到: {output_dir}')