import json
import os


def transform_file_paths(json_data):
    """
    将 JSON 中的 file_path 按照指定规则转换
    原格式: horse_run/001/000000.jpg
    新格式: horse_run/000000/images/001.jpg
    """
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    for frame in data.get('frames', []):
        old_path = frame['file_path']
        
        # 分割路径（兼容反斜杠和正斜杠）
        parts = old_path.replace('\\', '/').split('/')
        
        if len(parts) >= 3:
            # parts[0]: horse_run
            # parts[1]: 001 (子目录名)
            # parts[2]: 000000.jpg (原文件名)
            
            base_dir = parts[0]      # horse_run
            subdir = parts[1]        # 001
            filename = parts[2]      # 000000.jpg
            
            # 获取文件名（不含扩展名）和扩展名
            name_without_ext = os.path.splitext(filename)[0]  # 000000
            ext = os.path.splitext(filename)[1]               # .jpg
            
            # 构建新路径: horse_run/000000/images/001.jpg
            new_path = f"{base_dir}/{name_without_ext}/images/{subdir}{ext}"
            
            frame['file_path'] = new_path
    
    return data

# 使用示例
if __name__ == "__main__":
    input_json_path="/home/momo/Desktop/horse/transforms_test_old.json"
    output_json_path="/home/momo/Desktop/horse/transforms_test.json"
    # 读取原始 JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 转换路径
    transformed_data = transform_file_paths(original_data)
    
    # 保存转换后的 JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)
    
    print("路径转换完成！")