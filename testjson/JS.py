import os
import json
from pathlib import Path
from tqdm import tqdm

def convert_image_text_pairs_to_json(root_dir):
    """
    将图像文本对转换为JSON格式
    
    参数:
    root_dir: 包含图像和文本对的根目录路径
    """
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'}
    
    # 初始化全局image_id计数器
    global_image_id = 0
    
    # 遍历根目录下的所有子目录
    for item in range(0,2):
        item = str(item)
        item_path = os.path.join(root_dir, item)
        
        # 只处理数字命名的文件夹
        if os.path.isdir(item_path) and item.isdigit():
            print(f"\n处理文件夹: {item}")
            
            # 准备输出文件路径
            output_file = f"pair_{item}.json"
            pairs = []
            
            # 构建文本和图像的基路径
            caps_base = os.path.join(item_path, "caps")
            images_base = os.path.join(item_path, "filtered_images")
            
            # 检查必要的目录是否存在
            if not os.path.exists(caps_base) or not os.path.exists(images_base):
                print(f"警告: {item} 目录中缺少 caps 或 filtered_images 文件夹")
                continue
            
            # 获取所有类别文件夹
            categories = []
            for category in os.listdir(caps_base):
                caps_category_path = os.path.join(caps_base, category)
                if os.path.isdir(caps_category_path):
                    categories.append(category)
            
            # 使用tqdm显示进度条，遍历所有类别
            for category in tqdm(categories, desc=f"处理 {item} 文件夹的类别"):
                caps_category_path = os.path.join(caps_base, category)
                images_category_path = os.path.join(images_base, category)
                
                # 确保对应的图像类别目录存在
                if not os.path.exists(images_category_path):
                    print(f"警告: 图像目录 {images_category_path} 不存在")
                    continue
                
                # 处理每个文本文件
                for txt_file in os.listdir(caps_category_path):
                    if txt_file.endswith('.txt'):
                        # 构建文本文件路径并读取内容
                        txt_path = os.path.join(caps_category_path, txt_file)
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                        
                        # 获取文本文件的基本名称（不含扩展名）
                        base_name = os.path.splitext(txt_file)[0]
                        
                        # 查找对应的图像文件
                        image_file = None
                        for file in os.listdir(images_category_path):
                            file_base, file_ext = os.path.splitext(file)
                            if file_base == base_name and file_ext.lower() in image_extensions:
                                image_file = file
                                break
                        
                        # 如果没有找到对应的图像文件
                        if image_file is None:
                            print(f"警告: 未找到 {base_name} 对应的图像文件")
                            continue
                        
                        # 构建完整的图像路径
                        image_path = os.path.join(images_category_path, image_file)
                        
                        # 获取相对于脚本目录的图像路径
                        rel_image_path = os.path.relpath(image_path, script_dir)
                        
                        # 创建数据对，使用全局递增的image_id
                        pair = {
                            "image": rel_image_path,
                            "caption": [caption],
                            "image_id": global_image_id  # 使用全局计数器
                        }
                        
                        pairs.append(pair)
                        global_image_id += 1  # 递增全局计数器
            
            # 将结果写入JSON文件
            with open(output_file, 'w', encoding='utf-8') as f:
                for pair in pairs:
                    json_line = json.dumps(pair, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            print(f"已完成: {output_file}, 共 {len(pairs)} 对数据")

# 使用示例
if __name__ == "__main__":
    root_directory = "testset"  # 替换为您的实际路径
    convert_image_text_pairs_to_json(root_directory)