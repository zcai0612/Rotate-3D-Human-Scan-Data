from PIL import Image
import os

def resize_image(input_path, output_path, size=(512, 512)):
    with Image.open(input_path) as img:
        # 保持纵横比的情况下调整大小
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # 创建一个新的512x512白色背景图像
        new_img = Image.new('RGBA', size, (255, 255, 255, 0))
        
        # 将调整后的图像粘贴到中心位置
        x = (size[0] - img.size[0]) // 2
        y = (size[1] - img.size[1]) // 2
        new_img.paste(img, (x, y))
        
        # 保存图片
        new_img.save(output_path, 'PNG')

def process_directory(directory):
    # 确保输出目录存在
    output_dir = os.path.join(directory, 'resized')
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理目录中的所有PNG文件
    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(output_dir, filename)
            
            print(f"Processing {filename}...")
            resize_image(input_path, output_path)
            print(f"Saved resized image to {output_path}")

if __name__ == "__main__":
    assets_dir = "assets"
    process_directory(assets_dir)
    print("All images have been resized to 512x512!") 