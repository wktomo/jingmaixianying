import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from skimage.morphology import skeletonize

def generate_step3_conditions(input_mask_dir, output_cond_dir):
    """
    将 Step 2 的单通道掩码转换为 Step 3 训练所需的 3通道条件图
    通道分配逻辑:
    - Channel 0 (R): 血管边缘 (Edge) -> 约束轮廓
    - Channel 1 (G): 血管骨架 (Skeleton) -> 保持拓扑连通
    - Channel 2 (B): 血管掩码 (Mask) -> 确定基本区域
    """
    input_path = Path(input_mask_dir)
    output_path = Path(output_cond_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 支持常见的图片格式
    mask_files = sorted([f for f in input_path.glob("*.png")])
    
    if not mask_files:
        print(f"错误: 在 {input_mask_dir} 路径下未找到任何 .png 掩码文件！")
        return

    print(f"开始转换数据: 发现 {len(mask_files)} 个掩码文件...")

    for img_path in tqdm(mask_files):
        # 1. 读取掩码并二值化
        mask = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 2. 生成血管边缘 (Edge) - 使用 Canny 算子
        edge = cv2.Canny(binary_mask, 100, 200)

        # 3. 生成血管骨架 (Skeleton) - 使用形态学骨架化
        # 注意：skimage 需要 0-1 范围的布尔/浮点输入
        skeleton_input = binary_mask / 255.0
        skeleton = skeletonize(skeleton_input)
        skeleton = (skeleton * 255).astype(np.uint8)

        # 4. 合成 3 通道图像 (按照 RGB 顺序，OpenCV 写入时需注意通道顺序)
        # 项目约定顺序: [Edge, Skeleton, Mask] -> 对应 RGB
        # OpenCV 默认 BGR 写入，所以顺序设为 (Mask, Skeleton, Edge) 这样保存后就是 RGB 对应关系
        condition_img = cv2.merge([binary_mask, skeleton, edge])

        # 保存结果
        save_name = img_path.name
        cv2.imwrite(str(output_path / save_name), condition_img)

    print(f"转换完成！中间产物已存至: {output_cond_dir}")

if __name__ == "__main__":
    # --- 请根据你的实际目录修改此处 ---
    # 输入：你之前 inference.py 生成的 masks 文件夹
    INPUT_DIR = "output_results/masks" 
    
    # 输出：准备用于 Step 3 训练的 conditions 文件夹
    OUTPUT_DIR = "data/processed/conditions/train"
    
    generate_step3_conditions(INPUT_DIR, OUTPUT_DIR)