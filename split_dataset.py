

import os
import random
import shutil

# ========== 配置参数 ==========
DATA_DIR = 'all_data'          # 原始数据文件夹（包含图片和对应标注txt）
OUTPUT_DIR = 'dataset'         # 输出目录
TRAIN_RATIO = 0.7              # 训练集比例
VAL_RATIO = 0.2                # 验证集比例
TEST_RATIO = 0.1               # 测试集比例
# ==============================

random.seed(42)

# 创建输出目录结构
for split in ['train', 'val', 'test']:
    os.makedirs(f'{OUTPUT_DIR}/{split}/images', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/{split}/labels', exist_ok=True)

# 获取所有图片文件
images = [f for f in os.listdir(DATA_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(images)

# 计算划分数量
n = len(images)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

splits = {
    'train': images[:train_end],
    'val': images[train_end:val_end],
    'test': images[val_end:]
}

# 复制文件
for split, files in splits.items():
    for img_file in files:
        # 图片
        shutil.copy(f'{DATA_DIR}/{img_file}',
                    f'{OUTPUT_DIR}/{split}/images/{img_file}')
        # 标签（同名txt文件）
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        label_path = f'{DATA_DIR}/{label_file}'
        if os.path.exists(label_path):
            shutil.copy(label_path,
                        f'{OUTPUT_DIR}/{split}/labels/{label_file}')

print('数据集划分完成！')
print(f'训练集: {len(splits["train"])} 张')
print(f'验证集: {len(splits["val"])} 张')
print(f'测试集: {len(splits["test"])} 张')