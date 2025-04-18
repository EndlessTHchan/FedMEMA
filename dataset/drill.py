import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm


def convert_brats_dataset(input_root, output_root):
    """
    将 BraTS2020 数据集从 NIfTI 格式转换为 NPY 格式

    参数:
    input_root: BraTS2020 训练集根目录
    output_root: 输出 NPY 文件的根目录
    """
    # 创建输出目录
    os.makedirs(os.path.join(output_root, "vol"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "seg"), exist_ok=True)

    # 获取所有病例文件夹
    patient_folders = sorted(
        glob.glob(os.path.join(input_root, "BraTS20_Validation_*"))
    )

    # 创建用于列出病例ID的文件
    all_cases = []

    for folder in tqdm(patient_folders, desc="转换病例"):
        # 从文件夹名称中提取病例ID
        folder_name = os.path.basename(folder)
        case_id = folder_name.split("_")[-1]  # 提取数字ID (例如 "001")
        all_cases.append(case_id)

        # 构建各模态文件的路径
        flair_path = os.path.join(folder, f"{folder_name}_flair.nii")
        t1_path = os.path.join(folder, f"{folder_name}_t1.nii")
        t1ce_path = os.path.join(folder, f"{folder_name}_t1ce.nii")
        t2_path = os.path.join(folder, f"{folder_name}_t2.nii")

        # 加载 NIfTI 文件
        flair = nib.load(flair_path).get_fdata()
        t1 = nib.load(t1_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()

        # 创建体积数据数组 (240x240x155x4)
        vol_data = np.zeros((240, 240, 155, 4), dtype=np.float32)

        # 填入各模态数据
        # 注：假设原始数据尺寸已经是 240x240x155
        vol_data[..., 0] = flair
        vol_data[..., 1] = t1ce
        vol_data[..., 2] = t1
        vol_data[..., 3] = t2

        # 保存为 NPY 文件
        np.save(os.path.join(output_root, "vol", f"{case_id}_vol.npy"), vol_data)

    # 创建数据集分割
    # create_data_splits(all_cases, output_root)

    print(f"转换完成! 共处理 {len(patient_folders)} 个病例")


def create_data_splits(case_ids, output_root, train_ratio=0.7, val_ratio=0.15):
    """创建训练/验证/测试数据集划分"""
    np.random.seed(42)  # 固定随机种子以确保可重复性

    # 随机打乱病例ID
    shuffled_ids = case_ids.copy()
    np.random.shuffle(shuffled_ids)

    # 计算各子集大小
    n_samples = len(shuffled_ids)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    # 划分数据集
    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train : n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val :]

    # 保存为文本文件
    with open(os.path.join(output_root, "train.txt"), "w") as f:
        f.write("\n".join(train_ids))

    with open(os.path.join(output_root, "val.txt"), "w") as f:
        f.write("\n".join(val_ids))

    with open(os.path.join(output_root, "test.txt"), "w") as f:
        f.write("\n".join(test_ids))

    print(
        f"数据集划分完成: 训练集 {len(train_ids)}, 验证集 {len(val_ids)}, 测试集 {len(test_ids)}"
    )


if __name__ == "__main__":
    # 输入和输出路径
    input_root = r"F:\download\\archive\BraTS2020_ValidationData\\MICCAI_BraTS2020_ValidationData"
    output_root = r"E:\datasets\\test"

    # 运行转换
    convert_brats_dataset(input_root, output_root)
