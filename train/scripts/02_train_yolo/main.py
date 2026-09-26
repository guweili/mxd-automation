"""YOLO 训练脚本。

从多个 raw/ 目录读取手动标注的 XML，复制到 auto_work/ 临时目录进行训练。
不会修改 raw/ 下的任何文件。

流程：
  1. 扫描各 raw/ 目录，找出有 XML 标注的图片，每个目录随机取 N 条
  2. 清空 auto_work/，将数据复制进去，转 Pascal VOC XML → YOLO 格式
  3. 按比例随机划分训练集和验证集（默认 80% 训练 / 20% 验证）
  4. 用预训练 YOLOv8n 在 auto_work/ 中训练
  5. 训练产出保存到 scripts/02_train_yolo/runs/，best.pt 复制到 scripts/02_train_yolo/model/

依赖：pip install ultralytics
"""

# ============================================================
# 配置区域
# ============================================================

import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# 项目根目录（往上 4 级：main.py → 02_train_yolo → scripts → train → 项目根）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(PROJECT_ROOT)

# 数据根目录
DATA_ROOT = os.path.join(PROJECT_ROOT, "train", "data")

# 怪物类别定义：每个文件夹对应一个类别
MONSTER_DIRS = [
    "raw_僵尸猴子",
    "raw_大幽灵",
    "raw_小幽灵",
    "raw_灰鳄鱼",
    "raw_猴子",
    "raw_石面人",
    "raw_蓝水灵",
    "raw_青蛇",
    "raw_鳄鱼",
]


# 目录名 → 类别 ID（跳过无标注的目录）
def build_class_mapping():
    cls_id = 0
    mapping = {}
    classes = {}
    for dirname in MONSTER_DIRS:
        folder = Path(DATA_ROOT) / dirname
        if not folder.exists():
            continue
        xml_files = list(folder.glob("*.xml"))
        if not xml_files:
            print(f"[跳过] {dirname} 无标注文件")
            continue
        monster_name = dirname.replace("raw_", "")
        mapping[dirname] = cls_id
        classes[cls_id] = monster_name
        cls_id += 1
    return mapping, classes


DIR_TO_CLASS_ID, CLASSES = build_class_mapping()
NAME_TO_ID = {v: k for k, v in CLASSES.items()}

# 训练轮数
EPOCHS = 150

# 批次大小（显存不够可调小，如 4 或 8）
BATCH = 16

# 输入图片尺寸
IMG_SIZE = 640

# 训练设备：0 = 第一块 CUDA GPU，"cpu" = 仅用 CPU
DEVICE = "0"

# 验证集比例（0.2 = 20% 数据用于验证）
VAL_SPLIT = 0.2

# 随机种子（保证每次划分结果一致）
SEED = 42

# 预训练模型路径
MODEL_PATH = "train/model/yolov8n.pt"

# 每个目录最多取多少条数据（None 表示全部）
SAMPLE_PER_DIR = None

# 工作目录（每次训练自动清空重建）
WORK_DIR = "train/scripts/02_train_yolo/auto_work"

# 训练输出目录
SCRIPT_DIR = str(Path(__file__).resolve().parent)
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
# 最终模型保存到 train/model/
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "train", "model")


def parse_voc_xml(xml_path: str, force_class_id: int = None):
    """解析 Pascal VOC XML，返回 (boxes, img_w, img_h)。

    boxes: [(class_id, cx, cy, w, h), ...]，坐标已归一化到 [0, 1]
    若 force_class_id 不为 None，则所有目标都使用该类别 ID（按文件夹区分怪物种类）。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if force_class_id is not None:
            cls_id = force_class_id
        elif name not in NAME_TO_ID:
            continue
        else:
            cls_id = NAME_TO_ID[name]
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        boxes.append((cls_id, cx, cy, w, h))
    return boxes, img_w, img_h


def save_yolo_label(txt_path: str, boxes):
    """保存 YOLO 格式标注文件。每行：class_id cx cy w h"""
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for cls_id, cx, cy, w, h in boxes:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def prepare_data(work_dir: str, sample_per_dir=None):
    """从各怪物目录复制数据到 work/，按比例划分训练集和验证集，转 YOLO 格式。

    每个目录的所有目标都映射为该目录对应的怪物类别。
    sample_per_dir 为 None 时取该目录全部数据。
    返回 (训练集数量, 验证集数量)。
    """
    work = Path(work_dir)

    # 清空并重建工作目录
    if work.exists():
        shutil.rmtree(work)

    # 训练集和验证集目录
    train_img = work / "images" / "train"
    train_lbl = work / "labels" / "train"
    val_img = work / "images" / "val"
    val_lbl = work / "labels" / "val"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    # 从每个目录收集有效数据
    all_items = []
    for dirname, cls_id in DIR_TO_CLASS_ID.items():
        raw = Path(DATA_ROOT) / dirname
        items = []
        for xml_file in sorted(raw.glob("*.xml")):
            img_file = xml_file.with_suffix(".png")
            if not img_file.exists():
                img_file = xml_file.with_suffix(".jpg")
            if not img_file.exists():
                continue

            boxes, _, _ = parse_voc_xml(str(xml_file), force_class_id=cls_id)
            if not boxes:
                continue

            items.append((img_file, xml_file, boxes))

        # 随机采样
        random.seed(SEED)
        random.shuffle(items)
        if sample_per_dir is not None and len(items) > sample_per_dir:
            items = items[:sample_per_dir]

        print(f"[数据] {dirname} → 类别[{cls_id}]{CLASSES[cls_id]}: 选取 {len(items)} 张")
        all_items.extend(items)

    # 随机打乱后划分训练集/验证集
    random.seed(SEED)
    random.shuffle(all_items)
    split_idx = int(len(all_items) * (1 - VAL_SPLIT))
    train_items = all_items[:split_idx]
    val_items = all_items[split_idx:]

    # 复制到训练集目录
    valid_train = 0
    for img_file, xml_file, boxes in train_items:
        try:
            shutil.copy2(str(img_file), str(train_img / img_file.name))
            save_yolo_label(str(train_lbl / (xml_file.stem + ".txt")), boxes)
            valid_train += 1
        except Exception as e:
            print(f"[警告] 复制失败，跳过: {img_file.name} ({e})")

    # 复制到验证集目录
    valid_val = 0
    for img_file, xml_file, boxes in val_items:
        try:
            shutil.copy2(str(img_file), str(val_img / img_file.name))
            save_yolo_label(str(val_lbl / (xml_file.stem + ".txt")), boxes)
            valid_val += 1
        except Exception as e:
            print(f"[警告] 复制失败，跳过: {img_file.name} ({e})")

    print(f"[数据] 训练集: {valid_train} 张, 验证集: {valid_val} 张（共 {len(all_items)} 张）")
    return valid_train, valid_val


def create_data_yaml(work_dir: str):
    """生成 data.yaml 配置文件"""
    yaml_path = Path(work_dir) / "data.yaml"
    content = [
        f"path: {Path(work_dir).resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for idx, name in sorted(CLASSES.items()):
        content.append(f"  {idx}: {name}")
    content.append(f"nc: {len(CLASSES)}")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content) + "\n")
    print(f"[配置] data.yaml 已生成: {yaml_path}")


def train():
    """主入口：准备数据 → 训练 → 保存模型"""
    from ultralytics import YOLO

    print("=" * 60)
    print("YOLO 怪物分类训练")
    print(f"类别数: {len(CLASSES)}")
    for cid, cname in sorted(CLASSES.items()):
        print(f"  [{cid}] {cname}")
    print(f"每目录采样: {'全部' if SAMPLE_PER_DIR is None else SAMPLE_PER_DIR} 张")
    print(f"工作目录: {WORK_DIR}  (每次自动清空)")
    print(f"预训练模型: {MODEL_PATH}")
    print(f"训练轮数: {EPOCHS}")
    print(f"设备: {'CUDA GPU' if DEVICE != 'cpu' else 'CPU'}")
    print("=" * 60)

    print("\n[步骤1] 准备训练数据...")
    train_count, val_count = prepare_data(WORK_DIR, SAMPLE_PER_DIR)
    if train_count == 0:
        print("[错误] 没有可用的训练数据，请先手动标注一些图片")
        return
    create_data_yaml(WORK_DIR)
    data_yaml = str(Path(WORK_DIR) / "data.yaml")

    print("\n[步骤2] 开始训练...")
    model = YOLO(MODEL_PATH)
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project=RUNS_DIR,
        name="train",
        exist_ok=True,
        verbose=True,
        seed=SEED,
        workers=4,
        patience=20,
    )

    best_pt = Path(RUNS_DIR) / "train" / "weights" / "best.pt"
    if best_pt.exists():
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        dest = Path(MODEL_OUTPUT_DIR) / "best.pt"
        shutil.copy2(str(best_pt), str(dest))
        print(f"\n[完成] best.pt → {dest}")
    else:
        print("\n[警告] 未找到 best.pt，请检查训练输出")


if __name__ == "__main__":
    train()