"""自动标注脚本。

基于已手动标注的图片（Pascal VOC XML 格式），训练 YOLO 模型，
然后用模型对未标注的图片进行推理，生成同格式的 XML 标注文件。

流程：
  1. 清空 auto_work/ 工作目录（每次训练都重新开始）
  2. 扫描 raw/ 目录，找出有 XML 标注的图片作为训练集
  3. 将训练数据复制到 auto_work/，转 Pascal VOC XML → YOLO 格式
  4. 用预训练 YOLOv8n 在 auto_work/ 中微调
  5. 对无标注图片进行推理
  6. 将推理结果转为 Pascal VOC XML 保存到 raw/ 目录

auto_work/ 是临时训练目录，每次运行都会自动清空重建，无需手动管理。

依赖：pip install ultralytics
"""

# ============================================================
# 配置区域 - 修改这里的参数来控制行为
# ============================================================

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Set

# 类别映射（与 data.yaml 保持一致）
# 数字 → 类别名，用于训练时指定 YOLO 类别 ID
CLASSES = {0: "floor", 1: "monster", 2: "rope"}

# 类别名 → 数字，用于解析 XML 时把类别名转为 ID
NAME_TO_ID = {v: k for k, v in CLASSES.items()}

# 推理置信度阈值：低于此值的目标会被丢弃
# 调低 → 更多框（可能误检），调高 → 更精准（可能漏检）
CONFIDENCE_THRESHOLD = 0.3

# 推理数量限制：只对最后 N 张未标注图片进行推理，None = 全部
MAX_PREDICT = None

# 训练轮数：数据少时适当增大，数据多时减小
# 26 张数据建议 50-100 轮
EPOCHS = 100

# 是否跳过训练，直接用已有模型推理
# 设为 True 时：不清空 auto_work，不复训练，直接用上次的 best.pt 推理
# 适用场景：训练完看效果不满意，调了置信度阈值想重新推理
SKIP_TRAIN = False

# 训练设备：0 = 第一块 CUDA GPU，"cpu" = 仅用 CPU
# ultralytics 会自动检测 CUDA，这里显式指定确保使用 GPU
DEVICE = "0"

# ============================================================


def parse_voc_xml(xml_path: str) -> Tuple[List[Tuple[int, float, float, float, float]], int, int]:
    """解析 Pascal VOC XML，返回 (boxes, img_w, img_h)，boxes 中坐标为归一化值。

    Args:
        xml_path: XML 文件路径

    Returns:
        boxes: [(class_id, cx, cy, w, h), ...]，坐标已归一化到 [0, 1]
        img_w: 图片宽度
        img_h: 图片高度
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 读取图片尺寸
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        # 类别名
        name = obj.find("name").text
        if name not in NAME_TO_ID:
            continue  # 跳过未知类别
        cls_id = NAME_TO_ID[name]

        # 边界框坐标（像素值）
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # 转为 YOLO 格式：中心点 + 宽高，归一化
        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        boxes.append((cls_id, cx, cy, w, h))
    return boxes, img_w, img_h


def save_yolo_label(txt_path: str, boxes: List[Tuple[int, float, float, float, float]]):
    """保存 YOLO 格式标注文件。

    每行格式：class_id cx cy w h
    坐标均为归一化值（0~1），6 位小数。
    """
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for cls_id, cx, cy, w, h in boxes:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def prepare_training_data(raw_dir: str, work_dir: str) -> List[str]:
    """将手动标注数据转为 YOLO 训练格式，返回已标注的 stem 列表。

    只读取 raw/ 下的 XML 文件，不会修改原始标注。
    训练数据（图片 + YOLO txt）保存到 work_dir 下。
    """
    raw = Path(raw_dir)
    work = Path(work_dir)

    # 训练图片和标注的输出目录
    img_dir = work / "images" / "train"
    lbl_dir = work / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    annotated = []
    for xml_file in sorted(raw.glob("*.xml")):
        # 找到对应的图片文件（先试 png，再试 jpg）
        img_file = xml_file.with_suffix(".png")
        if not img_file.exists():
            img_file = xml_file.with_suffix(".jpg")
        if not img_file.exists():
            print(f"[警告] 找不到图片: {xml_file.stem}")
            continue

        # 解析 XML 获取标注框
        boxes, _, _ = parse_voc_xml(str(xml_file))
        if not boxes:
            print(f"[警告] {xml_file.name} 无有效标注")
            continue

        # 复制图片到训练目录，生成 YOLO 格式标注文件
        shutil.copy2(str(img_file), str(img_dir / img_file.name))
        save_yolo_label(str(lbl_dir / (xml_file.stem + ".txt")), boxes)
        annotated.append(xml_file.stem)

    print(f"[数据] 训练集: {len(annotated)} 张已标注图片")
    return annotated


def create_data_yaml(work_dir: str):
    """生成 YOLO data.yaml 配置文件，指定训练数据路径和类别名。"""
    yaml_path = Path(work_dir) / "data.yaml"
    content = [
        f"path: {Path(work_dir).resolve().as_posix()}",
        "train: images/train",
        "val: images/train",  # 数据少，验证集也用训练集
        "",
        "names:",
    ]
    for idx, name in sorted(CLASSES.items()):
        content.append(f"  {idx}: {name}")
    content.append(f"nc: {len(CLASSES)}")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content) + "\n")
    print(f"[配置] data.yaml 已生成: {yaml_path}")


def train_model(work_dir: str, model_path: str, epochs: int = 50):
    """训练 YOLO 模型，返回 best.pt 路径。

    使用预训练权重微调，训练结果保存在 work_dir/runs/ 下。
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 未安装 ultralytics，请先运行: pip install ultralytics")
        return None

    data_yaml = str(Path(work_dir) / "data.yaml")
    model = YOLO(model_path)

    # 开始训练（使用 CUDA 加速）
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,           # 输入图片尺寸
        batch=8,             # 批次大小（显存不够可调小）
        device=DEVICE,       # 使用 CUDA GPU 训练
        project=str(Path(work_dir) / "runs"),
        name="auto_annotate",
        exist_ok=True,       # 覆盖已有的同名训练结果
        verbose=True,
    )

    best_pt = Path(work_dir) / "runs" / "auto_annotate" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"[训练] 完成，模型保存在: {best_pt}")
        return str(best_pt)
    else:
        print("[错误] 训练未生成 best.pt")
        return None


def predict_and_save_xml(model_path: str, raw_dir: str, annotated_stems: Set[str],
                         max_predict: int = None):
    """对未标注图片进行推理，保存为 Pascal VOC XML。

    自动跳过已有 XML 的图片，不会覆盖手动标注。
    新生成的 XML 保存到 raw/ 目录，与图片同名。

    Args:
        model_path: 训练好的 best.pt 模型路径
        raw_dir: 原始数据目录
        annotated_stems: 已有 XML 的文件名集合（这些会被跳过）
        max_predict: 最多推理多少张，None 为全部
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 未安装 ultralytics")
        return

    import cv2

    raw = Path(raw_dir)
    model = YOLO(model_path)

    # 收集所有图片的 stem（不含扩展名）
    all_pngs = set(p.stem for p in raw.glob("*.png"))
    all_jpgs = set(p.stem for p in raw.glob("*.jpg"))
    all_images = all_pngs | all_jpgs

    # 差集：有图片但没有 XML 的 → 就是需要自动标注的
    unannotated = sorted(all_images - annotated_stems)

    total = len(unannotated)
    if max_predict is not None and max_predict > 0:
        unannotated = unannotated[-max_predict:]
        print(f"[推理] 待标注图片: {total} 张，本次推理最后 {len(unannotated)} 张")
    else:
        print(f"[推理] 待标注图片: {len(unannotated)} 张")

    for stem in unannotated:
        # 找到图片文件
        img_file = raw / (stem + ".png")
        if not img_file.exists():
            img_file = raw / (stem + ".jpg")
        if not img_file.exists():
            continue

        # 读取图片
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"[警告] 无法读取图片: {img_file.name}")
            continue
        h, w = img.shape[:2]

        # YOLO 推理（使用 CUDA 加速）
        results = model(img, device=DEVICE, verbose=False)
        result = results[0]

        # 没有检测到任何目标 → 保存空 XML
        if result.boxes is None or len(result.boxes) == 0:
            save_empty_xml(str(raw / (stem + ".xml")), stem, w, h)
            continue

        # 遍历所有检测框，过滤低置信度结果
        boxes_data = result.boxes
        objects = []
        for box in boxes_data:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue  # 低于阈值，丢弃
            cls_id = int(box.cls[0])
            if cls_id not in CLASSES:
                continue  # 未知类别，丢弃
            xyxy = box.xyxy[0].cpu().numpy()
            xmin = int(xyxy[0])
            ymin = int(xyxy[1])
            xmax = int(xyxy[2])
            ymax = int(xyxy[3])
            objects.append((CLASSES[cls_id], xmin, ymin, xmax, ymax))

        # 保存为 Pascal VOC XML（与手动标注格式一致）
        save_voc_xml(str(raw / (stem + ".xml")), stem, w, h, objects)

    print(f"[推理] 完成，已生成 {len(unannotated)} 个 XML 标注文件")


def save_voc_xml(xml_path: str, stem: str, img_w: int, img_h: int,
                 objects: List[Tuple[str, int, int, int, int]]):
    """保存 Pascal VOC 格式 XML 文件，与手动标注的格式完全一致。

    Args:
        xml_path: 保存路径
        stem: 文件名（不含扩展名）
        img_w: 图片宽度
        img_h: 图片高度
        objects: [(类别名, xmin, ymin, xmax, ymax), ...]
    """
    # 根节点
    annotation = ET.Element("annotation")

    # 基本信息
    ET.SubElement(annotation, "folder").text = "raw"
    ET.SubElement(annotation, "filename").text = stem + ".png"
    ET.SubElement(annotation, "path").text = str(
        Path(xml_path).parent / (stem + ".png"))

    # 来源信息
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    # 图片尺寸
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_w)
    ET.SubElement(size, "height").text = str(img_h)
    ET.SubElement(size, "depth").text = "3"

    # 是否分割（目标检测不需要）
    ET.SubElement(annotation, "segmented").text = "0"

    # 每个目标对象
    for name, xmin, ymin, xmax, ymax in objects:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        # 边界框
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)

    # 写入文件，用 tab 缩进，保持与手动标注格式一致
    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="\t")
    with open(xml_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    print(f"  [保存] {Path(xml_path).name} ({len(objects)} 个目标)")


def save_empty_xml(xml_path: str, stem: str, img_w: int, img_h: int):
    """保存空标注 XML（该图片没有检测到任何目标）。"""
    save_voc_xml(xml_path, stem, img_w, img_h, [])
    print(f"  [保存] {Path(xml_path).name} (0 个目标)")


def main():
    """主流程：清空工作区 → 扫描训练集 → 复制数据 → 训练 → 推理。"""
    # 路径计算：脚本在 train/scripts/ 下，base_dir = train/
    base_dir = Path(__file__).resolve().parent.parent.parent  # main.py → 01_auto_annotate → scripts → train/
    raw_dir = base_dir / "data" / "raw"          # 原始数据目录（图片 + XML）
    work_dir = base_dir / "scripts" / "01_auto_annotate" / "auto_work"   # 临时训练工作目录（每次自动清空）
    model_path = base_dir / "model" / "yolov8n.pt"  # 预训练模型

    # 打印当前配置
    print("=" * 60)
    print("自动标注流程")
    print(f"原始数据目录: {raw_dir}")
    print(f"临时工作目录: {work_dir}（每次自动清空重建）")
    print(f"预训练模型:   {model_path}")
    print(f"置信度阈值:   {CONFIDENCE_THRESHOLD}")
    print(f"最大推理数:   {MAX_PREDICT if MAX_PREDICT else '全部'}")
    print(f"训练轮数:     {EPOCHS}")
    print(f"跳过训练:     {SKIP_TRAIN}")
    print(f"训练设备:     {'CUDA (GPU)' if DEVICE != 'cpu' else 'CPU'}")
    print("=" * 60)

    # 检查 CUDA 是否可用
    try:
        import torch
        if DEVICE != "cpu" and not torch.cuda.is_available():
            print("[警告] CUDA 不可用，将回退到 CPU 训练")
            print("请检查: 1) NVIDIA 驱动  2) CUDA Toolkit  3) pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        elif DEVICE != "cpu" and torch.cuda.is_available():
            print(f"[CUDA] 可用，设备: {torch.cuda.get_device_name(0)}")
        else:
            print("[设备] 使用 CPU 训练（无 CUDA）")
    except ImportError:
        print("[警告] 未安装 torch，无法检测 CUDA 状态")

    # 检查预训练模型是否存在
    if not model_path.exists():
        print(f"[错误] 预训练模型不存在: {model_path}")
        print("请先下载 YOLOv8n 模型到 train/model/ 目录")
        return

    # 步骤0：扫描 raw/ 目录，收集已有 XML 标注的文件名
    # 这些是手动标注的 → 作为训练集；不会被自动标注覆盖
    xml_files = list(raw_dir.glob("*.xml"))
    annotated_stems = {f.stem for f in xml_files}
    print(f"\n[统计] raw/ 目录下已有标注: {len(annotated_stems)} 张")

    if len(annotated_stems) == 0:
        print("[错误] 没有找到任何 XML 标注文件，无法训练")
        return

    # 步骤1：清空临时工作目录，准备全新训练
    if not SKIP_TRAIN:
        print("\n[步骤1] 清空工作目录，准备训练数据...")
        if work_dir.exists():
            shutil.rmtree(str(work_dir))
            print(f"[清理] 已删除旧的 {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # 将 raw/ 下有 XML 的图片复制到 auto_work/，并生成 YOLO 格式标注
        prepare_training_data(str(raw_dir), str(work_dir))
        create_data_yaml(str(work_dir))
    else:
        print("\n[步骤1] 跳过训练，使用已有模型...")

    best_pt = Path(work_dir) / "runs" / "auto_annotate" / "weights" / "best.pt"

    if not SKIP_TRAIN:
        # 步骤2：在 auto_work/ 中训练模型
        print("\n[步骤2] 训练模型...")
        best_pt = train_model(str(work_dir), str(model_path), epochs=EPOCHS)
        if best_pt is None:
            print("[错误] 训练失败，终止")
            return
    else:
        if not best_pt.exists():
            print("[错误] 未找到已有模型，请先运行一次完整训练（SKIP_TRAIN=False）")
            return
        print(f"[复用] 已有模型: {best_pt}")

    # 步骤3：对 raw/ 中未标注的图片进行推理，生成 XML 保存到 raw/
    print("\n[步骤3] 推理未标注图片...")
    predict_and_save_xml(str(best_pt), str(raw_dir), annotated_stems, MAX_PREDICT)

    print("\n" + "=" * 60)
    print("自动标注完成！")
    print(f"新生成的 XML 已保存到: {raw_dir}")
    print("建议人工抽查部分自动标注结果，确认后继续标注更多图片再重新训练。")
    print("=" * 60)


if __name__ == "__main__":
    main()