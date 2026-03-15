"""
模型效果对比：官方预训练模型 vs 微调后模型
用法：
  python compare.py                  # 默认：只对比来自 new_images/ 的 val 样本
  python compare.py --all-val        # 对比 val.txt 中的全部图片
  python compare.py --all-val --max 50  # 全部 val，最多50张
  python compare.py img1.jpg img2.jpg   # 指定图片（无真实标签）
"""

import sys
import os
import argparse
from pathlib import Path
from paddlex import create_model

LABEL_MAP = {0: "0°", 1: "90°", 2: "180°", 3: "270°"}

NEW_IMAGES_DIR      = "./new_images"
VAL_TXT             = "./dataset/text_image_orientation/val.txt"
DATASET_IMAGES_DIR  = "./dataset/text_image_orientation"
FINETUNED_MODEL_DIR = "output/best_model/inference"


def get_new_image_stems() -> set[str]:
    """读取 new_images/ 目录，返回原始文件名的 stem 集合（如 {'img_82', 'img_84'}）"""
    if not os.path.isdir(NEW_IMAGES_DIR):
        return set()
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return {Path(f).stem for f in os.listdir(NEW_IMAGES_DIR)
            if Path(f).suffix.lower() in exts}


def load_val_samples(val_txt: str,
                     stems_filter: set[str] | None = None,
                     max_count: int | None = None) -> list[tuple[str, int]]:
    """
    从 val.txt 读取样本。
    stems_filter: 若指定，只保留文件名中包含该 stem 的条目（用于过滤 new_images）
    """
    samples = []
    base = os.path.abspath(DATASET_IMAGES_DIR)
    with open(val_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rel_path, label = parts[0], int(parts[1])
            if stems_filter is not None:
                filename = Path(rel_path).stem          # e.g. img_rot0_img_82
                # augment_rotation.py 命名规则: img_rot{angle}_{original_stem}
                # 取最后一个 _ 后的部分作为 original_stem
                original_stem = "_".join(filename.split("_")[2:])  # img_82
                if original_stem not in stems_filter:
                    continue
            abs_path = os.path.join(base, rel_path)
            samples.append((abs_path, label))
            if max_count and len(samples) >= max_count:
                break
    return samples


def predict_single(model, input_path: str) -> tuple[int | None, float | None]:
    def _first_val(v):
        """从 list/tuple/ndarray 中取第一个值；标量直接返回。"""
        if v is None:
            return None
        try:
            return v[0]
        except Exception:
            return v

    for res in model.predict(input_path, batch_size=1):
        # PaddleX 版本差异：有的返回 label_ids，有的返回 class_ids
        label_val = _first_val(res.get("label_ids"))
        if label_val is None:
            label_val = _first_val(res.get("class_ids"))
        score_val = _first_val(res.get("scores"))

        label = int(label_val) if label_val is not None else None
        score = float(score_val) if score_val is not None else None
        return label, score
    return None, None


def get_cmp_mark(
    off_label: int | None,
    off_score: float | None,
    ft_label: int | None,
    ft_score: float | None,
    gt_label: int,
) -> str:
    """对比标记规则：++ / --* / + / - / = / **"""
    off_ok = (off_label == gt_label)
    ft_ok = (ft_label == gt_label)

    if ft_ok and not off_ok:
        return "++"
    if off_ok and not ft_ok:
        return "--*"
    if off_ok and ft_ok:
        if ft_score is None or off_score is None:
            return "="
        if ft_score > off_score:
            return "+"
        if ft_score < off_score:
            return "-"
        return "="
    return "**"


def compare(samples: list[tuple[str, int]], has_gt: bool):
    print("正在加载官方预训练模型...")
    official_model = create_model(model_name="PP-LCNet_x1_0_doc_ori")

    finetuned_model = None
    if os.path.isdir(FINETUNED_MODEL_DIR):
        print(f"正在加载微调模型 ({FINETUNED_MODEL_DIR})...")
        finetuned_model = create_model(
            model_name="PP-LCNet_x1_0_doc_ori",
            model_dir=FINETUNED_MODEL_DIR,
        )
    else:
        print(f"未找到微调模型目录 ({FINETUNED_MODEL_DIR})，将仅运行官方模型。")
        print("请先完成训练并导出模型后再进行对比。\n")

    name_w = 35
    col_w  = 18
    gt_w   = 8
    cmp_w  = 8
    total_cols = name_w + (gt_w if has_gt else 0) + col_w + (col_w if finetuned_model else 0) + (cmp_w if has_gt and finetuned_model else 0)
    sep = "=" * (total_cols + 4)

    print("\n" + sep)
    header = f"{'图片':<{name_w}}"
    if has_gt:
        header += f" {'真实':^{gt_w}}"
    header += f" {'官方模型':^{col_w}}"
    if finetuned_model:
        header += f" {'微调模型':^{col_w}}"
    if has_gt and finetuned_model:
        header += f" {'对比':^{cmp_w}}"
    print(header)
    print(sep)

    off_correct = ft_correct = total = 0

    for input_path, gt_label in samples:
        name = os.path.basename(input_path)
        total += 1

        off_label, off_score = predict_single(official_model, input_path)
        off_score_text = f"{off_score:.3f}" if off_score is not None else "N/A"
        off_str = f"{LABEL_MAP.get(off_label, off_label)} ({off_score_text})"
        if has_gt and off_label == gt_label:
            off_correct += 1

        row = f"{name:<{name_w}}"
        if has_gt:
            row += f" {LABEL_MAP.get(gt_label, gt_label):^{gt_w}}"
        row += f" {off_str:^{col_w}}"

        if finetuned_model:
            ft_label, ft_score = predict_single(finetuned_model, input_path)
            ft_score_text = f"{ft_score:.3f}" if ft_score is not None else "N/A"
            ft_str = f"{LABEL_MAP.get(ft_label, ft_label)} ({ft_score_text})"
            if has_gt and ft_label == gt_label:
                ft_correct += 1
            row += f" {ft_str:^{col_w}}"
            if has_gt:
                cmp_mark = get_cmp_mark(off_label, off_score, ft_label, ft_score, gt_label)
                row += f" {cmp_mark:^{cmp_w}}"
            elif off_label != ft_label:
                row += "  ← 结论不同"

        print(row)

    print(sep)
    print("格式：预测方向（置信度）")
    if has_gt and finetuned_model:
        print("对比列：++(微调对/官方错), --*(微调错需关注), +(都对且微调置信度更高), -(都对且更低), =(都对且置信度相同), **(都错)")

    if has_gt and total > 0:
        print(f"\n准确率汇总（共 {total} 张）：")
        print(f"  官方模型: {off_correct}/{total} = {off_correct/total*100:.1f}%")
        if finetuned_model:
            print(f"  微调模型: {ft_correct}/{total} = {ft_correct/total*100:.1f}%")
            diff = ft_correct / total * 100 - off_correct / total * 100
            print(f"  微调收益: {diff:+.1f}%")

    os.makedirs("output", exist_ok=True)
    first_path = samples[0][0]
    for res in official_model.predict(first_path, batch_size=1):
        res.save_to_img("./output/demo_official.png")
        res.save_to_json("./output/res_official.json")
    if finetuned_model:
        for res in finetuned_model.predict(first_path, batch_size=1):
            res.save_to_img("./output/demo_finetuned.png")
            res.save_to_json("./output/res_finetuned.json")
    print("\n第一张图的完整结果已保存至 output/ 目录。")


def main():
    parser = argparse.ArgumentParser(description="官方模型 vs 微调模型效果对比")
    parser.add_argument("inputs", nargs="*", help="指定图片路径或 URL，省略则使用 val.txt")
    parser.add_argument("--all-val", action="store_true",
                        help="对比 val.txt 中的全部图片（默认只对比 new_images 来源的样本）")
    parser.add_argument("--max", type=int, default=None,
                        help="最多加载的图片数（默认全部）")
    args = parser.parse_args()

    if args.inputs:
        samples = [(p, -1) for p in args.inputs]
        has_gt = False
    else:
        if not os.path.isfile(VAL_TXT):
            print(f"错误: 找不到 {VAL_TXT}")
            sys.exit(1)

        if args.all_val:
            stems_filter = None
            desc = "val.txt 全部"
        else:
            stems_filter = get_new_image_stems()
            if not stems_filter:
                print(f"警告: {NEW_IMAGES_DIR} 中未找到图片，回退到 val.txt 全部样本。")
                stems_filter = None
                desc = "val.txt 全部"
            else:
                desc = f"new_images 来源（{len(stems_filter)} 张原图）"

        samples = load_val_samples(VAL_TXT, stems_filter=stems_filter, max_count=args.max)
        has_gt = True
        print(f"对比范围：{desc}，从 val.txt 载入 {len(samples)} 张图片")

    if not samples:
        print("没有找到可对比的样本，请检查 val.txt 和 new_images/ 目录。")
        sys.exit(1)

    compare(samples, has_gt)


if __name__ == "__main__":
    main()
