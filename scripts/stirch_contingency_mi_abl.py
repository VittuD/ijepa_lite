"""
Stitch marginal-ablation viz outputs into a contingency table image.

Layout:
         learned                sincos
H_cond=OFF, floor=OFF   [img]  [img]
H_cond=ON,  floor=OFF   [img]  [img]
H_cond=OFF, floor=ON    [img]  [img]
H_cond=ON,  floor=ON    [img]  [img]

Usage:
  python scripts/stitch_contingency.py
  python scripts/stitch_contingency.py --date 2026-03-12 --epoch 00999 --out out.png
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── layout ────────────────────────────────────────────────────────────────────

GRID = [
    # (row_label,               exp_learned,                      exp_sincos)
    ("H_cond=OFF\nfloor=OFF",  "mi_HmargBS_learned",             "mi_HmargBS_sincos"),
    ("H_cond=ON\nfloor=OFF",   "mi_HcondHmargBS_learned",        "mi_HcondHmargBS_sincos"),
    ("H_cond=OFF\nfloor=ON",   "mi_HmargBS_floor_learned",       "mi_HmargBS_floor_sincos"),
    ("H_cond=ON\nfloor=ON",    "mi_HcondHmargBS_floor_learned",  "mi_HcondHmargBS_floor_sincos"),
]
COL_LABELS = ["learned", "sincos"]

# ── helpers ───────────────────────────────────────────────────────────────────

def find_img(base: Path, exp_name: str, epoch: str) -> Path:
    pattern = f"*_{exp_name}/rank0/viz_output_{exp_name}/epoch_{epoch}/stl10_test.png"
    matches = sorted(base.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No image found for '{exp_name}' epoch {epoch}\n"
            f"  searched: {base / pattern}"
        )
    return matches[-1]


def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",  default="2026-03-12")
    parser.add_argument("--epoch", default="00999")
    parser.add_argument("--out",   default=None)
    args = parser.parse_args()

    base = Path("outputs") / args.date
    out_path = Path(args.out) if args.out else base / f"contingency_{args.epoch}.png"

    # load all images
    imgs: dict[str, Image.Image] = {}
    missing = []
    for _, exp_l, exp_s in GRID:
        for exp_name in (exp_l, exp_s):
            try:
                imgs[exp_name] = Image.open(find_img(base, exp_name, args.epoch))
                print(f"  ✓  {exp_name}")
            except FileNotFoundError as e:
                print(f"  ✗  {e}")
                missing.append(exp_name)

    if missing:
        print(f"\n{len(missing)} image(s) missing — aborting.")
        raise SystemExit(1)

    # dimensions
    sample      = next(iter(imgs.values()))
    W, H        = sample.size
    LABEL_W     = 160
    HEADER_H    = 50
    PAD         = 6
    n_rows      = len(GRID)
    n_cols      = len(COL_LABELS)

    total_w = LABEL_W + PAD + n_cols * W + (n_cols - 1) * PAD + PAD
    total_h = HEADER_H + PAD + n_rows * H + (n_rows - 1) * PAD + PAD

    canvas = Image.new("RGB", (total_w, total_h), (230, 230, 230))
    draw   = ImageDraw.Draw(canvas)
    font_h = load_font(18)   # column headers
    font_r = load_font(14)   # row labels

    # column headers
    for c, label in enumerate(COL_LABELS):
        cx = LABEL_W + PAD + c * (W + PAD) + W // 2
        draw.text((cx, HEADER_H // 2), label, fill=(0, 0, 0),
                  font=font_h, anchor="mm")

    # rows
    for r, (row_label, exp_l, exp_s) in enumerate(GRID):
        y = HEADER_H + PAD + r * (H + PAD)

        # row label (vertically centred, two lines)
        draw.text((LABEL_W // 2, y + H // 2), row_label, fill=(0, 0, 0),
                  font=font_r, anchor="mm", align="center")

        # images
        for c, exp_name in enumerate((exp_l, exp_s)):
            x = LABEL_W + PAD + c * (W + PAD)
            canvas.paste(imgs[exp_name], (x, y))

    canvas.save(out_path)
    print(f"\nSaved → {out_path}  ({total_w}×{total_h})")


if __name__ == "__main__":
    main()
