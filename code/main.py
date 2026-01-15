import argparse
import time
from pathlib import Path
from typing import List, Dict

import cv2
from utils import load_image, split_image, apply_alignment, create_color_image, auto_crop
from alignment import align_bruteforce, pyramid_align
from enhancement import enhance_image

def process_image(
    input_path: str | Path,
    output_dir: str | Path,
    metric: str = "ncc",
    use_pyramid: bool = False
) -> Dict[str, object]:
    print("\n" + "=" * 60)
    print(f"Processing: {input_path}")
    print("=" * 60)
    start = time.time()

    # 1) Load & split
    stacked = load_image(str(input_path))
    b, g, r = split_image(stacked)
    print(f"Split shapes: {b.shape}")

    # 2) Align
    metric_name = metric.lower()
    if metric_name not in ("ssd", "ncc"):
        print(f"[warn] Unknown metric '{metric}', using 'ncc'.")
        metric_name = "ncc"

    if use_pyramid:
        dx_g, dy_g = pyramid_align(b, g, levels=5, base_search=4, refine_search=2, metric=metric_name)
        dx_r, dy_r = pyramid_align(b, r, levels=5, base_search=4, refine_search=2, metric=metric_name)
    else:
        dx_g, dy_g, _ = align_bruteforce(b, g, search_range=15, metric=metric_name, edge_crop=0.10)
        dx_r, dy_r, _ = align_bruteforce(b, r, search_range=15, metric=metric_name, edge_crop=0.10)

    print(f"  Green shift: dx={dx_g}, dy={dy_g}")
    print(f"  Red   shift: dx={dx_r}, dy={dy_r}")

    # 3) Apply shifts & compose
    g_al = apply_alignment(g, dx_g, dy_g)
    r_al = apply_alignment(r, dx_r, dy_r)
    unaligned_rgb = create_color_image(b, g, r)
    aligned_rgb   = create_color_image(b, g_al, r_al)

    # 4) Enhance
    enhanced_rgb = enhance_image(aligned_rgb, gamma_value=1.08, do_unsharp=True)

    # 5) Auto-crop
    cropped_rgb, crop_box = auto_crop(enhanced_rgb, thresh=12, max_crop=120)

    # 6) Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    cv2.imwrite(str(out / f"{stem}_unaligned.jpg"), cv2.cvtColor(unaligned_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / f"{stem}_aligned.jpg"),   cv2.cvtColor(aligned_rgb,   cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / f"{stem}_enhanced.jpg"),  cv2.cvtColor(cropped_rgb,   cv2.COLOR_RGB2BGR))

    elapsed = time.time() - start
    print(f"Done in {elapsed:.2f}s | crop={crop_box}")
    return {'image': stem, 'g_shift': (int(dx_g), int(dy_g)), 'r_shift': (int(dx_r), int(dy_r)), 'time': float(elapsed)}

def main() -> None:
    parser = argparse.ArgumentParser(description="Prokudin-Gorskii Reconstruction")
    parser.add_argument("--input", required=False, default="../data", help="Image file or folder")
    parser.add_argument("--output", default="../results", help="Output folder")
    parser.add_argument("--metric", default="ncc", choices=["ssd", "ncc"], help="Similarity metric")
    parser.add_argument("--pyramid", action="store_true", help="Use pyramid alignment (bonus)")
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_file():
        files = [inp]
    else:
        files = sorted(list(inp.glob("*.jpg")) + list(inp.glob("*.tif")))
        if not files:
            raise FileNotFoundError(f"No .jpg or .tif files found under: {inp}")

    results: List[Dict[str, object]] = []
    for f in files:
        results.append(process_image(str(f), args.output, metric=args.metric, use_pyramid=args.pyramid))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Image':<20} {'G Shift':<15} {'R Shift':<15} {'Time (s)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['image']:<20} {str(r['g_shift']):<15} {str(r['r_shift']):<15} {r['time']:<10.2f}")

if __name__ == "__main__":
    main()
