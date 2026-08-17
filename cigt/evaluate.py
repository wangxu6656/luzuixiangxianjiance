import os
import sys

import cv2
import numpy as np

from detector import DetectionConfig, FilterDetector


def read(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def evaluate(folder, expect, det):
    files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    correct = 0
    rows = []
    for f in files:
        img = read(os.path.join(folder, f))
        r = det.detect(img)
        got = r.has_flavor_line
        ok = got == expect
        correct += ok
        rows.append((f, r.filter_found, got, r.offset_ratio, r.reject, ok))
    for f, ff, got, off, rej, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {f}: filter={ff} line={got} offset={off} reject={rej}")
    print(f"  {folder}: {correct}/{len(rows)} correct\n")
    return correct, len(rows)


def main():
    det = FilterDetector(DetectionConfig(min_area=30))
    total_c, total_n = 0, 0
    for folder, expect in [("picthave", True), ("picno", False)]:
        c, n = evaluate(folder, expect, det)
        total_c += c
        total_n += n
    acc = total_c / max(total_n, 1) * 100
    print(f"TOTAL: {total_c}/{total_n} = {acc:.1f}%")


if __name__ == "__main__":
    sys.exit(main())