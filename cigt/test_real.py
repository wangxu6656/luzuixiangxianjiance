import os
import sys

import cv2
import numpy as np

from detector import DetectionConfig, FilterDetector


def read(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    det = FilterDetector(DetectionConfig(min_area=30))
    os.makedirs("debug_out", exist_ok=True)
    for f in sorted(os.listdir("pict")):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = read(os.path.join("pict", f))
        r = det.detect(img)
        desc = ";".join(
            f"({ln.x},{ln.y}) off={ln.offset_ratio:.3f}" for ln in r.lines
        )
        print(
            f"{f}: filter={r.filter_found} line={r.has_flavor_line} "
            f"ellipse=({r.filter_ellipse.cx:.0f},{r.filter_ellipse.cy:.0f}) "
            f"rx={r.filter_ellipse.rx:.0f} ry={r.filter_ellipse.ry:.0f} "
            f"offset={r.offset_ratio} off_mm={r.offset_mm} "
            f"qualified={r.qualified} reject={r.reject} n={len(r.lines)} {desc}"
        )
        cv2.imencode(".jpg", det.annotate(img, r))[1].tofile(os.path.join("debug_out", "res_" + f))


if __name__ == "__main__":
    sys.exit(main())