import numpy as np
import cv2
import sys

from detector import FilterDetector, DetectionConfig


def make_filter_img(size=400, line_pos=None, with_line=True, line_color=(40, 40, 140)):
    img = np.full((size, size, 3), 30, dtype=np.uint8)
    assert img.dtype == np.uint8
    cx, cy = size // 2, size // 2
    r = 150
    cv2.circle(img, (cx, cy), r + 12, (60, 60, 60), -1)
    cv2.circle(img, (cx, cy), r + 8, (120, 120, 120), -1)
    cv2.circle(img, (cx, cy), r, (235, 235, 235), -1)
    if with_line:
        px, py = (cx, cy) if line_pos is None else line_pos
        cv2.circle(img, (px, py), 30, line_color, -1)
    return img


def run_case(name, img, expect_line):
    det = FilterDetector(DetectionConfig(min_filter_area=500, min_area=10))
    res = det.detect(img)
    ok = res.has_flavor_line == expect_line
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: filter={res.filter_found} line={res.has_flavor_line} "
          f"(expect={expect_line}) lines={len(res.lines)}")
    if res.lines:
        ln = res.lines[0]
        print(f"        pos=({ln.x},{ln.y}) rel=({ln.rel_x:.2f},{ln.rel_y:.2f}) area={ln.area}")
    return ok


def main():
    results = []
    results.append(run_case("有香线-中心", make_filter_img(), True))
    results.append(run_case("有香线-偏移棕色", make_filter_img(line_pos=(260, 230), line_color=(60, 40, 20)), True))
    results.append(run_case("有香线-偏移红色", make_filter_img(line_pos=(150, 250), line_color=(30, 30, 200)), True))
    results.append(run_case("无香线", make_filter_img(with_line=False), False))

    det = FilterDetector(DetectionConfig(min_filter_area=500, min_area=10))
    annotated = det.annotate(make_filter_img(), det.detect(make_filter_img()))
    print("annotate output shape:", annotated.shape)
    cv2.imwrite("test_annotated.png", annotated)
    print("saved test_annotated.png")

    all_pass = all(results)
    print("ALL PASS" if all_pass else "SOME FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())