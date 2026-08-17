from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectionConfig:
    blur_ksize: int = 7
    canny_low: int = 40
    canny_high: int = 140

    min_filter_area: int = 500
    min_filter_circularity: float = 0.3
    min_filter_coverage: float = 0.25

    use_color: bool = True
    use_dark: bool = True
    sat_threshold: int = 70
    dark_value: int = 110
    min_area: int = 30
    min_area_ratio: float = 0.004
    max_area_ratio: float = 0.25
    max_center_dist: float = 0.6
    morph_ksize: int = 3
    line_radius_ratio: float = 0.9

    reject_threshold: float = 0.25
    reject_missing_line: bool = True
    reject_no_filter: bool = True
    filter_radius_mm: float = 3.9

    hough_param1: int = 100
    hough_param2: int = 45
    min_filter_ratio: float = 0.04
    max_filter_ratio: float = 0.65


@dataclass
class EllipseInfo:
    cx: float
    cy: float
    rx: float
    ry: float
    angle: float
    area: float

    @property
    def radius(self) -> float:
        return (abs(self.rx) + abs(self.ry)) / 2.0


@dataclass
class FlavorLineResult:
    x: int
    y: int
    area: int
    rel_x: float
    rel_y: float
    offset_ratio: float
    offset_mm: float
    ellipse: Optional[EllipseInfo] = None


@dataclass
class DetectionResult:
    filter_found: bool = False
    filter_ellipse: Optional[EllipseInfo] = None
    has_flavor_line: bool = False
    lines: List[FlavorLineResult] = field(default_factory=list)
    offset_ratio: Optional[float] = None
    offset_mm: Optional[float] = None
    qualified: Optional[bool] = None
    reject: bool = False


class FilterDetector:
    def __init__(self, cfg: Optional[DetectionConfig] = None):
        self.cfg = cfg or DetectionConfig()

    def set_config(self, cfg: DetectionConfig) -> None:
        self.cfg = cfg

    def detect(self, image: np.ndarray) -> DetectionResult:
        result = DetectionResult()
        if image is None:
            return result
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        filtered = self._filter(gray)
        edges = self._edges(filtered)
        ellipse = self._find_filter_ellipse(filtered, edges)
        if ellipse is None:
            ellipse = self._find_filter_fallback(gray)
        if ellipse is None:
            result.reject = self.cfg.reject_no_filter
            return result

        result.filter_found = True
        result.filter_ellipse = ellipse
        result.lines = self._detect_lines(image, ellipse)
        result.has_flavor_line = len(result.lines) > 0

        if result.lines:
            main = min(result.lines, key=lambda ln: ln.offset_ratio)
            result.offset_ratio = main.offset_ratio
            result.offset_mm = main.offset_mm
            exceed = any(ln.offset_ratio > self.cfg.reject_threshold for ln in result.lines)
            result.qualified = not exceed
        else:
            result.qualified = not self.cfg.reject_missing_line
        result.reject = not result.qualified
        return result

    def _filter(self, gray: np.ndarray) -> np.ndarray:
        k = max(1, self.cfg.blur_ksize | 1)
        return cv2.GaussianBlur(gray, (k, k), 0)

    def _edges(self, filtered: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(filtered, self.cfg.canny_low, self.cfg.canny_high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        return edges

    def _fit_ellipse(self, contour: np.ndarray) -> Optional[tuple]:
        if len(contour) < 5:
            return None
        try:
            return cv2.fitEllipse(contour)
        except cv2.error:
            return None

    def _ellipse_to_info(self, ellipse: tuple) -> EllipseInfo:
        (cx, cy), (rx, ry), angle = ellipse
        return EllipseInfo(
            cx=float(cx),
            cy=float(cy),
            rx=float(rx) / 2.0,
            ry=float(ry) / 2.0,
            angle=float(angle),
            area=float(np.pi * rx * ry / 4.0),
        )

    def _coverage(self, edges: np.ndarray, ellipse: tuple) -> float:
        mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, 1)
        total = int((mask > 0).sum())
        if total == 0:
            return 0.0
        overlap = int(((edges > 0) & (mask > 0)).sum())
        return overlap / total

    def _find_filter_ellipse(self, filtered: np.ndarray, edges: np.ndarray) -> Optional[EllipseInfo]:
        h, w = filtered.shape
        min_dim = float(min(h, w))
        min_r = max(10.0, min_dim * self.cfg.min_filter_ratio)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cfg.min_filter_area:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perim * perim)
            if circularity < self.cfg.min_filter_circularity:
                continue
            ellipse = self._fit_ellipse(c)
            if ellipse is None:
                continue
            info = self._ellipse_to_info(ellipse)
            if info.radius < min_r:
                continue
            cov = self._coverage(edges, ellipse)
            if cov < self.cfg.min_filter_coverage:
                continue
            score = info.area * cov
            if score > best_score:
                best_score = score
                best = ellipse
        if best is None:
            return None
        info = self._ellipse_to_info(best)
        if not (0 < info.cx < w and 0 < info.cy < h):
            return None
        return info

    def _find_filter_fallback(self, gray: np.ndarray) -> Optional[EllipseInfo]:
        h, w = gray.shape
        target = 400.0
        scale = min(1.0, target / max(h, w))
        g = gray
        if scale < 1.0:
            g = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        gh, gw = g.shape
        g_blur = cv2.GaussianBlur(g, (self.cfg.blur_ksize, self.cfg.blur_ksize), 0)
        min_dim = float(min(h, w))
        min_r = max(5, int(min_dim * self.cfg.min_filter_ratio * scale))
        max_r = int(min_dim * self.cfg.max_filter_ratio * scale)
        best = None
        best_score = -1e9
        for param2 in (self.cfg.hough_param2, 30):
            circles = cv2.HoughCircles(
                g_blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=0.5 * min(gh, gw),
                param1=self.cfg.hough_param1, param2=param2,
                minRadius=min_r, maxRadius=max_r,
            )
            if circles is None:
                continue
            for (cx, cy, r) in circles[0, :]:
                cx, cy, r = float(cx), float(cy), float(r)
                interior = np.zeros(g.shape, dtype=np.uint8)
                cv2.circle(interior, (int(cx), int(cy)), int(r), 255, -1)
                ring = np.zeros(g.shape, dtype=np.uint8)
                cv2.circle(ring, (int(cx), int(cy)), int(r * 1.25), 255, -1)
                cv2.circle(ring, (int(cx), int(cy)), int(r * 0.85), 0, -1)
                inner = cv2.mean(g_blur, interior)[0]
                ring_mean = cv2.mean(g_blur, ring)[0]
                dark_center = self._has_dark_center(g_blur, cx, cy, r)
                score = inner - ring_mean + (40.0 if dark_center else 0.0)
                if score > best_score:
                    best_score = score
                    best = (cx / scale, cy / scale, r / scale)
        if best is None:
            return None
        cx, cy, r = best
        return EllipseInfo(cx=cx, cy=cy, rx=r, ry=r, angle=0.0, area=np.pi * r * r)

    def _has_dark_center(self, gray: np.ndarray, cx: float, cy: float, r: float) -> bool:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), max(1, int(r * 0.35)), 255, -1)
        return cv2.mean(gray, mask)[0] < 115

    def _ellipse_mask(self, shape, ellipse: tuple, scale: float = 1.0) -> np.ndarray:
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, -1)
        return mask

    def _detect_lines(self, image: np.ndarray, filter_ellipse: EllipseInfo) -> List[FlavorLineResult]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, sat, val = cv2.split(hsv)
        shape = image.shape[:2]

        inner = np.zeros(shape, dtype=np.uint8)
        cv2.ellipse(
            inner,
            (int(filter_ellipse.cx), int(filter_ellipse.cy)),
            (
                int(filter_ellipse.rx * self.cfg.line_radius_ratio),
                int(filter_ellipse.ry * self.cfg.line_radius_ratio),
            ),
            0,
            0,
            360,
            255,
            -1,
        )
        cand = np.zeros(shape, dtype=np.uint8)
        if self.cfg.use_dark:
            cand = np.maximum(cand, (val < self.cfg.dark_value).astype(np.uint8))
        if self.cfg.use_color:
            cand = np.maximum(cand, (sat > self.cfg.sat_threshold).astype(np.uint8))
        cand = cv2.bitwise_and(cand, inner)
        k = max(1, self.cfg.morph_ksize)
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

        num, _, stats, centroids = cv2.connectedComponentsWithStats(cand, 8)
        cx, cy = filter_ellipse.cx, filter_ellipse.cy
        radius = filter_ellipse.radius
        filter_area = np.pi * filter_ellipse.rx * filter_ellipse.ry
        lines: List[FlavorLineResult] = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            min_a = max(self.cfg.min_area, int(self.cfg.min_area_ratio * filter_area))
            if area < min_a:
                continue
            if filter_area > 0 and area > self.cfg.max_area_ratio * filter_area:
                continue
            bx, by = float(centroids[i, 0]), float(centroids[i, 1])
            offset = np.hypot(bx - cx, by - cy)
            if radius > 0 and offset / radius > self.cfg.max_center_dist:
                continue
            comp = (cand == i).astype(np.uint8)
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ellipse_info = None
            if contours:
                ell = self._fit_ellipse(contours[0])
                if ell is not None:
                    ellipse_info = self._ellipse_to_info(ell)
            offset_ratio = offset / radius if radius > 0 else 0.0
            offset_mm = offset_ratio * self.cfg.filter_radius_mm
            rel_x = (bx - cx) / radius if radius > 0 else 0.0
            rel_y = (by - cy) / radius if radius > 0 else 0.0
            lines.append(
                FlavorLineResult(
                    x=int(bx),
                    y=int(by),
                    area=area,
                    rel_x=rel_x,
                    rel_y=rel_y,
                    offset_ratio=float(offset_ratio),
                    offset_mm=float(offset_mm),
                    ellipse=ellipse_info,
                )
            )
        return lines

    def annotate(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        annotated = image.copy()
        if result.filter_ellipse is not None:
            ell = result.filter_ellipse
            cv2.ellipse(
                annotated,
                (int(ell.cx), int(ell.cy)),
                (int(ell.rx), int(ell.ry)),
                0,
                0,
                360,
                (0, 255, 0),
                2,
            )
            cv2.circle(annotated, (int(ell.cx), int(ell.cy)), 3, (0, 255, 0), -1)
        for line in result.lines:
            r = max(3, int(round(np.sqrt(line.area))))
            cv2.circle(annotated, (line.x, line.y), r, (0, 0, 255), -1)
            if line.ellipse is not None:
                cv2.ellipse(
                    annotated,
                    (int(line.ellipse.cx), int(line.ellipse.cy)),
                    (int(line.ellipse.rx), int(line.ellipse.ry)),
                    line.ellipse.angle,
                    0,
                    360,
                    (255, 0, 255),
                    1,
                )
        if result.filter_found:
            if result.has_flavor_line:
                status = "OK" if result.qualified else "REJECT"
                color = (0, 255, 0) if result.qualified else (0, 0, 255)
            else:
                status = "REJECT" if result.reject else "NO LINE"
                color = (0, 0, 255) if result.reject else (0, 255, 255)
            cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if result.offset_ratio is not None:
                cv2.putText(
                    annotated,
                    f"offset={result.offset_ratio:.3f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
        return annotated