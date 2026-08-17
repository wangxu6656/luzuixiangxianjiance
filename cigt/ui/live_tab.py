import time
from typing import Optional

import cv2
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from detector import FilterDetector
from history.storage import HistoryDB
from plc.reject_controller import RejectController

from .config_panel import ConfigPanel, cv_to_qpixmap, list_cameras


class VideoThread(QThread):
    frame_ready = pyqtSignal(object, object)

    def __init__(self, camera_index: int, config_getter, parent: QWidget | None = None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.config_getter = config_getter
        self.detector = FilterDetector(config_getter())
        self._stop = False

    def stop(self) -> None:
        self._stop = True
        self.wait()

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.open(self.camera_index)
        while not self._stop:
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                self.detector.set_config(self.config_getter())
                result = self.detector.detect(frame)
                annotated = self.detector.annotate(frame, result)
                self.frame_ready.emit(annotated, result)
            except Exception:
                pass
            self.msleep(1)
        cap.release()


class LiveTab(QWidget):
    def __init__(self, db: HistoryDB, parent: QWidget | None = None):
        super().__init__(parent)
        self.thread: Optional[VideoThread] = None
        self.db = db
        self.plc: Optional[RejectController] = None
        self.reject_count = 0
        self._last_history = 0.0
        self._build()

    def _build(self) -> None:
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.video_label = QLabel("未启动")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setAlignment(self.video_label.alignment())
        left.addWidget(self.video_label, 1)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setMaximumHeight(140)
        left.addWidget(self.result_box)

        self.stat_label = QLabel("剔除计数: 0")
        left.addWidget(self.stat_label)

        root.addLayout(left, 1)

        right = QVBoxLayout()
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("摄像头"))
        self.cam_combo = QComboBox()
        self.refresh_cameras()
        cam_row.addWidget(self.cam_combo, 1)
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.clicked.connect(self.refresh_cameras)
        cam_row.addWidget(self.btn_refresh)
        right.addLayout(cam_row)

        self.config_panel = ConfigPanel(self)
        right.addWidget(self.config_panel)

        self.btn_toggle = QPushButton("开始识别")
        self.btn_toggle.setMinimumHeight(40)
        self.btn_toggle.clicked.connect(self.toggle)
        right.addWidget(self.btn_toggle)

        right.addStretch(1)
        root.addLayout(right)

    def refresh_cameras(self) -> None:
        indices = list_cameras()
        self.cam_combo.clear()
        if not indices:
            self.cam_combo.addItem("未检测到摄像头", -1)
        else:
            for i in indices:
                self.cam_combo.addItem(f"摄像头 {i}", i)

    def toggle(self) -> None:
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
            if self.plc is not None:
                self.plc.close()
                self.plc = None
            self.btn_toggle.setText("开始识别")
            self.video_label.setText("已停止")
            return
        idx = self.cam_combo.currentData()
        if idx is None or idx < 0:
            QMessageBox.warning(self, "提示", "未检测到可用摄像头")
            return
        self.plc = RejectController(**self.config_panel.plc_config())
        self.thread = VideoThread(idx, self.config_panel.config)
        self.thread.frame_ready.connect(self.on_frame)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.start()
        self.btn_toggle.setText("停止识别")

    def _on_thread_finished(self) -> None:
        self.btn_toggle.setText("开始识别")
        if self.plc is not None:
            self.plc.close()
            self.plc = None

    def on_frame(self, annotated, result) -> None:
        pixmap = cv_to_qpixmap(annotated)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        lines = []
        if result.filter_found:
            if result.has_flavor_line:
                status = "合格" if result.qualified else "剔除"
                lines.append(f"判定: {status}  香线数量={len(result.lines)}")
                lines.append(
                    f"偏移距离: {result.offset_ratio:.3f} (相对半径) / {result.offset_mm:.2f} mm"
                )
                for i, ln in enumerate(result.lines, 1):
                    lines.append(
                        f"  芯线{i}: 圆心=({ln.x},{ln.y}) 偏移比={ln.offset_ratio:.3f} "
                        f"偏移={ln.offset_mm:.2f}mm 面积={ln.area}"
                    )
            else:
                lines.append("判定: 无香线 -> " + ("剔除" if result.reject else "合格"))
            ell = result.filter_ellipse
            lines.append(
                f"滤棒椭圆: 圆心=({ell.cx:.1f},{ell.cy:.1f}) 半轴=({ell.rx:.1f},{ell.ry:.1f})"
            )
            if result.reject:
                self.reject_count += 1
                self.stat_label.setText(f"剔除计数: {self.reject_count}")
                if self.plc is not None:
                    self.plc.send_reject(result)
            now = time.time()
            if now - self._last_history >= 1.0:
                self.db.add("live", result)
                self._last_history = now
        else:
            lines.append("未找到滤棒，请调整画面或参数")
        self.result_box.setText("\n".join(lines))