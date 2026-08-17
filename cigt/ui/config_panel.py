from typing import List

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from detector import DetectionConfig, FilterDetector


def cv_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(img)


def list_cameras(max_index: int = 8) -> List[int]:
    indices = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            indices.append(i)
        cap.release()
    return indices


class ConfigPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None):
        super().__init__("识别参数", parent)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        det_group = QGroupBox("香线检测")
        det_layout = QVBoxLayout(det_group)
        self.chk_color = QCheckBox("颜色检测 (高饱和度)")
        self.chk_color.setChecked(True)
        det_layout.addWidget(self.chk_color)
        self.chk_dark = QCheckBox("暗色检测 (低亮度)")
        self.chk_dark.setChecked(True)
        det_layout.addWidget(self.chk_dark)
        form = QFormLayout()
        self.spin_sat, sat_row = self._slider_spin(40, 255, 70, "饱和度阈值")
        form.addRow(sat_row)
        self.spin_dark, dark_row = self._slider_spin(1, 180, 110, "亮度阈值")
        form.addRow(dark_row)
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(1, 2000)
        self.spin_min_area.setValue(30)
        form.addRow("最小面积", self.spin_min_area)
        det_layout.addLayout(form)
        layout.addWidget(det_group)

        judge_group = QGroupBox("判定与剔除阈值")
        judge_layout = QFormLayout(judge_group)
        self.spin_thresh, thresh_row = self._slider_spin(5, 60, 25, "芯线偏移阈值 (%)")
        judge_layout.addRow(thresh_row)
        self.chk_missing = QCheckBox("缺香线即判定剔除")
        self.chk_missing.setChecked(True)
        judge_layout.addRow(self.chk_missing)
        self.spin_radius_mm = QSpinBox()
        self.spin_radius_mm.setRange(1, 50)
        self.spin_radius_mm.setValue(4)
        judge_layout.addRow("滤嘴半径 (mm)", self.spin_radius_mm)
        layout.addWidget(judge_group)

        plc_group = QGroupBox("PLC 剔除信号输出")
        plc_layout = QFormLayout(plc_group)
        self.chk_plc = QCheckBox("启用剔除信号输出")
        self.chk_plc.setChecked(False)
        plc_layout.addRow(self.chk_plc)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["simulate", "serial", "udp", "tcp"])
        plc_layout.addRow("输出模式", self.cmb_mode)
        self.edit_addr = QLineEdit("127.0.0.1")
        plc_layout.addRow("地址/IP", self.edit_addr)
        self.spin_port = QSpinBox()
        self.spin_port.setRange(1, 65535)
        self.spin_port.setValue(5000)
        plc_layout.addRow("端口", self.spin_port)
        self.edit_serial = QLineEdit("COM3")
        plc_layout.addRow("串口号", self.edit_serial)
        self.spin_baud = QSpinBox()
        self.spin_baud.setRange(1200, 921600)
        self.spin_baud.setValue(9600)
        plc_layout.addRow("波特率", self.spin_baud)
        self.edit_cmd = QLineEdit("REJECT")
        plc_layout.addRow("指令前缀", self.edit_cmd)
        layout.addWidget(plc_group)

        hint = QLabel("原理：滤波→边缘提取→椭圆拟合滤棒/芯线→圆心→偏移距离→阈值比较→剔除")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)

    def _slider_spin(self, lo: int, hi: int, value: int, label: str):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(value)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        wrap = QWidget()
        h = QHBoxLayout(wrap)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel(label))
        h.addWidget(slider, 1)
        h.addWidget(spin)
        return spin, wrap

    def config(self) -> DetectionConfig:
        return DetectionConfig(
            use_color=self.chk_color.isChecked(),
            use_dark=self.chk_dark.isChecked(),
            sat_threshold=self.spin_sat.value(),
            dark_value=self.spin_dark.value(),
            min_area=self.spin_min_area.value(),
            reject_threshold=self.spin_thresh.value() / 100.0,
            reject_missing_line=self.chk_missing.isChecked(),
            filter_radius_mm=float(self.spin_radius_mm.value()),
        )

    def plc_config(self) -> dict:
        return {
            "enabled": self.chk_plc.isChecked(),
            "mode": self.cmb_mode.currentText(),
            "address": self.edit_addr.text().strip(),
            "port": self.spin_port.value(),
            "serial_port": self.edit_serial.text().strip(),
            "baudrate": self.spin_baud.value(),
            "command_prefix": self.edit_cmd.text().strip() or "REJECT",
        }

    def apply_to(self, detector: FilterDetector) -> None:
        detector.set_config(self.config())