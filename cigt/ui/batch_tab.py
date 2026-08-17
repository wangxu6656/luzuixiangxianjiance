import csv
import os
from typing import List, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from detector import DetectionConfig, FilterDetector

from history.storage import HistoryDB

from .config_panel import ConfigPanel, cv_to_qpixmap

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class BatchWorker(QThread):
    progressed = pyqtSignal(int, int, str)
    finished_all = pyqtSignal(list)

    def __init__(self, files: List[str], out_dir: str, config_getter, db, parent: QWidget | None = None):
        super().__init__(parent)
        self.files = files
        self.out_dir = out_dir
        self.config_getter = config_getter
        self.db = db
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        detector = FilterDetector(self.config_getter())
        rows: List[dict] = []
        total = len(self.files)
        for i, path in enumerate(self.files):
            if self._cancel:
                break
            name = os.path.basename(path)
            self.progressed.emit(i + 1, total, name)
            try:
                detector.set_config(self.config_getter())
                img = cv_read(path)
                result = detector.detect(img)
                self.db.add("batch:" + name, result)
                if result.filter_found:
                    annotated = detector.annotate(img, result)
                    out_path = os.path.join(self.out_dir, "anno_" + name)
                    cv_write(out_path, annotated)
                rows.append(self._row(name, result))
            except Exception:
                rows.append(
                    {
                        "file": name,
                        "filter_found": False,
                        "has_flavor_line": False,
                        "count": 0,
                        "positions": "",
                        "offset_ratio": None,
                        "offset_mm": None,
                        "qualified": None,
                        "reject": True,
                        "note": "处理异常",
                    }
                )
        self.finished_all.emit(rows)

    def _row(self, name: str, result) -> dict:
        positions = ";".join(f"({ln.x},{ln.y})" for ln in result.lines)
        if result.filter_found:
            return {
                "file": name,
                "filter_found": True,
                "has_flavor_line": result.has_flavor_line,
                "count": len(result.lines),
                "positions": positions,
                "offset_ratio": result.offset_ratio,
                "offset_mm": result.offset_mm,
                "qualified": result.qualified,
                "reject": result.reject,
                "note": "",
            }
        return {
            "file": name,
            "filter_found": False,
            "has_flavor_line": False,
            "count": 0,
            "positions": "",
            "offset_ratio": None,
            "offset_mm": None,
            "qualified": None,
            "reject": True,
            "note": "未找到滤嘴",
        }


def cv_read(path: str):
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def cv_write(path: str, bgr) -> None:
    import cv2

    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, bgr)
    if ok:
        buf.tofile(path)


class BatchTab(QWidget):
    def __init__(self, db: HistoryDB, parent: QWidget | None = None):
        super().__init__(parent)
        self.db = db
        self.worker: Optional[BatchWorker] = None
        self._files: List[str] = []
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)

        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("请选择包含香烟截面图片的文件夹")
        self.btn_browse = QPushButton("选择文件夹")
        self.btn_browse.clicked.connect(self.choose_dir)
        self.btn_files = QPushButton("选择图片")
        self.btn_files.clicked.connect(self.choose_files)
        dir_row.addWidget(self.dir_edit, 1)
        dir_row.addWidget(self.btn_browse)
        dir_row.addWidget(self.btn_files)
        root.addLayout(dir_row)

        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("输出文件夹（默认为输入文件夹下 anno_out）")
        self.btn_out = QPushButton("选择输出文件夹")
        self.btn_out.clicked.connect(self.choose_out)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(self.btn_out)
        root.addLayout(out_row)

        self.config_panel = ConfigPanel(self)
        root.addWidget(self.config_panel)

        ctrl_row = QHBoxLayout()
        self.btn_run = QPushButton("开始批量识别")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self.run_batch)
        ctrl_row.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_batch)
        ctrl_row.addWidget(self.btn_cancel)
        self.progress = QProgressBar()
        ctrl_row.addWidget(self.progress, 1)
        root.addLayout(ctrl_row)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            ["文件名", "滤嘴", "香线", "数量", "偏移比", "偏移mm", "判定", "位置(像素)", "备注", "标注图"]
        )
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        root.addWidget(self.table, 1)

        self.btn_export = QPushButton("导出 CSV 结果")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_csv)
        root.addWidget(self.btn_export)

    def choose_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.dir_edit.setText(path)
            self._files = sorted(
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            )

    def choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)"
        )
        if files:
            self.dir_edit.setText(os.path.dirname(files[0]))
            self._files = sorted(files)

    def choose_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.out_edit.setText(path)

    def _resolve_out_dir(self) -> str:
        out = self.out_edit.text().strip()
        if not out:
            src = self.dir_edit.text().strip()
            out = os.path.join(src or os.getcwd(), "anno_out")
        os.makedirs(out, exist_ok=True)
        return out

    def run_batch(self) -> None:
        if not self._files:
            QMessageBox.warning(self, "提示", "请先选择文件夹或图片")
            return
        out_dir = self._resolve_out_dir()
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setRange(0, len(self._files))
        self.progress.setValue(0)
        self.table.setRowCount(0)
        self.worker = BatchWorker(self._files, out_dir, self.config_panel.config, self.db)
        self.worker.progressed.connect(self.on_progress)
        self.worker.finished_all.connect(self.on_finished)
        self.worker.start()

    def cancel_batch(self) -> None:
        if self.worker is not None:
            self.worker.cancel()

    def on_progress(self, done: int, total: int, name: str) -> None:
        self.progress.setValue(done)
        self.progress.setFormat(f"{done}/{total}  正在处理: {name}")

    def on_finished(self, rows: List[dict]) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._rows = rows
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [
                row["file"],
                "是" if row["filter_found"] else "否",
                "有" if row["has_flavor_line"] else "无",
                str(row["count"]),
                "" if row["offset_ratio"] is None else f"{row['offset_ratio']:.4f}",
                "" if row["offset_mm"] is None else f"{row['offset_mm']:.2f}",
                "剔除" if row["reject"] else ("合格" if row["qualified"] else "-"),
                row["positions"],
                row["note"],
                "anno_" + row["file"] if row["filter_found"] else "",
            ]
            for c, val in enumerate(values):
                item = QTableWidgetItem(val)
                if row["reject"]:
                    item.setForeground(Qt.red)
                self.table.setItem(r, c, item)
        self.btn_export.setEnabled(True)
        if rows:
            has = sum(1 for r in rows if r["has_flavor_line"])
            rej = sum(1 for r in rows if r["reject"])
            QMessageBox.information(
                self,
                "完成",
                f"共处理 {len(rows)} 张，检出香线 {has} 张，无香线 {len(rows) - has} 张，剔除 {rej} 张。",
            )

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存 CSV", "香线识别结果.csv", "CSV (*.csv)")
        if not path or not self._rows:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["文件名", "滤嘴", "有香线", "香线数量", "偏移比", "偏移mm", "判定", "位置(像素)", "备注"])
            for row in self._rows:
                writer.writerow(
                    [
                        row["file"],
                        row["filter_found"],
                        row["has_flavor_line"],
                        row["count"],
                        "" if row["offset_ratio"] is None else round(row["offset_ratio"], 4),
                        "" if row["offset_mm"] is None else round(row["offset_mm"], 2),
                        "剔除" if row["reject"] else "合格",
                        row["positions"],
                        row["note"],
                    ]
                )
        QMessageBox.information(self, "完成", f"已导出到 {path}")