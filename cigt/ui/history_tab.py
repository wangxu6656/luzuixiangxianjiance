from typing import List

from PyQt5.QtCore import QDate, Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from history.storage import HistoryDB

HEADERS = ["ID", "时间", "来源", "滤嘴", "有香线", "数量", "偏移比", "偏移mm", "合格", "剔除", "位置"]


def fmt_bool(v):
    return {None: "-", 1: "是", 0: "否"}.get(v, str(v))


class HistoryTab(QWidget):
    def __init__(self, db: HistoryDB, parent: QWidget | None = None):
        super().__init__(parent)
        self.db = db
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("从"))
        self.date_from = QDateEdit(QDate.currentDate().addDays(-7))
        self.date_from.setCalendarPopup(True)
        row.addWidget(self.date_from)
        row.addWidget(QLabel("至"))
        self.date_to = QDateEdit(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        row.addWidget(self.date_to)
        row.addWidget(QLabel("结果"))
        self.cmb_result = QComboBox()
        self.cmb_result.addItems(["全部", "合格", "剔除"])
        row.addWidget(self.cmb_result)
        self.btn_query = QPushButton("查询")
        self.btn_query.clicked.connect(self.query)
        row.addWidget(self.btn_query)
        self.btn_export = QPushButton("导出 CSV")
        self.btn_export.clicked.connect(self.export_csv)
        row.addWidget(self.btn_export)
        self.lbl_summary = QLabel("")
        row.addWidget(self.lbl_summary, 1)
        root.addLayout(row)

        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        root.addWidget(self.table, 1)

        self.query()

    def query(self) -> None:
        start = self.date_from.date().toString("yyyy-MM-dd")
        end = self.date_to.date().toString("yyyy-MM-dd") + " 23:59:59"
        rf = self.cmb_result.currentText()
        result_filter = {"全部": "all", "合格": "ok", "剔除": "reject"}[rf]
        rows = self.db.query(start=start, end=end, result_filter=result_filter)
        self._rows = rows
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [
                str(row[0]),
                str(row[1]),
                str(row[2]),
                fmt_bool(row[3]),
                fmt_bool(row[4]),
                str(row[5]),
                "" if row[6] is None else f"{row[6]:.4f}",
                "" if row[7] is None else f"{row[7]:.3f}",
                fmt_bool(row[8]),
                fmt_bool(row[9]),
                str(row[10]),
            ]
            for c, val in enumerate(values):
                item = QTableWidgetItem(val)
                if row[9]:
                    item.setForeground(Qt.red)
                self.table.setItem(r, c, item)
        rejects = sum(1 for r in rows if r[9])
        self.lbl_summary.setText(f"共 {len(rows)} 条，剔除 {rejects} 条")

    def export_csv(self) -> None:
        rows = getattr(self, "_rows", [])
        if not rows:
            QMessageBox.information(self, "提示", "当前无数据可导出")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存 CSV", "检测历史.csv", "CSV (*.csv)")
        if not path:
            return
        self.db.export_csv(rows, path)
        QMessageBox.information(self, "完成", f"已导出 {len(rows)} 条到 {path}")