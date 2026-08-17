from PyQt5.QtWidgets import QMainWindow, QTabWidget

from history.storage import HistoryDB

from .batch_tab import BatchTab
from .history_tab import HistoryTab
from .live_tab import LiveTab


class MainWindow(QMainWindow):
    def __init__(self, db: HistoryDB):
        super().__init__()
        self.db = db
        self.setWindowTitle("香烟滤嘴香线识别系统")
        self.resize(1180, 800)
        tabs = QTabWidget()
        tabs.addTab(LiveTab(self.db, self), "实时识别")
        tabs.addTab(BatchTab(self.db, self), "图片批量识别")
        tabs.addTab(HistoryTab(self.db, self), "历史数据")
        self.setCentralWidget(tabs)