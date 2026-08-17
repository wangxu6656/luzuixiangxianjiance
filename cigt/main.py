import sys
import traceback

from PyQt5.QtWidgets import QApplication, QMessageBox

from history.storage import HistoryDB


def _handle_exception(exc_type, exc_value, exc_tb) -> None:
    traceback.print_exception(exc_type, exc_value, exc_tb)
    with open("error.log", "a", encoding="utf-8") as f:
        f.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
    try:
        QMessageBox.critical(None, "错误", "发生异常，详情见 error.log\n\n" + str(exc_value))
    except Exception:
        pass


def main() -> None:
    sys.excepthook = _handle_exception
    app = QApplication(sys.argv)
    db = HistoryDB("history.db")
    from ui.main_window import MainWindow

    win = MainWindow(db)
    win.show()
    code = app.exec_()
    db.close()
    sys.exit(code)


if __name__ == "__main__":
    main()