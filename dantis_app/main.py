
from PyQt5.QtWidgets import QApplication

from view.main_window import MainWindow
import sys

"""
main.py
=======

Entry point of the application. Initializes the Qt application,
loads the UI stylesheet, and displays the main window.

This script is responsible for bootstrapping the user interface,
handling style setup, and launching the event loop for the application.

Attributes
----------
app : QApplication
    The Qt application instance.
window : MainWindow
    The main application window displayed on startup.
"""

if __name__ == "__main__": 
    from PyQt5.QtGui import QPalette, QColor

    def apply_light_palette(app: QApplication) -> None:
        pal = QPalette()
        # Window background and main text
        pal.setColor(QPalette.Window, QColor("#ffffff"))
        pal.setColor(QPalette.WindowText, QColor("#1f2937"))
        # Edit/list surfaces
        pal.setColor(QPalette.Base, QColor("#ffffff"))
        pal.setColor(QPalette.AlternateBase, QColor("#f7f8fa"))
        pal.setColor(QPalette.Text, QColor("#1f2937"))
        # Buttons
        pal.setColor(QPalette.Button, QColor("#ffffff"))
        pal.setColor(QPalette.ButtonText, QColor("#1f2937"))
        # Tooltips
        pal.setColor(QPalette.ToolTipBase, QColor("#ffffe0"))
        pal.setColor(QPalette.ToolTipText, QColor("#1f2937"))
        # Selections
        pal.setColor(QPalette.Highlight, QColor("#3b82f6"))
        pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        # BrightText for specific contrasts
        pal.setColor(QPalette.BrightText, QColor("#ffffff"))
        app.setPalette(pal)

    app = QApplication(sys.argv)

    if sys.platform == "darwin":
        apply_light_palette(app)

    with open("style.qss", "r") as style_file:
        style_str = style_file.read()
    app.setStyleSheet(style_str)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())