
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
    app = QApplication(sys.argv) 

    with open("style.qss", "r") as style_file:
        style_str = style_file.read()
    app.setStyleSheet(style_str)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())