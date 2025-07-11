
from PyQt5.QtWidgets import QApplication
from view.main_window import MainWindow
import sys

if __name__ == "__main__": 
    # TODO Quitar esto cuando se solucione el problema con TensorFlow y CUDA
    #import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Añadido temporalmente para evitar problemas con TensorFlow y CUDA
    
    app = QApplication(sys.argv) 

    ## Loading style file
    with open("style.qss", "r") as style_file:
        style_str = style_file.read()
    app.setStyleSheet(style_str)

    ## ANOTHER WAYT TO READ THE STYLE SHEET
    #style_file = QFile("style.qss")
    #style_file.open(QFile.ReadOnly | QFile.Text)
    #style_stream = QTextStream(style_file)
    #app.setStyleSheet(style_stream.readAll())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())