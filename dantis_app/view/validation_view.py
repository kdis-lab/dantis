from PyQt5.QtWidgets import QWidget, QSpinBox
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDrag
from PyQt5.QtWidgets import QListWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHeaderView, QButtonGroup
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QPushButton

from core.utils import show_error
from controller.validation_controller import ValidationController

class ViewValidation(QWidget):
    def __init__(self, main_window, ui, DatasetController=None):
        super().__init__()
        self.main_window = main_window
        self.ui = ui
        self.validationController = ValidationController()
        self.datasetController = DatasetController
        self.validation_config = {}
        self.init_ui()

    def init_ui(self):
        self.group_validacion = QButtonGroup(self)
        self.group_validacion.addButton(self.ui.radioButton_train_test)
        self.group_validacion.addButton(self.ui.radioButton_crossVal)
        self.group_validacion.addButton(self.ui.radioButton_sliding)

        self.group_si_no = QButtonGroup(self)
        self.group_si_no.addButton(self.ui.radioButton_no)
        self.group_si_no.addButton(self.ui.radioButton_yes)

        self.ui.radioButton_train_test.setChecked(True)
        self.ui.radioButton_no.setChecked(True)
        self.ui.next_val.clicked.connect(self.on_next_val_click)

        self.ui.radioButton_train_test.clicked.connect(self.on_radioButton_train_test_click)
        self.ui.radioButton_crossVal.clicked.connect(self.on_radioButton_crossVal_click)
        self.ui.radioButton_sliding.clicked.connect(self.on_radioButton_sliding_click)
        self.ui.radioButton_no.clicked.connect(self.on_radioButton_no_click)
        self.ui.radioButton_yes.clicked.connect(self.on_radioButton_yes_click)
        self.ui.pushButton_insertRow.clicked.connect(self.on_insert_row_click)

        # Crear los widgets personalizados
        drop_list_widget = DropListWidget()
        drop_table_widget = DropTableWidget(drop_list_widget)

        # Obtener el layout donde están los widgets originales
        layout = self.ui.horizontalLayout_listTabla

        # Reemplazar widgets en el layout
        layout.replaceWidget(self.ui.listWidget_datasets, drop_list_widget)
        layout.replaceWidget(self.ui.tableWidget_TVT, drop_table_widget)

        # Copiar columnas y encabezados del widget original a drop_table_widget
        self.old_table = self.ui.tableWidget_TVT
        drop_table_widget.setColumnCount(self.old_table.columnCount())
        drop_table_widget.setHorizontalHeaderLabels(
            [self.old_table.horizontalHeaderItem(i).text() for i in range(self.old_table.columnCount())]
        )

        # Ajustar columnas para que tengan ancho uniforme
        header = drop_table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        # SizePolicy para listWidget: mínimo horizontal, expand vertical
        size_policy_list = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        drop_list_widget.setSizePolicy(size_policy_list)

        # SizePolicy para tableWidget: expand horizontal y vertical
        size_policy_table = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        drop_table_widget.setSizePolicy(size_policy_table)

        # Ocultar widgets antiguos
        self.ui.listWidget_datasets.hide()
        self.ui.tableWidget_TVT.hide()

        # Asignar los widgets nuevos para que el resto del código funcione igual
        self.ui.listWidget_datasets = drop_list_widget
        self.ui.tableWidget_TVT = drop_table_widget

        # Configuraciones necesarias
        self.ui.listWidget_datasets.setDragEnabled(True)
        self.ui.tableWidget_TVT.setDragEnabled(True)

        self.ui.listWidget_datasets.setDisabled(True)
        self.ui.tableWidget_TVT.setDisabled(True)

        for i in range(self.ui.gridLayout_crossVal.count()):
            widget_crossval = self.ui.gridLayout_crossVal.itemAt(i).widget()
            widget_sliding = self.ui.gridLayout_sliding.itemAt(i).widget()
            if widget_crossval:
                widget_crossval.setDisabled(True)
            if widget_sliding:
                widget_sliding.setDisabled(True)

    def on_delete_row_click(self):
        button = self.sender()
        if button:
            for row in range(self.ui.tableWidget_TVT.rowCount()):
                widget = self.ui.tableWidget_TVT.cellWidget(row, self.ui.tableWidget_TVT.columnCount() - 1)
                if widget == button:
                    # Devuelve datos a la lista
                    for col in range(self.ui.tableWidget_TVT.columnCount() - 1):  # evitar la columna del botón
                        item = self.ui.tableWidget_TVT.item(row, col)
                        if item:
                            self.ui.listWidget_datasets.addItem(item.text())
                    self.ui.tableWidget_TVT.removeRow(row)
                    break


    def initial_parameter_configuration(self):
        datasets = self.datasetController.get_data()
        names = [dataset.name for dataset in datasets.values()]
        self.ui.listWidget_datasets.clear()
        self.ui.listWidget_datasets.addItems(names)

    def on_insert_row_click(self):
        rowPosition = self.ui.tableWidget_TVT.rowCount()
        self.ui.tableWidget_TVT.insertRow(rowPosition)

        # Crear botón de borrar
        btn_borrar = QPushButton("🗑️")
        btn_borrar.clicked.connect(self.on_delete_row_click)

        self.ui.tableWidget_TVT.setCellWidget(rowPosition, self.ui.tableWidget_TVT.columnCount() - 1, btn_borrar)


    def on_radioButton_train_test_click(self):
        # ACTIVAMOS ELEMENTOS PARA TRAIN/TEST
        self.ui.spinBox_train.setDisabled(False)
        self.ui.spinBox_val.setDisabled(False)
        self.ui.spinBox_test.setDisabled(False)

        # DESACTIVAMOS ELEMENTOS PARA CROSS VALIDATION
        self.ui.spinBox_crossVal.setValue(self.ui.spinBox_crossVal.minimum())
        self.ui.spinBox_crossVal.setDisabled(True)

        self.ui.spinBox_crossval_percent.setValue(self.ui.spinBox_crossval_percent.minimum())
        self.ui.spinBox_crossval_percent.setDisabled(True)

        # DESACTIVAMOS ELEMENTOS PARA SLIDING SPLIT
        self.ui.spinBox_sliding.setValue(self.ui.spinBox_sliding.minimum())
        self.ui.spinBox_sliding.setDisabled(True)

        self.ui.spinBox_sliding_percent.setValue(self.ui.spinBox_sliding_percent.minimum())
        self.ui.spinBox_sliding_percent.setDisabled(True)

    def on_radioButton_crossVal_click(self):
        # DESACTIVAMOS ELEMENTOS PARA TRAIN/TEST
        self.ui.spinBox_train.setValue(self.ui.spinBox_train.minimum())
        self.ui.spinBox_train.setDisabled(True)

        self.ui.spinBox_val.setValue(self.ui.spinBox_val.minimum())
        self.ui.spinBox_val.setDisabled(True)

        self.ui.spinBox_test.setValue(self.ui.spinBox_test.minimum())
        self.ui.spinBox_test.setDisabled(True)

        # ACTIVAMOS ELEMENTOS PARA CROSS VALIDATION
        self.ui.spinBox_crossVal.setDisabled(False)
        self.ui.spinBox_crossval_percent.setDisabled(False)

        # DESACTIVAMOS ELEMENTOS PARA SLIDING SPLIT
        self.ui.spinBox_sliding.setValue(self.ui.spinBox_sliding.minimum())
        self.ui.spinBox_sliding.setDisabled(True)

        self.ui.spinBox_sliding_percent.setValue(self.ui.spinBox_sliding_percent.minimum())
        self.ui.spinBox_sliding_percent.setDisabled(True)

    def on_radioButton_sliding_click(self):
        # DESACTIVAMOS ELEMENTOS PARA TRAIN/TEST
        self.ui.spinBox_train.setValue(self.ui.spinBox_train.minimum())
        self.ui.spinBox_train.setDisabled(True)

        self.ui.spinBox_val.setValue(self.ui.spinBox_val.minimum())
        self.ui.spinBox_val.setDisabled(True)

        self.ui.spinBox_test.setValue(self.ui.spinBox_test.minimum())
        self.ui.spinBox_test.setDisabled(True)

        # DESACTIVAMOS ELEMENTOS PARA CROSS VALIDATION
        self.ui.spinBox_crossVal.setValue(self.ui.spinBox_crossVal.minimum())
        self.ui.spinBox_crossVal.setDisabled(True)

        self.ui.spinBox_crossval_percent.setValue(self.ui.spinBox_crossval_percent.minimum())
        self.ui.spinBox_crossval_percent.setDisabled(True)

        # ACTIVAMOS ELEMENTOS PARA SLIDING SPLIT
        self.ui.spinBox_sliding.setDisabled(False)
        self.ui.spinBox_sliding_percent.setDisabled(False)

    def on_next_val_click(self): 
        self.check_info_val()
    
    def check_info_val(self):
        if self.validar_datos(): 
            self.ui.model_btn_2.setDisabled(False)
            self.ui.model_btn_1.setDisabled(False)

            # Para cambiar a la página de los modelos
            self.ui.model_btn_2.setChecked(True)

    def validar_datos(self): 
        if self.ui.radioButton_yes.isChecked():
            rows = self.ui.tableWidget_TVT.rowCount()
            columns = self.ui.tableWidget_TVT.columnCount()

            # Obtener los índices de columna por nombre
            column_names = {}
            for col in range(columns):
                header_item = self.ui.tableWidget_TVT.horizontalHeaderItem(col)
                if header_item is not None:
                    column_names[header_item.text().strip().lower()] = col

            
            # Asegurarnos de que las columnas necesarias existen
            if "train" not in column_names or "test" not in column_names:
                print("Error: Las columnas 'train' y/o 'test' no existen en la tabla.")
            else:
                train_col = column_names["train"]
                val_col = column_names["validation"]
                test_col = column_names["test"]

                info_columns = {}

                for row in range(rows):
                    train_item = self.ui.tableWidget_TVT.item(row, train_col)
                    val_item = self.ui.tableWidget_TVT.item(row, val_col)
                    test_item = self.ui.tableWidget_TVT.item(row, test_col)

                    train_text = train_item.text().strip() if train_item else ""
                    val_text = val_item.text().strip() if val_item else ""
                    test_text = test_item.text().strip() if test_item else ""

                    if (train_text and not test_text) or (test_text and not train_text):
                        show_error(f"Error en la fila {row + 1}. Debes introducir contenido tanto en 'train' como en 'test'. La columna de 'validation' es opcional.")
                        return False
                
                    if train_text and test_text: 
                        info_columns[row] = {
                            "train": train_text, 
                            "val": val_text,
                            "test": test_text }
                        
                # VALIDACIÓN DE CONSISTENCIA DE DATASETS
                for row, datasets_dict in info_columns.items():
                    nombres_archivos = [datasets_dict['train'], datasets_dict['test']]
                    if datasets_dict['val']:
                        nombres_archivos.append(datasets_dict['val'])
                    datasets = self.datasetController.get_data()
                    datasets_seleccionados = [d for d in datasets.values() if d.name in nombres_archivos]

                    if len(datasets_seleccionados) != len(nombres_archivos):
                        show_error(f"Error en la fila {row + 1}. Uno o más datasets especificados no están disponibles.")
                        return False

                    # Usar el primero como referencia
                    ref_x_col = datasets_seleccionados[0].x_col
                    ref_y_col = datasets_seleccionados[0].y_col

                    for dataset in datasets_seleccionados[1:]:
                        if dataset.x_col != ref_x_col or dataset.y_col != ref_y_col:
                            show_error(f"Error en la fila {row + 1}. Los datasets deben tener las mismas columnas 'x' y la misma columna 'y'.")
                            return False
                
                self.datasetController.set_split_data(info_columns)

                    
        if self.ui.radioButton_train_test.isChecked():
            if (self.ui.spinBox_train.value() == 0) | (self.ui.spinBox_test.value() == 0): 
                show_error("En la validación Train/Test debe rellenar los campos de 'Entrenamiento' y 'Testeo'. El campo opcional es 'Validación'.")
                return False
            else: 
                if self.ui.spinBox_train.value() + self.ui.spinBox_test.value() + self.ui.spinBox_val.value() != 100.0:
                    show_error("Los porcentajes deben sumar 100.0")
                    return False
                else: 
                    self.validation_config = {
                        "type": "train/test", 
                        "train": self.ui.spinBox_train.value(), 
                        "validation": self.ui.spinBox_val.value(), 
                        "test": self.ui.spinBox_test.value()
                    }
                    self.validationController.set_validation_config(self.validation_config)
                    return True

        elif self.ui.radioButton_crossVal.isChecked():
            if (self.ui.spinBox_crossVal.value() == 0) or (self.ui.spinBox_crossval_percent.value() == 0):
                show_error("Debe rellenar el campo para crossValidation y el porcentaje de test.")
                return False
            else: 
                self.validation_config = {
                        "type": "crossVal", 
                        "crossVal": self.ui.spinBox_crossVal.value(),
                        "percentage_crossVal": self.ui.spinBox_crossval_percent.value()
                }
                self.validationController.set_validation_config(self.validation_config)
                return True
            
        elif self.ui.radioButton_sliding.isChecked():
            if (self.ui.spinBox_sliding.value() == 0) or (self.ui.spinBox_sliding_percent.value() == 0):
                show_error("Debe rellenar el campo para Sliding Split y el porcentaje de test.")
                return False
            else: 
                self.validation_config = {
                        "type": "Sliding_split", 
                        "sliding": self.ui.spinBox_sliding.value(), 
                        "percentage_sliding": self.ui.spinBox_sliding_percent.value()
                }
                self.validationController.set_validation_config(self.validation_config)
                return True

    def on_radioButton_no_click(self):
        self.ui.listWidget_datasets.setDisabled(True)
        self.ui.tableWidget_TVT.setDisabled(True)

    def on_radioButton_yes_click(self):
        self.ui.listWidget_datasets.setDisabled(False)
        self.ui.tableWidget_TVT.setDisabled(False)

        

class DropListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QListWidget.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)

    def startDrag(self, supportedActions):
        self._drag_item = self.currentItem()
        if self._drag_item:
            drag = QDrag(self)
            mimeData = QMimeData()
            mimeData.setText(self._drag_item.text())
            drag.setMimeData(mimeData)

            result = drag.exec_(Qt.MoveAction)
            if result == Qt.MoveAction and self._drag_item: 
                # Eliminar solo si el drop fue exitoso y en otro widget
                # Si quieres evitar eliminar cuando cae en sí mismo, puedes verificar destino
                self.takeItem(self.row(self._drag_item))
                self._drag_item = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        if event.source() == self:
            # Evitar drops sobre sí mismo
            event.ignore()
            return

        if event.mimeData().hasText():
            text = event.mimeData().text()
            if not any(self.item(i).text() == text for i in range(self.count())):
                self.addItem(text)
            event.acceptProposedAction()

class DropTableWidget(QTableWidget):
    def __init__(self, list_widget: QListWidget, parent=None):
        super().__init__(parent)
        self.list_widget = list_widget
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QTableWidget.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item:
            mimeData = QMimeData()
            mimeData.setText(item.text())

            drag = QDrag(self)
            drag.setMimeData(mimeData)

            if drag.exec_(Qt.MoveAction) == Qt.MoveAction:
                self.setItem(self.currentRow(), self.currentColumn(), None)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        if event.mimeData().hasText():
            text = event.mimeData().text()
            pos = event.pos()
            row = self.rowAt(pos.y())
            col = self.columnAt(pos.x())

            if row == -1 or col == -1:
                return

            existing_item = self.item(row, col)
            if existing_item:
                # Devuelve el anterior valor a la lista
                self.list_widget.addItem(existing_item.text())

            self.setItem(row, col, QTableWidgetItem(text))
            event.acceptProposedAction()