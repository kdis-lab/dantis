from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import QThread, QUrl, Qt
from PyQt5.QtGui import QTextTableFormat, QTextCursor, QImage, QTextImageFormat
from PyQt5.QtGui import QImage, QTextImageFormat, QTextCursor, QTextTableFormat, QTextDocument

import pandas as pd
import random, os
import matplotlib.pyplot as plt
from io import BytesIO
import logging

from controller.statds_statistical_controller import StatisticalTestController
from core.utils import show_error, verify_info
from controller.training_orchestrator import TrainingOrchestrator
from controller.download_controller import download_data_click, download_data_test_click
from controller.results_formatter import format_training_info
from controller.training_worker import TrainingWorker

class ViewResults(QWidget):
    def __init__(self, main_window, ui, datasetController,
                modelsController, metricsController, validationController):
        """
        GUI component for training models, displaying results, and performing statistical tests.
        This class allows the user to initiate training, view results in a text area, and
        perform statistical comparisons between models using post-hoc tests and visual outputs.

        Parameters
        ----------
        main_window : QMainWindow
            Reference to the main application window.
        ui : object
            A reference to the main UI object with all widgets.
        datasetController : DatasetController
            Controller instance managing dataset logic.
        modelsController : ModelsController
            Controller instance managing model logic.
        metricsController : MetricsController
            Controller instance managing evaluation metrics.
        validationController : ValidationController
            Controller instance managing validation methods.

        Attributes
        ----------
        ui : object
            Reference to the main UI interface.
        main_window : QMainWindow
            Reference to the main application window.
        training_orchestrator : TrainingOrchestrator
            Handles training logic in a background thread.
        statistic : StatisticalController
            Performs statistical analysis and post-hoc comparisons.
        datasetController : DatasetController
            Dataset management logic.
        modelsController : ModelsController
            Model management logic.
        metricsController : MetricsController
            Metrics configuration and validation.
        validationController : ValidationController
            Cross-validation and splitting strategy management.
        save_results : dict
            Stores the results of model training for further processing or exporting.
        statistical_results : dict
            Stores the results of statistical analysis for reporting or exporting.

        Methods
        -------
        init_ui()
            Sets up UI element bindings and connects signals to slots.
        set_save_results(results)
            Assigns training results to internal storage.
        modify_root()
            Opens a file dialog to allow user to change the export/save directory.
        enable_buttons(enable)
            Enables or disables statistical test related buttons.
        check_columns_and_rows()
            Validates if all rows in the results table are populated for both train and test values.
        on_generate_results_click()
            Initiates the training process and displays a placeholder message in the UI.
        append_training_message()
            Inserts a 'Training...' placeholder message into the QTextEdit.
        append_training_result(new_html)
            Replaces the placeholder with formatted HTML containing training results.
        on_generate_statistical_tests_click()
            Executes statistical comparisons between models and displays the results visually.
        validate_data_statistical(num_datasets, num_models)
            Checks if there are enough datasets and models to perform statistical tests.
        """
        super().__init__()
        self.ui = ui
        self.main_window = main_window
        self.training_orchestrator = TrainingOrchestrator(datasetController, 
                                                          modelsController, 
                                                          metricsController, 
                                                          validationController)
        self.datasetController = datasetController
        self.modelsController = modelsController
        self.metricsController = metricsController
        self.validationController = validationController
        self.statistic = StatisticalTestController()
        self.save_results = {}
        self.statistical_results = {}
        self.init_ui()

    def init_ui(self):
        """
        Connects UI buttons and actions to their corresponding slot functions.
        Handles button click bindings for training and statistics.
        """
        self.ui.generate_results.clicked.connect(self.on_generate_results_click)
        self.ui.generate_tests.clicked.connect(self.on_generate_statistical_tests_click)
        self.ui.download_data.clicked.connect(lambda: download_data_click(self.save_results))
        self.ui.download_data_test.clicked.connect(lambda: download_data_test_click(self.statistical_results))
        self.ui.modify_root.clicked.connect(self.modify_root)

        self.ui.test_comboBox.addItems(self.statistic.list_available_tests())
        self.ui.postHoc_comboBox.addItems(self.statistic.list_available_post_hoc())
        self.ui.alpha_comboBox.addItems(['0.01', '0.05', '0.1'])
        self.ui.alpha_comboBox.setCurrentText('0.05')  # Default to 0.05

        self.enable_buttons(True)

        ruta_por_defecto = os.path.expanduser("~")
        self.ui.lineEdit_root.setText(ruta_por_defecto)
        self.modelsController.set_path(ruta_por_defecto)

    def set_save_results (self, results): 
        """
        Stores training results in internal memory for later use.

        Parameters
        ----------
        results : dict
            Dictionary containing model performance metrics.
        """
        self.save_results = results

    def modify_root(self): 
        """
        Opens a file dialog for selecting a new save/export path.
        Updates internal path reference for saving results.
        """
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecciona una carpeta",
            ""
        )
        if path:
            self.ui.lineEdit_root.setText(path)
            self.modelsController.set_path(path)

    def enable_buttons(self, enable: bool):
        """
        Enables or disables buttons related to statistical testing.

        Parameters
        ----------
        enable : bool
            If True, enables the buttons; otherwise disables them.
        """
        self.ui.test_comboBox.setEnabled(enable)
        self.ui.postHoc_comboBox.setEnabled(enable)
        self.ui.generate_tests.setEnabled(enable)
        self.ui.alpha_comboBox.setEnabled(enable)

    def check_columns_and_rows(self):
        """
        Validates that all table cells in the results view contain valid train/test entries.

        Returns
        -------
        bool
            True if all cells are properly filled; False otherwise.
        """
        train_col = None
        test_col = None
        table = self.ui.tableWidget_TVT

        for col in range(table.columnCount()):
            header = table.horizontalHeaderItem(col)
            if header:
                name = header.text().strip().lower()
                if name == "train":
                    train_col = col
                elif name == "test":
                    test_col = col

        for row in range(table.rowCount()):
            train_item = table.item(row, train_col)
            test_item = table.item(row, test_col)

            train_text = train_item.text().strip() if train_item else ""
            test_text = test_item.text().strip() if test_item else ""

            if (train_text and not test_text) or (test_text and not train_text):
                return False

        return True

    def on_generate_results_click(self):
        """
        Handles the action of generating model results.
        Launches the training process and appends output to the result display area.
        """
        self.enable_buttons(False)

        if self.ui.radioButton_yes.isChecked():
           if not self.check_columns_and_rows(): 
               show_error("Comprueba la tabla de validación debido a que se ha eliminado uno de los datasets. No se puede ejecutar el entrenamiento.")
               return 

        threshold = self.ui.threshold.value()
        if threshold == 0:
            show_error("Debe de introducir un valor mayor a 0 en el campo threshold para poder generar resultados para los modelos.")
            return
        
        (datasets, x_cols, y_cols,
         models, metrics, val_opts) = self.training_orchestrator.collect_training_info()

        if not verify_info(datasets, x_cols, y_cols, models, metrics, val_opts):
            return

        info_text = format_training_info(
            datasets, x_cols, y_cols,
            models, metrics, val_opts, threshold
        )
        self.ui.text_generate_results.setHtml(info_text)

        self.thread = QThread()
        self.worker = TrainingWorker(
            datasets, x_cols, y_cols,
            models, metrics, val_opts, threshold, self.modelsController, self)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.notify_training.connect(self.append_training_message)
        self.worker.generate_result.connect(self.append_training_result)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(lambda: self.enable_buttons(True))
        self.thread.start()

    def append_training_message(self):
        """
        Inserts a temporary 'Training...' message into the QTextEdit while training is in progress.
        """
        cursor = self.ui.text_generate_results.textCursor()
        cursor.movePosition(QTextCursor.End) 
        self.ui.text_generate_results.setTextCursor(cursor)
        html_marker = '<br/><p id="entrenando-marker" style="color: gray; font-weight: bold;">Entrenando...</p>'
        cursor.insertHtml(html_marker)
        cursor = self.ui.text_generate_results.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.ui.text_generate_results.setTextCursor(cursor)

    def append_training_result(self, new_html: str):
        """
        Replaces the temporary training message with formatted results.

        Parameters
        ----------
        new_html : str
            HTML string containing the formatted training output.
        """
        full_html = self.ui.text_generate_results.toHtml()

        marker_start = full_html.find('<a name="entrenando-marker"></a>')
        if marker_start != -1:
            marker_end = full_html.find('</p>', marker_start)
            if marker_end != -1:
                marker_end += len('</p>')
                updated_html = (
                    full_html[:marker_start] +
                    new_html +
                    full_html[marker_end:]
                )
                self.ui.text_generate_results.setHtml(updated_html)
        else:
            self.ui.text_generate_results.setHtml(full_html + new_html)

        cursor = self.ui.text_generate_results.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.ui.text_generate_results.setTextCursor(cursor)
                
    def on_generate_statistical_tests_click(self):
        """
        Executes statistical analysis over the collected model results.
        Displays tables and visual results in the QTextEdit widget.
        """
        logging.debug("Generando resultados estadísticos...")
        results = self.worker.get_results()
        num_datasets = len(self.datasetController.get_all_datasets())
        num_models = len(self.modelsController.get_models())

        if  self.validate_data_statistical(num_datasets, num_models):
            show_error("Debe introducir al menos 8 datasets y 2 algoritmos para poder ejecutar los test estadísticos.") 
            return

        test_selected = self.ui.test_comboBox.currentText()
        post_hoc_selected = self.ui.postHoc_comboBox.currentText()
        alpha_selected = self.ui.alpha_comboBox.currentText()
        alpha_value = float(alpha_selected)
        text_edit = self.ui.test_generate_tests
        text_edit.clear()
        document = text_edit.document()

        for metric_selected, df in results.items():
            datasets = df.index.tolist()
            models = df.columns.tolist()

            filas = len(datasets) + 1
            columnas = len(models) + 1

            result_entry = {
                "metric": metric_selected,
                "data_table": df.to_dict(),
                "test_selected": test_selected,
                "alpha": alpha_value,
                "post_hoc_selected": post_hoc_selected,
                "test_results": {}, 
            }

            cursor = QTextCursor(document)
            cursor.movePosition(QTextCursor.End)
            cursor.insertBlock()
            cursor.insertText(f"📊 Resultados para la métrica '{metric_selected}'\n")
            cursor.insertBlock()

            table_format = QTextTableFormat()
            table_format.setBorder(1)
            table_format.setCellPadding(4)
            table_format.setCellSpacing(2)

            table = cursor.insertTable(filas, columnas, table_format)

            for col in range(1, columnas):
                model_name = str(models[col - 1])
                celda = table.cellAt(0, col)
                celda_cursor = celda.firstCursorPosition()
                celda_cursor.insertText(model_name)

            for fila in range(1, filas):
                dataset_name = str(datasets[fila - 1])
                table.cellAt(fila, 0).firstCursorPosition().insertText(dataset_name)

                for col in range(1, columnas):
                    valor = df.iloc[fila - 1, col - 1]
                    texto = f"{valor:.2f}" if pd.notnull(valor) else "N/A"
                    table.cellAt(fila, col).firstCursorPosition().insertText(texto)

            cursor.movePosition(QTextCursor.End)
            cursor.insertBlock()
            cursor.insertBlock()

            a = self.statistic.run_test(test_selected, df, alpha=alpha_value, post_hoc_selected=post_hoc_selected)

            if isinstance(a, list):
                result_entry["test_results"] = {}

                data_to_table = a[0]
                result_entry["test_results"] = data_to_table.to_dict()

                cursor.insertBlock()
                cursor.insertText(f"📊 Resultados Test estadístico '{test_selected}'\n")
                cursor.insertBlock()
                
                rows, cols = data_to_table.shape
                table = cursor.insertTable(rows + 1, cols, table_format)

                for col in range(cols):
                    table.cellAt(0, col).firstCursorPosition().insertText(str(data_to_table.columns[col]))

                for fila in range(rows):
                    for col in range(cols):
                        value = data_to_table.iat[fila, col]
                        text = f"{value:.4f}" if isinstance(value, float) else str(value)
                        table.cellAt(fila + 1, col).firstCursorPosition().insertText(text)

                cursor = QTextCursor(document)
                cursor.movePosition(QTextCursor.End)

                cursor.insertBlock()
                cursor.insertBlock()
                cursor.insertText(f"📊 Resultados Test estadístico '{post_hoc_selected}'\n")
                cursor.insertBlock()

                table_format = QTextTableFormat()
                table_format.setBorder(1)
                table_format.setCellPadding(4)
                table_format.setCellSpacing(2)
            
                if post_hoc_selected != "Nemenyi":
                    data_to_table = a[1]
                    result_entry["post_hoc_results"] = data_to_table.to_dict()

                    rows, cols = data_to_table.shape
                    table = cursor.insertTable(rows + 1, cols, table_format)

                    for col in range(cols):
                        table.cellAt(0, col).firstCursorPosition().insertText(str(data_to_table.columns[col]))

                    for fila in range(rows):
                        for col in range(cols):
                            value = data_to_table.iat[fila, col]
                            text = f"{value:.4f}" if isinstance(value, float) else str(value)
                            table.cellAt(fila + 1, col).firstCursorPosition().insertText(text)
                    
                    cursor = QTextCursor(document)
                    cursor.movePosition(QTextCursor.End)

                cursor.insertBlock()
                cursor.insertBlock()
                cursor.insertBlock()

                figure = a[-1]
                result_entry["_figures"] = [figure]
                buffer = BytesIO()
                figure.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)

                image = QImage()
                image.loadFromData(buffer.read())
                buffer.close()

                if not image.isNull():
                    if post_hoc_selected == "Nemenyi":
                        image = image.scaledToWidth(750, Qt.SmoothTransformation)

                    image_format = QTextImageFormat()
                    img_name = f"stat_fig_{random.randint(1000,9999)}.png"
                    document.addResource(QTextDocument.ImageResource, QUrl(img_name), image)
                    image_format.setName(img_name)
                    cursor.insertImage(image_format)
                    cursor.insertBlock()
                plt.close(figure)
                plt.clf()
                
            elif isinstance(a, pd.DataFrame):
                result_entry["test_results"] = a.to_dict()

                cursor.insertBlock()
                cursor.insertText(f"📊 Resultados Test estadístico '{test_selected}'\n")
                cursor.insertBlock()

                rows, cols = a.shape
                table = cursor.insertTable(rows + 1, cols, table_format)

                for col in range(cols):
                    table.cellAt(0, col).firstCursorPosition().insertText(str(a.columns[col]))

                for fila in range(rows):
                    for col in range(cols):
                        value = a.iat[fila, col]
                        text = f"{value:.4f}" if isinstance(value, float) else str(value)
                        table.cellAt(fila + 1, col).firstCursorPosition().insertText(text)

                cursor.insertBlock()
            else:
                result_entry["test_results"] = str(a)
                cursor.insertBlock()
                cursor.insertText(f"📊 Resultados Test estadístico '{test_selected}'\n")
                cursor.insertText(str(a))
                cursor.insertBlock()

            self.statistical_results[metric_selected] = result_entry
            cursor.insertBlock()
            cursor.insertBlock()

    def validate_data_statistical(self, num_datasets, num_models):
        """
        Validates whether enough datasets and models are available to run statistical tests.

        Parameters
        ----------
        num_datasets : int
            Number of datasets used for training.
        num_models : int
            Number of models evaluated.

        Returns
        -------
        bool
            True if insufficient data is available; False if validation passes.
        """
        if num_datasets < 8 or num_models < 2:
            return True
        else:
            return False