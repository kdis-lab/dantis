from PyQt5.QtWidgets import QWidget, QFileDialog, QSplitter, QVBoxLayout, QApplication
from PyQt5.QtCore import QThread, QUrl, Qt, QEvent, QTimer 
from PyQt5.QtGui import QTextTableFormat, QTextCursor, QImage, QTextImageFormat
from PyQt5.QtGui import QImage, QTextImageFormat, QTextCursor, QTextTableFormat, QTextDocument

import pandas as pd
import random, os
import matplotlib.pyplot as plt
from io import BytesIO
import numbers

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
        self._inserted_figures = {}
        self._image_cache = {}

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._resize_images_in_textedit)

        self.init_ui()

    def init_ui(self):
        """
        Connects UI buttons and actions to their corresponding slot functions.
        Handles button click bindings for training and statistics.
        """
        self.add_splitter_between_vertical_layouts()
        self.ui.test_generate_tests.installEventFilter(self)
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

        default_route = os.path.expanduser("~")
        self.ui.lineEdit_root.setText(default_route)
        self.modelsController.set_path(default_route)

    def eventFilter(self, source, event):
        if source == self.ui.test_generate_tests and event.type() == QEvent.Resize:
            self._resize_timer.start(200)
        return super().eventFilter(source, event)


    def _resize_images_in_textedit(self):
        if not hasattr(self, "_inserted_figures"):
            return


        text_edit = self.ui.test_generate_tests
        widget_width = text_edit.viewport().width()


        MIN_WIDTH = 400
        MAX_WIDTH = 1000
        target_width = max(MIN_WIDTH, min(widget_width, MAX_WIDTH))


        for img_name, original_qimage in self._inserted_figures.items():
            if img_name in self._image_cache and target_width in self._image_cache[img_name]:
                scaled = self._image_cache[img_name][target_width]
            else:
                scaled = original_qimage.scaledToWidth(
                target_width, Qt.SmoothTransformation
                )
                self._image_cache.setdefault(img_name, {})[target_width] = scaled

            doc = text_edit.document()
            doc.addResource(QTextDocument.ImageResource, QUrl(img_name), scaled)

    def add_splitter_between_vertical_layouts(self):
        """
        Replace horizontalLayout_12 with a QSplitter that allows resizing
        between the two vertical layouts (verticalLayout_12 and verticalLayout_11).
        """
        # Create the splitter (horizontal = left/right)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setOpaqueResize(False)

        # --- Left container ---
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        while self.ui.verticalLayout_12.count():
            item = self.ui.verticalLayout_12.takeAt(0)
            if item.widget():
                left_layout.addWidget(item.widget())
            elif item.layout():
                left_layout.addLayout(item.layout())
            elif item.spacerItem():
                left_layout.addItem(item.spacerItem())

        # --- Right container ---
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)

        while self.ui.verticalLayout_11.count():
            item = self.ui.verticalLayout_11.takeAt(0)
            if item.widget():
                right_layout.addWidget(item.widget())
            elif item.layout():
                right_layout.addLayout(item.layout())
            elif item.spacerItem():
                right_layout.addItem(item.spacerItem())

        # Add both containers to the splitter
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)

        # Optional: set initial relative sizes (e.g., 50/50)
        splitter.setSizes([300, 300])

        # Remove the old contents of horizontalLayout_12
        while self.ui.horizontalLayout_12.count():
            item = self.ui.horizontalLayout_12.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Add the splitter into the original horizontal layout
        self.ui.horizontalLayout_12.addWidget(splitter)

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
        self.ui.generate_results.setEnabled(enable)

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
        if self.ui.radioButton_yes.isChecked():
           if not self.check_columns_and_rows(): 
               show_error("Comprueba la tabla de validación debido a que se ha eliminado uno de los datasets. No se puede ejecutar el entrenamiento.")
               return 

        threshold = self.ui.threshold.value()
        if threshold == 0:
            show_error("Debe de introducir un valor mayor a 0 en el campo threshold para poder generar resultados para los modelos.")
            return
        
        self.enable_buttons(False)

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
        results = self.worker.get_results()
        num_datasets = len(self.datasetController.get_all_datasets())
        num_models = len(self.modelsController.get_models())

        if  self.validate_data_statistical(num_datasets, num_models):
            show_error("Debe introducir al menos 8 datasets y 2 algoritmos para poder ejecutar los test estadísticos.") 
            return

        all_empty = all(len(v) == 0 for v in results.values())
        if all_empty:
            show_error("Ninguno de los datasets contienen resultados válidos.")
            return

        self.enable_buttons(False)
        QApplication.processEvents()

        test_selected = self.ui.test_comboBox.currentText()
        post_hoc_selected = self.ui.postHoc_comboBox.currentText()
        alpha_selected = self.ui.alpha_comboBox.currentText()
        alpha_value = float(alpha_selected)
        text_edit = self.ui.test_generate_tests
        text_edit.clear()
        document = text_edit.document()

        if not hasattr(self, "_inserted_figures"):
            self._inserted_figures = {}

        def _end_cursor():
            c = QTextCursor(document)
            c.movePosition(QTextCursor.End)
            return c

        def _space(c, n=1):
            for _ in range(n):
                c.insertBlock()

        def _make_table_format():
            fmt = QTextTableFormat()
            fmt.setBorder(1)
            fmt.setCellPadding(4)
            fmt.setCellSpacing(2)
            return fmt

        def _insert_heading(text):
            c = _end_cursor()
            _space(c, 1)
            c.insertText(text)
            _space(c, 1)

        def _insert_matrix_table(df, datasets, models):
            c = _end_cursor()
            table_format = _make_table_format()
            filas = len(datasets) + 1
            columnas = len(models) + 1
            table = c.insertTable(filas, columnas, table_format)

            for col in range(1, columnas):
                model_name = str(models[col - 1])
                table.cellAt(0, col).firstCursorPosition().insertText(model_name)

            for fila in range(1, filas):
                dataset_name = str(datasets[fila - 1])
                table.cellAt(fila, 0).firstCursorPosition().insertText(dataset_name)
                for col in range(1, columnas):
                    valor = df.iloc[fila - 1, col - 1]
                    texto = f"{valor:.2f}" if pd.notnull(valor) and isinstance(valor, numbers.Number) else "N/A"
                    table.cellAt(fila, col).firstCursorPosition().insertText(texto)

        def _insert_dataframe_table(dataframe):
            c = _end_cursor()
            table_format = _make_table_format()
            rows, cols = dataframe.shape
            table = c.insertTable(rows + 1, cols, table_format)

            for col in range(cols):
                table.cellAt(0, col).firstCursorPosition().insertText(str(dataframe.columns[col]))

            for fila in range(rows):
                for col in range(cols):
                    value = dataframe.iat[fila, col]
                    text = f"{value:.4f}" if isinstance(value, float) else str(value)
                    table.cellAt(fila + 1, col).firstCursorPosition().insertText(text)

        def _insert_figure_from_matplotlib(figure):
            img_name = f"stat_fig_{random.randint(1000,9999)}.png"
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            image = QImage()
            image.loadFromData(buffer.read())
            buffer.close()
            if not image.isNull():
                image_format = QTextImageFormat()
                image_format.setName(img_name)
                document.addResource(QTextDocument.ImageResource, QUrl(img_name), image)
                c = _end_cursor()
                c.insertImage(image_format)
                if not hasattr(self, "_inserted_figures"):
                    self._inserted_figures = {}
                self._inserted_figures[img_name] = image
            plt.close(figure)
            plt.clf()
        # ----------------------------------------------------------- #
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

            _insert_heading(f"📊 Resultados para la métrica '{metric_selected}'\n")
            _insert_matrix_table(df, datasets, models)

            c = _end_cursor()
            _space(c, 2)

            a = self.statistic.run_test(test_selected, df, alpha=alpha_value, post_hoc_selected=post_hoc_selected)

            if isinstance(a, list):
                result_entry["test_results"] = {}
                data_to_table = a[0]
                result_entry["test_results"] = data_to_table.to_dict()

                _insert_heading(f"📊 Resultados Test estadístico '{test_selected}'\n")
                _insert_dataframe_table(data_to_table)

                c = _end_cursor()
                _space(c, 2)

                if "anova" in test_selected.lower():
                    table_data = a[1]
                    result_entry["post_hoc_results"] = table_data.to_dict()
                    _insert_dataframe_table(table_data)

                    c = _end_cursor()
                    _space(c, 2)
                    c.insertText("⚠️ No se generarán resultados del post-hoc porque el test seleccionado no es un test de rangos.\n")
                    continue

                if not ("Friedman" in test_selected or "Quade" in test_selected):
                    c.insertText("⚠️ No se generarán resultados del post-hoc porque el test seleccionado no es un test de rangos.\n")
                    self.statistical_results[metric_selected] = result_entry

                _space(c, 1)
                c.insertText(f"📊 Resultados Test estadístico '{post_hoc_selected}'\n")
                _space(c, 1)

                if post_hoc_selected != "Nemenyi":
                    table_data = a[1]
                    result_entry["post_hoc_results"] = table_data.to_dict()
                    _insert_dataframe_table(table_data)
                    c = _end_cursor()

                figure = a[-1]
                result_entry["_figures"] = [figure]
                _insert_figure_from_matplotlib(figure)

            elif isinstance(a, pd.DataFrame):
                result_entry["test_results"] = a.to_dict()
                a.reset_index(inplace=True)

                _insert_heading(f"📊 Resultados Test estadístico '{test_selected}'\n")
                _insert_dataframe_table(a)

                c = _end_cursor()
                _space(c, 3)

            else:
                result_entry["test_results"] = str(a)
                c = _end_cursor()
                _space(c, 1)
                c.insertText(f"📊 Resultados Test estadístico '{test_selected}'\n")
                c.insertText(str(a))
                _space(c, 2)

            self.statistical_results[metric_selected] = result_entry
            c = _end_cursor()
            _space(c, 2)

        self.enable_buttons(True)


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