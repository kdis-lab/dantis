from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
from PyQt5.QtCore import QThread, QObject, pyqtSignal
from PyQt5.QtGui import QTextTableFormat, QTextCursor, QImage, QTextImageFormat
from collections import defaultdict
import base64
import pandas as pd
import random
import os

from controller.statds_statistical_controller import StatisticalTestController
from controller.model_controller import ModelController
from core.utils import show_error
from core.dataset_manager import DatasetManager
from controller.training_orchestrator import TrainingOrchestrator

class ViewResults(QWidget):
    def __init__(self, main_window, ui, 
                datasetController,
                modelsController,
                metricsController,
                validationController):
        super().__init__()
        self.ui = ui
        self.main_window = main_window

        # Controllers
        #self.dataset_manager = DatasetManager(datasetController, validationController)
        self.training_orchestrator = TrainingOrchestrator(datasetController, 
                                                          modelsController, 
                                                          metricsController, 
                                                          validationController)
        
        self.modelsController = modelsController
        self.metricsController = metricsController
        self.validationController = validationController
        self.statistic = StatisticalTestController()

        self.init_ui()


    def init_ui(self):
        # FOR TRAINING MODELS
        self.ui.generate_results.clicked.connect(self.on_generate_results_click)
        # FOR STATISTICAL TESTS
        self.ui.generate_tests.clicked.connect(self.on_generate_statistical_tests_click)
        # DOWNLOAD DATA
        self.ui.download_data.clicked.connect(self.download_data_click)
        # MODIFY ROOT SAVE MODELS
        self.ui.modify_root.clicked.connect(self.modify_root)

        # FOR STATISTICAL TESTS   
        self.ui.test_comboBox.addItems(self.statistic.list_available_tests())
        self.ui.postHoc_comboBox.addItems(self.statistic.list_available_post_hoc())
        self.ui.alpha_comboBox.addItems(['0.01', '0.05', '0.1'])
        self.ui.alpha_comboBox.setCurrentText('0.05')  # Default to 0.05

        self.enable_buttons(True)

        ruta_por_defecto = os.path.expanduser("~")
        self.ui.lineEdit_root.setText(ruta_por_defecto)
        self.modelsController.set_path(ruta_por_defecto)

    def modify_root(self): 
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecciona una carpeta",
            ""  # Ruta inicial opcional
        )
        if path:
            self.ui.lineEdit_root.setText(path)
            self.modelsController.set_path(path)

    def enable_buttons(self, enable: bool):
        self.ui.test_comboBox.setEnabled(enable)
        self.ui.postHoc_comboBox.setEnabled(enable)
        self.ui.generate_tests.setEnabled(enable)
        self.ui.alpha_comboBox.setEnabled(enable)

    def procesar_config(self, datasets, x_cols, y_cols, models, metrics, val_opts):
        # Verificar si datasets están vacíos o sin datos reales
        if not datasets or all(ds.data.empty for ds in datasets.values()):
            show_error(f"No se ha introducido ninguno dataset. No se puede llevar a cabo el entrenamiento.")
            return False

        # Verificar si x_cols están vacíos o ninguna columna está marcada como True
        if not x_cols or all(not any(cols.values()) for cols in x_cols.values()):
            show_error(f"No se ha introducido ninguna x_col. No se puede llevar a cabo el entrenamiento.")
            return False

        # Verificar si y_cols está vacío o todos son vacíos / None
        if not y_cols or all(not y for y in y_cols.values()):
            show_error(f"No se ha introducido ninguna y_col. No se puede llevar a cabo el entrenamiento.")
            return False

        # Verificar si models está vacío
        if not models:
            show_error(f"No se ha introducido ningun modelo. No se puede llevar a cabo el entrenamiento.")
            return False

        # Verificar si metrics está vacío
        if not metrics:
            show_error(f"No se ha introducido ninguna métrica. No se puede llevar a cabo el entrenamiento.")
            return False

        # Verificar si val_opts está vacío
        if not val_opts:
            show_error(f"No se ha introducido ningun tipo de validación. No se puede llevar a cabo el entrenamiento.")
            return False
        
        return True

    def on_generate_results_click(self):
        self.enable_buttons(False)

        # Comprobación de contenido de threshold
        threshold = self.ui.threshold.value()
        if threshold == 0:
            show_error("Debe de introducir un valor mayor a 0 en el campo threshold para poder generar resultados para los modelos.")
            return
        
        (datasets, x_cols, y_cols,
         models, metrics, val_opts) = self.training_orchestrator.collect_training_info()

        if not self.procesar_config(datasets, x_cols, y_cols, models, metrics, val_opts):
            return

        # Recogemos los datos de los modelos y los mostramos en el QTextEdit
        info_text = self.format_training_info(
            datasets, x_cols, y_cols,
            models, metrics, val_opts, threshold
        )
        self.ui.text_generate_results.setHtml(info_text)

        # Crear thread y worker
        self.thread = QThread()
        self.worker = TrainingWorker(
            datasets, x_cols, y_cols,
            models, metrics, val_opts, threshold, self.modelsController, self)
        self.worker.moveToThread(self.thread)

        # Conectar señales
        self.thread.started.connect(self.worker.run)
        self.worker.notify_training.connect(self.append_training_message)
        self.worker.generate_result.connect(self.append_training_result)
        #self.worker.finished.connect(self.remove_training_message)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Conectar para reactivar botones al finalizar
        self.worker.finished.connect(lambda: self.enable_buttons(True))

        # Iniciar el hilo
        self.thread.start()


    def append_training_message(self):
        """
        Añade '<b>Entrenando...</b>' al final del QTextEdit si no está ya.
        """
        # Forzar el cursor al final del QTextEdit
        cursor = self.ui.text_generate_results.textCursor()
        cursor.movePosition(QTextCursor.End)  # Muy importante
        self.ui.text_generate_results.setTextCursor(cursor)

        # Insertar HTML con marcador (comentario, no id)
        html_marker = '<br/><p id="entrenando-marker" style="color: gray; font-weight: bold;">Entrenando...</p>'
        cursor.insertHtml(html_marker)

        # Forzar actualización visual
        #self.ui.text_generate_results.repaint()
        #QApplication.processEvents()

        # 🔽 Forzar scroll al final
        cursor = self.ui.text_generate_results.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.ui.text_generate_results.setTextCursor(cursor)

    def append_training_result(self, new_html: str):
        """
        Sustituye la última ocurrencia de '<b>Entrenando...</b>' por el resultado recibido.
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
            # Añadir al final si no hay marcador
            self.ui.text_generate_results.setHtml(full_html + new_html)

        # 🔽 Forzar scroll al final
        cursor = self.ui.text_generate_results.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.ui.text_generate_results.setTextCursor(cursor)

        

    ## ESTA FUNCION DE PODRIA ACORTAR BASTANTE Y CREAR LA PARTE DE DESCARGAR EN UN CONTROLADOR
    def download_data_click(self):
        # Obtener el texto de los QTextEdit
        texto_entrenamiento = self.ui.text_generate_results.toPlainText()
        texto_test = self.ui.test_generate_tests.toPlainText()

        # Pedir al usuario dónde guardar el archivo TXT
        opciones = QFileDialog.Options()
        archivo_txt, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Informe como TXT",
            "",
            "Text Files (*.txt)",
            options=opciones
        )

        if archivo_txt:
            with open(archivo_txt, mode='w', encoding='utf-8') as archivo:
                archivo.write("INFORME DEL ENTRENAMIENTO DE MODELOS Y TEST ESTADÍSTICOS EJECUTADOS\n")
                archivo.write("\n=== RESULTADOS ENTRENAMIENTO DE MODELOS ===\n")
                archivo.write(texto_entrenamiento + "\n")
                archivo.write("\n\n")  # Línea en blanco entre secciones
                archivo.write("\n=== RESULTADOS DE LOS TEST ESTADÍSTICOS ===\n")
                archivo.write(texto_test + "\n")


    def on_generate_statistical_tests_click(self):
        results = self.worker.get_results()
        example_data = list(results.items())[0][1]
        if  self.validate_data_statistical(example_data): 
            return
        # Datos desde la UI
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

            # ⬇️ Aseguramos nuevo cursor limpio al final del documento
            cursor = QTextCursor(document)
            cursor.movePosition(QTextCursor.End)

            cursor.insertBlock()
            cursor.insertText(f"📊 Resultados para la métrica '{metric_selected}'\n")
            cursor.insertBlock()

            formato_tabla = QTextTableFormat()
            formato_tabla.setBorder(1)
            formato_tabla.setCellPadding(4)
            formato_tabla.setCellSpacing(2)

            tabla = cursor.insertTable(filas, columnas, formato_tabla)

            # Encabezados columnas
            for col in range(1, columnas):
                model_name = str(models[col - 1])
                celda = tabla.cellAt(0, col)
                celda_cursor = celda.firstCursorPosition()
                celda_cursor.insertText(model_name)

            # Rellenar filas con datos
            for fila in range(1, filas):
                dataset_name = str(datasets[fila - 1])
                tabla.cellAt(fila, 0).firstCursorPosition().insertText(dataset_name)

                for col in range(1, columnas):
                    valor = df.iloc[fila - 1, col - 1]
                    texto = f"{valor:.2f}" if pd.notnull(valor) else "N/A"
                    tabla.cellAt(fila, col).firstCursorPosition().insertText(texto)

            cursor.movePosition(QTextCursor.End)
            cursor.insertBlock()
            cursor.insertBlock()

            a = self.statistic.run_test(test_selected, df, alpha=alpha_value)

            if isinstance(a, dict):
                # Mostrar resultados principales
                cursor.insertBlock()
                cursor.insertText(f"📊 Resultados Test estadístico '{test_selected}'\n")
                cursor.insertBlock()

                formato_tabla = QTextTableFormat()
                formato_tabla.setBorder(1)
                formato_tabla.setCellPadding(4)
                formato_tabla.setCellSpacing(2)
                
                for k, v in a.items():
                    if k == "figure_base64":
                        continue

                    if isinstance(v, (list, tuple)) and all(isinstance(i, (list, tuple)) for i in v):
                        rows = len(v)
                        cols = len(v[0]) if rows > 0 else 0

                        tabla = cursor.insertTable(rows + 1, cols, formato_tabla)

                        # Encabezados
                        for col in range(cols):
                            celda = tabla.cellAt(0, col)
                            celda_cursor = celda.firstCursorPosition()
                            celda_cursor.insertText(f"{k} {col + 1}")

                        # Datos
                        for fila in range(1, rows + 1):
                            for col in range(cols):
                                celda = tabla.cellAt(fila, col)
                                celda_cursor = celda.firstCursorPosition()
                                valor = v[fila - 1][col]
                                texto = f"{valor:.4f}" if isinstance(valor, float) else str(valor)
                                celda_cursor.insertText(texto)

                        cursor.movePosition(QTextCursor.End)
                        cursor.insertBlock()

                    elif isinstance(v, dict):
                        # Diccionario interno → tabla clave-valor
                        tabla = cursor.insertTable(len(v) + 1, 2, formato_tabla)

                        # Encabezado
                        tabla.cellAt(0, 0).firstCursorPosition().insertText(f"{k} (Clave)")
                        tabla.cellAt(0, 1).firstCursorPosition().insertText("Valor")

                        for idx, (subk, subv) in enumerate(v.items()):
                            celda_key = tabla.cellAt(idx + 1, 0)
                            celda_val = tabla.cellAt(idx + 1, 1)
                            celda_key.firstCursorPosition().insertText(str(subk))
                            texto = f"{subv:.4f}" if isinstance(subv, float) else str(subv)
                            celda_val.firstCursorPosition().insertText(texto)

                        cursor.movePosition(QTextCursor.End)
                        cursor.insertBlock()

                    elif isinstance(v, pd.DataFrame):
                        # DataFrame → tabla
                        rows, cols = v.shape
                        tabla = cursor.insertTable(rows + 1, cols, formato_tabla)

                        # Encabezados
                        for col in range(cols):
                            celda = tabla.cellAt(0, col)
                            celda_cursor = celda.firstCursorPosition()
                            celda_cursor.insertText(v.columns[col])

                        # Datos
                        for fila in range(rows):
                            for col in range(cols):
                                celda = tabla.cellAt(fila + 1, col)
                                celda_cursor = celda.firstCursorPosition()
                                valor = v.iat[fila, col]
                                texto = f"{valor:.4f}" if isinstance(valor, float) else str(valor)
                                celda_cursor.insertText(texto)

                        cursor.movePosition(QTextCursor.End)
                        cursor.insertBlock()
                    else:
                        # Valor simple → línea de texto
                        texto = f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                        cursor.insertText(texto + "\n")
                        cursor.insertBlock()

                # Mostrar figura si existe
                if "figure_base64" in a and a["figure_base64"]:
                    img_data = base64.b64decode(a["figure_base64"])
                    image = QImage()
                    image.loadFromData(img_data)
                    if not image.isNull():
                        image_format = QTextImageFormat()
                        img_name = f"stat_fig_{random.randint(1000,9999)}.png"
                        document.addResource(document.ImageResource, img_name, image)
                        image_format.setName(img_name)
                        cursor.insertImage(image_format)
                        cursor.insertBlock()
            elif isinstance(a, pd.DataFrame):
                cursor.insertBlock()
                cursor.insertText(f"📊 Resultados Test estadístico '{test_selected}'\n")
                cursor.insertBlock()

                rows, cols = a.shape
                tabla = cursor.insertTable(rows + 1, cols, formato_tabla)

                # Encabezados
                for col in range(cols):
                    celda = tabla.cellAt(0, col)
                    celda_cursor = celda.firstCursorPosition()
                    celda_cursor.insertText(str(a.columns[col]))

                # Datos
                for fila in range(rows):
                    for col in range(cols):
                        celda = tabla.cellAt(fila + 1, col)
                        celda_cursor = celda.firstCursorPosition()
                        valor = a.iat[fila, col]
                        texto = f"{valor:.4f}" if isinstance(valor, float) else str(valor)
                        celda_cursor.insertText(texto)

                cursor.insertBlock()
            else:
                cursor.insertText(f"📊 Resultados Test estadístico '{test_selected}'\n")
                cursor.insertText(str(a))
                cursor.insertBlock()

            cursor.movePosition(QTextCursor.End)
            cursor.insertBlock()

    """
    def add_info_to_split_datasets (self, datasets, split_datasets): 
        final_split_datasets = {}

        # Paso 1: Construir split_datasets a partir de datasets_particionados
        if split_datasets:
            for group_id, splits in split_datasets.items():
                final_split_datasets[group_id] = {}
                for split_name, file_name in splits.items():
                    # Buscar el DatasetInfo que coincide por nombre
                    for key, ds_info in list(datasets.items()):
                        if ds_info.name == file_name:
                            # Insertar en split_datasets y eliminar de datasets
                            final_split_datasets[group_id][split_name] = ds_info
                            del datasets[key]
                            break 
        return datasets, final_split_datasets
    """

    """
    ## ESTA FUNCION SE PODRIA QUITAR DE AQUI Y METERLA EN ALGUNOS DE LOS CONTROLADORES O CREAR ALGUNO
    def get_entered_values(self, all_parameters=False): 
        original_datasets = self.datasetController.datos
        models = self.modelsController.models
        data_split_datasets = self.validationController.get_info_tabla()
        
        # Introduce relevant information from partitioned datasets
        datasets, split_datasets = self.add_info_to_split_datasets(original_datasets, data_split_datasets)
        print("datasets:", datasets)
        print("split_datasets:", split_datasets)

        # For complete datasets
        x_cols_all_datasets = [info.x_col for info in datasets.values()]
        # Filtrar cada diccionario
        x_cols_datasets = [{k: v for k, v in d.items() if v} for d in x_cols_all_datasets]
        y_cols_datasets = [info.y_col for info in datasets.values()]

        # For partitioned datasets
        if split_datasets:
            x_cols_all_split_datasets = [info.x_col for info in split_datasets.values()]
            # Filtrar cada diccionario
            x_cols_split_datasets = [{k: v for k, v in d.items() if v} for d in x_cols_all_split_datasets]
            y_cols_split_datasets = [info.y_col for info in split_datasets.values()]

        if all_parameters:
            metrics = self.metricsController.checkboxes
            options_val = self.validationController.get_validation_config()
            return datasets, split_datasets, x_cols_datasets, y_cols_datasets, x_cols_split_datasets, y_cols_split_datasets, models, metrics, options_val
        else:
            return datasets, split_datasets, x_cols_datasets, y_cols_datasets, x_cols_split_datasets, y_cols_split_datasets,models
    """


    ## ESTA FUNCION SE PODRIA QUITAR DE AQUI Y METERLA EN ALGUNOS DE LOS CONTROLADORES O CREAR ALGUNO
    def format_training_info(self, datasets, x_cols, y_cols,
                             models, metrics, val_opts, threshold) -> str:
        lines = []
        lines.append("<h2>DATOS PARA ENTRENAMIENTO:</h2>")

        # Separar completos y particionados
        full_datasets = {idx: ds for idx, ds in datasets.items() if not isinstance(ds, dict)}
        split_datasets = {idx: ds for idx, ds in datasets.items() if isinstance(ds, dict)}

        # Datasets completos
        if full_datasets:
            lines.append("<b>Datasets completos</b>:<br>")
            for idx, info in full_datasets.items():
                lines.append(f"<u>Dataset {idx + 1}: {info.name}</u><br>")
                entradas = ", ".join([k for k, v in x_cols[idx].items() if v])
                lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>Entradas:</b> {entradas}<br>")
                lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>Salidas:</b> {y_cols[idx]}<br><br>")
            lines.append("<br>")

        # Datasets particionados
        if split_datasets:
            lines.append("<b>Datasets particionados</b>:<br>")
            for idx, parts in split_datasets.items():
                lines.append(f"<u>Dataset {idx + 1}:</u><br>")
                for split_name, info in parts.items():
                    lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>{split_name}:</b> {info.name}<br>")
                entradas = ", ".join(x_cols[idx].keys())
                lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Entrada/s:</b> {entradas}<br>")
                lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Salida/s:</b> {y_cols[idx]}<br><br>")
            lines.append("<br>")

        lines.append("<b>Modelos</b>:<br>")
        for m in models.values():
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;- {m['model']}<br>")
        lines.append("<br>")

        lines.append("<b>Métricas</b>:<br>")
        for met in metrics:
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;- {met}<br>")
        lines.append("<br>")

        lines.append("<b>Opciones validación:</b><br>")
        for k, v in val_opts.items():
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;- {k}: {v}<br>")
        lines.append("<br>")

        lines.append(f"<b>Threshold</b>: {threshold}<br>")
        lines.append("<hr>")

        return "\n".join(lines)

    
    def validate_data_statistical(self, data):
        #Valida que es posible aplicar los datos estadisticos.
        datasets, models = data.shape
        print(datasets, models)
        if datasets < 8 or models < 2:
            return True
        else:
            return False
    


### TODA ESTA CLASE DEBERIA DE MOVERSE A OTRO FICHERO, OTRO CONTROLADOR O CREAR UNO NUEVO PARA DICHA CLASE NADA MAS
class TrainingWorker(QObject):
    generate_result = pyqtSignal(str)
    notify_training = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, datasets, x_cols, y_cols, models,
                 metrics, options_val, threshold, modelsController, controller):
        super().__init__()
        self.datasets = datasets
        self.models = models
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.metrics = metrics
        self.options_val = options_val
        self.threshold = threshold
        self.controller = controller
        self.modelController = modelsController
        self.results_generated = defaultdict(list)

    def _initialise_results_structure(self):
        for metrica in self.metrics:
            if metrica not in self.results_generated:
                self.results_generated[metrica] = defaultdict(list)

    def _convert_results_to_dataframe(self):
        for metrica in self.metrics:
            df = pd.DataFrame(self.results_generated[metrica])
            self.results_generated[metrica] = df.set_index("Datasets")

    def _emit_result(self, dataset, results):
        texto_formateado = self.format_results(dataset, results)
        self.generate_result.emit(texto_formateado)

    def strip_extension(self, filename):
            return filename.rsplit('.', 1)[0]

    def _build_dataset_name(self, dataset):
        parts = [self.strip_extension(dataset['train'].name)]

        # Agregar 'val' solo si no es None ni está vacío
        val_info = dataset.get('val')
        if val_info is not None and hasattr(val_info, 'data') and val_info.data is not None and not val_info.data.empty:
            parts.append(self.strip_extension(val_info.name))

        parts.append(self.strip_extension(dataset['test'].name))

        return "__".join(parts)

    def _update_generated_results(self, model_name, dataset, results):
        name = ""
        if isinstance(dataset, dict) and 'train' in dataset and 'test' in dataset:
            name = self._build_dataset_name(dataset)
        else: 
            name = self.strip_extension(dataset.name)

        for metrica in self.metrics:
            valor = results["test"]["metrics"].get(metrica)
            self.results_generated[metrica][model_name].append(valor if valor is not None else None)

            if name not in self.results_generated[metrica]["Datasets"]:
                self.results_generated[metrica]["Datasets"].append(name)

    def process_dataset(self, model_name, hyperparams, model_path=None):
        folds = None
        X_test = None
        y_test = None

        if not self.datasets:
            return
        
        for id, dataset in self.datasets.items():
            self.notify_training.emit()
            folds, (X_test, y_test) = self.modelController.split_dataset(
                dataset, self.x_cols[id], self.y_cols[id], self.options_val)
                
            results = self.modelController.generate_results_by_dataset(
                id, dataset, self.y_cols, model_name, hyperparams, model_path, folds, 
                X_test, y_test, self.metrics, self.threshold 
            )
            self._update_generated_results(model_name, dataset, results)
            self._emit_result(dataset, results)

    def run(self):
        self._initialise_results_structure()      
        for model_id, model_data in self.models.items():
            model_name = model_data['model']
            hyperparams = model_data['hyperparameters']
            model_path = None
            if model_data.get('paths'):
                model_path = model_data['paths']

            self.process_dataset(model_name, hyperparams, model_path)

        self._convert_results_to_dataframe()
        self.finished.emit()

    def get_results(self):
        """
        Devuelve los resultados generados por el worker.
        """
        return self.results_generated
    

    def format_results(self, dataset_name, resultados: dict) -> str:
        texto = []
        texto.append(f" <b>Resultados para Dataset {resultados.get('dataset_id')+1} con modelo {resultados.get('model')}</b><br>")
        texto.append("<br>")
        texto.append("--- Validación Cruzada (Folds) ---<br>")

        folds = resultados.get("folds", [])
        for fold in folds:
            texto.append(f"&nbsp;Fold {fold['fold']}:<br>")
            texto.append(f"&nbsp;&nbsp;Métricas:<br>")
            for clave, valor in fold['metrics'].items():
                texto.append(f"&nbsp;&nbsp;&nbsp;- {clave}: {valor}<br>")

        texto.append("<br>")
        texto.append("--- Evaluación con Test Final ---<br>")
        test = resultados.get("test", {})
        texto.append("&nbsp;Métricas:<br>")
        for clave, valor in test.get("metrics", {}).items():
            texto.append(f"&nbsp;&nbsp;- {clave}: {valor}<br>")

        return "\n".join(texto)