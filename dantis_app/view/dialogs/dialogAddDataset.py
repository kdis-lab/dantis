from controller.dataset_controller import load_predefined_datasets_name

import os
import pandas as pd

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QRadioButton, QHBoxLayout, QCheckBox
from PyQt5.QtWidgets import QFormLayout, QAction, QMenu, QGroupBox, QWidget
from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton, QSizePolicy, QFileDialog
from PyQt5.QtWidgets import QLineEdit, QScrollArea, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QPoint, Qt



class DialogAddDataset(QDialog):
    def __init__(self, ui, parent=None, datasetController = None):
        super().__init__(parent)     
        self.parent_window = parent  # Referencia a la ventana principal
        self.stacked_widget = self.parent_window.ui.stackedWidget
        self.ui = ui

        self.datasets_widgets = []
        #self.ui.radioButton_supervisado.toggled.connect(self.actualizar_todos_y_col)

        self.datasetController = datasetController
        self.contador = self.datasetController._actual_id

        self.github_repo_url = "https://api.github.com/repos/Elenitaalva/Datasets/contents"
        self.datasets_predefinidos = load_predefined_datasets_name(self.github_repo_url)

        self.setWindowTitle("Añadir Dataset")        

        # Layout principal
        layout = QVBoxLayout()

        # Crear los radio buttons para elegir entre Local y URL
        self.radio_predefinido = QRadioButton("Dataset predefinido")
        self.radio_local = QRadioButton("Local")
        self.radio_url = QRadioButton("URL")
        
        # Establecer "Predefinido" como opción predeterminada
        self.radio_predefinido.setChecked(True)

        # Conectar los radio buttons a las funciones de cambio de vista
        self.radio_predefinido.toggled.connect(self.toggle_input_method)
        self.radio_local.toggled.connect(self.toggle_input_method)
        self.radio_url.toggled.connect(self.toggle_input_method)

        # Crear el layout para los radio buttons
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_predefinido)
        radio_layout.addWidget(self.radio_local)
        radio_layout.addWidget(self.radio_url)

        # Crear el formulario donde se mostrarán los widgets dependiendo de la selección
        self.form_layout = QFormLayout()

        # Crear un campo para la selección de un dataset predefinido (se muestra si se selecciona "Dataset predefinido")
        self.predefinido_label = QLabel("\nSelecciona un dataset predefinido:")
        self.predefinido_button = QComboBox()
        self.predefinido_button.addItems(self.datasets_predefinidos)
        self.predefinido_button.setDisabled(False) 

        # Crear el botón para cargar el dataset predefinido
        self.predefinio_button_load = QPushButton("Cargar Dataset predefinido")
        self.predefinio_button_load.setDisabled(False)   
        self.predefinio_button_load.clicked.connect(self.load_github_dataset) 

        # Agregar el botón local al formulario
        self.form_layout.addRow(self.predefinido_label)
        self.form_layout.addRow(self.predefinido_button)
        self.form_layout.addRow(self.predefinio_button_load)

        # Crear un campo para la selección de archivo local (se muestra si se selecciona "Local")
        self.local_label = QLabel("\nSelecciona un dataset de forma local: ")
        self.local_button = QPushButton("Seleccionar archivo")
        self.local_button.setDisabled(True)  
        self.local_button.clicked.connect(self.load_local_file)

        # Agregar el botón local al formulario
        self.form_layout.addRow(self.local_label)
        self.form_layout.addRow(self.local_button)

        # Crear un campo de texto para la URL (se muestra si se selecciona "URL")
        self.url_label = QLabel("Introduce la URL del dataset:") 
        self.url_input = QLineEdit()
        self.url_input.setDisabled(True)  # Deshabilitar por defecto

        # Crear el botón para cargar el dataset desde la URL
        self.url_button = QPushButton("Cargar Dataset desde URL")
        self.url_button.setDisabled(True)  # Deshabilitar por defecto
        self.url_button.clicked.connect(self.load_url_file)

        # Agregar los widgets del formulario
        self.form_layout.addRow(QLabel(" "))
        self.form_layout.addRow(self.url_label, self.url_input)
        self.form_layout.addRow(self.url_button)

        # Añadir los radio buttons y el formulario al layout principal
        layout.addLayout(radio_layout)
        layout.addLayout(self.form_layout)

        # Crear un área de vista previa para mostrar el archivo CSV cargado
        self.preview_label = QLabel("\nArchivo cargado:")
        self.preview_text = QLabel("")  # Texto que mostrará el nombre del archivo cargado
        layout.addWidget(self.preview_label)
        layout.addWidget(self.preview_text)

        # Añadir botones de aceptar y cancelar
        button_layout = QHBoxLayout()
        self.accept_button = QPushButton("Aceptar")
        self.cancel_button = QPushButton("Cancelar")

        # Conectar el botón de aceptar para mostrar el archivo cargado en verticalLayout_6 de pagina 0
        self.accept_button.clicked.connect(self.confirm_selection)
        self.cancel_button.clicked.connect(self.reject)  # Esto cerrará el diálogo sin realizar acciones

        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Establecer el layout de la ventana
        self.setLayout(layout)

    def toggle_input_method(self):
        # Verificar qué opción está seleccionada
        if self.radio_predefinido.isChecked():
            self.predefinido_button.setEnabled(True)
            self.predefinio_button_load.setEnabled(True)
            self.local_button.setEnabled(False)
            self.url_input.setDisabled(True)
            self.url_button.setDisabled(True)
            self.preview_text.setText("") ## PARA LIMPIAR LA PARTE DE ARCHIVO CARGADO DE LA INTERFAZ
        elif self.radio_local.isChecked():
            # Si "Local" está seleccionado, habilitar el botón de selección de archivo
            self.predefinido_button.setEnabled(False)
            self.predefinio_button_load.setEnabled(False)
            self.local_button.setEnabled(True)
            self.url_input.setDisabled(True)
            self.url_button.setDisabled(True)
            self.preview_text.setText("") ## PARA LIMPIAR LA PARTE DE ARCHIVO CARGADO DE LA INTERFAZ
        elif self.radio_url.isChecked():
            # Si "URL" está seleccionado, habilitar el campo de texto y el botón
            self.predefinido_button.setEnabled(False)
            self.predefinio_button_load.setEnabled(False)
            self.local_button.setEnabled(False)
            self.url_input.setEnabled(True)
            self.url_button.setEnabled(True)
            self.preview_text.setText("") ## PARA LIMPIAR LA PARTE DE ARCHIVO CARGADO DE LA INTERFAZ
        

    ### FUNCIONES DE CONTROLADOR
    def load_local_file(self):
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, "Seleccionar archivos", "", "Archivos CSV (*.csv);;Todos los archivos (*)")
        
        if file_paths:
            self.file_path = file_paths  # Guarda la lista de rutas
            # Mostrar todos los nombres de archivos en el área de vista previa
            nombres_archivos = [os.path.basename(path) for path in file_paths]
            self.preview_text.setText("\n".join(nombres_archivos))
            

    def load_github_dataset(self):   
        valor = self.predefinido_button.currentText()
        #Carga el dataset seleccionado desde GitHub.
        if valor:
            self.file_path = f"https://raw.githubusercontent.com/Elenitaalva/Datasets/main/{valor}"
            self.preview_text.setText(valor)

    def load_url_file(self): 
        valor = self.url_input.text()
        self.file_path = valor
        self.preview_text.setText(valor)

    def create_dataset_layout(self, dataset_id):
        file_path = self.datasetController.get_data().get(dataset_id)
        columnas = self.datasetController.get_columns(dataset_id)
        loaded_file_label = QLabel(f"Dataset {dataset_id}: {file_path.name}")

        # Obtener configuración previa si existe
        x_col_values = self.datasetController.get_options_x_col(dataset_id) or {}
        y_col_value = self.datasetController.get_y_col(dataset_id)

        # Checkboxes X_col
        x_col_group = QGroupBox("X_col (Características)")
        checkbox_widget = QWidget()
        checkbox_layout = QVBoxLayout()
        x_col_checkboxes = []
        col_check_states = {}

        for col in columnas:
            checkbox = QCheckBox(col)

            # Si hay datos previos, usarlos; si no, marcar todos como True
            is_checked = x_col_values.get(col, True)
            checkbox.setChecked(is_checked)
            col_check_states[col] = is_checked

            checkbox.stateChanged.connect(lambda state, c=col: self.handle_checkbox_change(dataset_id, c, state, col_check_states))
            checkbox_layout.addWidget(checkbox)
            x_col_checkboxes.append(checkbox)

        self.datasetController.options_x_col = {
            cb.text(): cb.isChecked() for cb in x_col_checkboxes
        }
        self.datasetController.add_x_col(dataset_id, self.datasetController.options_x_col)

        checkbox_layout.addStretch()
        checkbox_widget.setLayout(checkbox_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(checkbox_widget)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        scroll_area.setFixedHeight(100)
        scroll_area.setFixedWidth(250)

        x_col_layout = QVBoxLayout()
        x_col_layout.addWidget(scroll_area)
        x_col_group.setLayout(x_col_layout)

        # Y_col combo
        y_col_combo = QComboBox()
        y_col_combo.addItems(columnas)

        y_label = QLabel("y_col")

        # Usar valor de y_col previo si está definido
        if y_col_value in columnas:
            y_col_combo.setCurrentText(y_col_value)
        else:
            y_col_value = y_col_combo.currentText()  # valor por defecto si no hay

        self.datasetController.add_y_col(dataset_id, y_col_value)

        y_col_combo.currentIndexChanged.connect(lambda: self.handle_checkbox_change_y(dataset_id, y_col_combo, y_col_value))

        def on_y_col_changed():
            seleccionado_y = y_col_combo.currentText()
            self.handle_checkbox_change_y(dataset_id, y_col_combo, seleccionado_y)
            self.actualizar_todos_y_col()

        y_col_combo.currentIndexChanged.connect(on_y_col_changed)

        # Botones
        visualization_button = QPushButton("Visualización")
        visualization_button.clicked.connect(lambda: self.visualization_dataset(dataset_id))  

        delete_button = QPushButton("Eliminar")
        delete_button.clicked.connect(lambda: self.delete_dataset(dataset_id, loaded_file_label, x_col_group, y_label, y_col_combo, visualization_button, delete_button))

        # Layout horizontal del dataset
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(loaded_file_label)
        dataset_layout.addWidget(x_col_group)
        dataset_layout.addWidget(y_label)
        dataset_layout.addWidget(y_col_combo)
        dataset_layout.addWidget(visualization_button)
        dataset_layout.addWidget(delete_button)

        self.layout_page0 = self.stacked_widget.widget(0).findChild(QScrollArea, "scrollArea_3")
        scroll_area_widget = self.layout_page0.widget()

        if scroll_area_widget.layout() is None:
            scroll_area_widget.setLayout(QVBoxLayout())

        scroll_area_widget.layout().addLayout(dataset_layout)

        self.datasets_widgets.append({
            "x_checkboxes": x_col_checkboxes,
            "y_label": y_label,
            "y_combo": y_col_combo
        })

        self.actualizar_todos_y_col()

    # Función para confirmar la selección y mostrar el archivo en verticalLayout_6
    def confirm_selection(self):
        if self.radio_local.isChecked() and self.preview_text.text().strip():
            for file_path in self.file_path:
                id = self.datasetController.add_datasets(file_path, 0)
                self.create_dataset_layout(id)

        elif self.radio_predefinido.isChecked() and self.preview_text.text().strip():
            id = self.datasetController.add_datasets(self.file_path, 1)
            self.create_dataset_layout(id)         

        elif self.radio_url.isChecked() and self.preview_text.text().strip():
            id = self.datasetController.add_datasets(self.file_path, 2)
            self.create_dataset_layout(id) 

        # Cerrar el diálogo
        self.accept()

    def handle_checkbox_change(self, id, col_name, state, col_check_states):
        col_check_states[col_name] = (state == Qt.Checked)
        self.datasetController.set_options_x_col(col_check_states)
        self.datasetController.add_x_col(id, col_check_states)

    def handle_checkbox_change_y(self, id, y_col_combo, seleccionado_y):
        seleccionado_y = y_col_combo.currentText()
        self.datasetController.add_y_col(id, seleccionado_y)

    def visualization_dataset(self, id):
        # Extraer los datos del archivo especificado
        datos_a_visualizar = self.datasetController.datos[id]

        # Crear un DataFrame con los datos
        if isinstance(datos_a_visualizar.data, pd.DataFrame):
            dataset = datos_a_visualizar.data.copy()  # Crear una copia para no modificar el original
        else:
            print("Error: los datos no están en formato DataFrame.")
            return

        # Variable para rastrear el DataFrame actualmente mostrado
        current_data = dataset.copy()

        # Crear un QDialog para la visualización del dataset
        dialog = QDialog(self)
        dialog.setWindowTitle("Visualización del Dataset")
        dialog.resize(800, 800)

        # Crear un QTableWidget para mostrar el contenido del dataset
        table_widget = QTableWidget()

        # Configurar el QTableWidget con los datos del DataFrame
        def update_table(data):
            nonlocal current_data
            current_data = data  # Actualizar el DataFrame actualmente mostrado
            num_rows, num_columns = data.shape
            table_widget.setRowCount(num_rows)
            table_widget.setColumnCount(num_columns)
            table_widget.setHorizontalHeaderLabels(data.columns)

            for row in range(num_rows):
                for col in range(num_columns):
                    item = QTableWidgetItem(str(data.iat[row, col]))
                    table_widget.setItem(row, col, item)

        update_table(dataset)

        # Crear un layout principal para el QDialog
        main_layout = QVBoxLayout()

        # Crear un layout para los controles de filtrado
        filter_layout = QHBoxLayout()

        # Agregar un combo box para seleccionar la columna
        column_selector = QComboBox()
        column_selector.addItems(dataset.columns)
        filter_layout.addWidget(QLabel("Columna:"))
        filter_layout.addWidget(column_selector)

        # Agregar un campo de texto para el valor del filtro
        filter_input = QLineEdit()
        filter_layout.addWidget(QLabel("Valor:"))
        filter_layout.addWidget(filter_input)

        # Agregar un botón para aplicar el filtro
        filter_button = QPushButton("Aplicar Filtro")
        filter_layout.addWidget(filter_button)

        # Agregar un botón para limpiar el filtro
        clear_filter_button = QPushButton("Limpiar Filtro")
        filter_layout.addWidget(clear_filter_button)

        # Conectar el botón de filtro a la lógica de filtrado
        def apply_filter():
            column = column_selector.currentText()
            value = filter_input.text()

            if value:
                filtered_data = current_data[current_data[column].astype(str).str.contains(value, na=False, case=False)]
                update_table(filtered_data)
            else:
                update_table(dataset)  # Restaurar el dataset original si no hay valor de filtro

        filter_button.clicked.connect(apply_filter)

        # Conectar el botón para limpiar el filtro
        def clear_filter():
            filter_input.clear()
            update_table(dataset)

        clear_filter_button.clicked.connect(clear_filter)

        # Crear un menú desplegable para ordenar las columnas
        def show_sort_menu(column_index):
            menu = QMenu()

            sort_asc_action = QAction("Orden Ascendente", menu)
            sort_desc_action = QAction("Orden Descendente", menu)
            reset_action = QAction("Mostrar Todo", menu)

            def sort_ascending():
                sorted_data = current_data.sort_values(by=current_data.columns[column_index], ascending=True)
                update_table(sorted_data)

            def sort_descending():
                sorted_data = current_data.sort_values(by=current_data.columns[column_index], ascending=False)
                update_table(sorted_data)

            def reset_table():
                update_table(dataset)

            sort_asc_action.triggered.connect(sort_ascending)
            sort_desc_action.triggered.connect(sort_descending)
            reset_action.triggered.connect(reset_table)

            menu.addAction(sort_asc_action)
            menu.addAction(sort_desc_action)
            menu.addAction(reset_action)

            header_pos = table_widget.mapToGlobal(table_widget.horizontalHeader().pos())
            menu.exec_(header_pos + QPoint(column_index * 100, table_widget.horizontalHeader().height()))

        table_widget.horizontalHeader().sectionClicked.connect(show_sort_menu)

        # Agregar los controles de filtrado y la tabla al layout principal
        main_layout.addLayout(filter_layout)
        main_layout.addWidget(table_widget)

        dialog.setLayout(main_layout)

        # Mostrar el diálogo de visualización
        dialog.exec_()

    
    def actualizar_todos_y_col(self):
        nuevos_grupos = []

        for grupo in self.datasets_widgets:
            x_checkboxes = grupo.get("x_checkboxes")
            y_label = grupo.get("y_label")
            y_combo = grupo.get("y_combo")

            # Verifica que los widgets aún existen
            if y_label is None or y_label.parent() is None or y_combo is None or y_combo.parent() is None:
                continue  # Ignora grupos eliminados

            y_col = y_combo.currentText()
            for cb in x_checkboxes:
                if cb is None or cb.parent() is None:
                    continue
                elif cb.text() == y_col:
                    cb.setChecked(False)
                    cb.setEnabled(False)
                else:
                    cb.setEnabled(True)

            # Solo mantenemos en la lista los grupos válidos
            nuevos_grupos.append(grupo)

        # Actualizamos con los grupos realmente existentes
        self.datasets_widgets = nuevos_grupos
    
    def clear_scrollArea3(self):
        scroll_area = self.stacked_widget.widget(0).findChild(QScrollArea, "scrollArea_3")
        if scroll_area is None:
            print("No se encontró el scrollArea_3.")
            return

        scroll_content = scroll_area.widget()
        if scroll_content is None:
            print("El scrollArea no tiene contenido.")
            return

        layout = scroll_content.layout()
        if layout is None:
            print("El contenido del scrollArea no tiene layout.")
            return

        while layout.count():
            item = layout.takeAt(0)
            if item.layout():
                sublayout = item.layout()
                while sublayout.count():
                    subitem = sublayout.takeAt(0)
                    widget = subitem.widget()
                    if widget:
                        widget.setParent(None)
            else:
                widget = item.widget()
                if widget:
                    widget.setParent(None)

        self.datasets_widgets.clear()
    
    def aniadir_nuevos_datos(self):
        self.clear_scrollArea3()
        # Recorremos todos los datasets actuales en el controller
        for id in self.datasetController.get_data():
            self.create_dataset_layout(id)


    def delete_dataset(self, id, label, x_col_group, y_label, y_col_combo, visualization, button):
        if self.datasetController.eliminar(id):
            # Buscar y eliminar widgets del layout
            label.setParent(None)
            x_col_group.setParent(None)
            y_label.setParent(None)
            y_col_combo.setParent(None)
            visualization.setParent(None)
            button.setParent(None)

            # Eliminar del registro datasets_widgets usando identidad del y_label
            self.datasets_widgets = [
                grupo for grupo in self.datasets_widgets
                if grupo["y_label"] != y_label
            ]

            self.clear_scrollArea3()
            self.aniadir_nuevos_datos()
        else:
            print("Error al eliminar dataset con ID:", id)


    def get_url(self):
        # Obtener la URL del campo de texto
        return self.url_input.text()
