from PyQt5.QtWidgets import QWidget, QDialog

from view.dialogs.dialogAddDataset import DialogAddDataset
from controller.dataset_controller import DatasetController

from core.utils import show_error

class ViewDatasets(QWidget):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.global_dataset_id = 1
        self.datasetController = DatasetController()
        self.init_ui()

    def init_ui(self):
        self.ui.datasets_btn_2.setChecked(True)
        self.ui.add_dataset.clicked.connect(self.on_add_dataset_click)
        self.ui.next_dataset.clicked.connect(self.on_next_dataset_click)


    def on_add_dataset_click(self):
        dialog = DialogAddDataset(self.ui, parent=self , datasetController=self.datasetController)
        
        # Mostrar el diálogo de forma modal
        dialog.exec_() == QDialog.Accepted

    def on_next_dataset_click(self):     
        self.check_info_dataset()


    def check_info_dataset(self):
        if self.validar_datos(): 
            self.ui.val_btn_2.setDisabled(False)
            self.ui.val_btn_1.setDisabled(False)

            # Para cambiar a la página de los modelos
            self.ui.val_btn_2.setChecked(True)

    def datasets_iguales(d1, d2):
        # Comparar id (opcional, porque suele ser único, podrías omitirlo)
        # if d1.id != d2.id:
        #     return False

        # Comparar nombre
        if d1.name != d2.name:
            return False

        # Comparar path
        if d1.path != d2.path:
            return False

        # Comparar dataframes
        if not d1.data.equals(d2.data):
            return False

        # Comparar listas x_col (mismo orden y contenido)
        if d1.x_col != d2.x_col:
            return False

        # Comparar y_col
        if d1.y_col != d2.y_col:
            return False

        # Si pasa todas las comparaciones, son iguales
        return True

    def validar_datos(self): 
        contenido = self.ui.scrollArea_3.widget()
        datos = self.datasetController.get_data()
        checkboxes_vacios = False
        if contenido is not None and contenido.findChildren(QWidget):
            # Verificar que todos tengan al menos una x_col
            for dataset_id, dataset_info in datos.items():
                if not any(dataset_info.x_col.values()):
                    checkboxes_vacios = True
                    break
            
            if checkboxes_vacios:
                show_error("Para poder continuar, debe seleccionar al menos una opción de x_col en cada dataset.")
                return False
            
            # --- Detectar datasets duplicados ---
            from itertools import combinations

            def datasets_iguales(d1, d2):
                if d1.name == d2.name:
                    return True
                """
                if not d1.data.equals(d2.data):
                    return False
                if d1.x_col != d2.x_col:
                    return False
                if d1.y_col != d2.y_col:
                    return False
                """
                return False

            ids = list(datos.keys())
            for id1, id2 in combinations(ids, 2):
                if datasets_iguales(datos[id1], datos[id2]):
                    show_error(f"¡Atención! Los datasets {id1} y {id2} son tienen el mismo nombre. Por favor elimina uno de ellos.")
                    return False

            self.check_ListAndTabla()
            
            return True
        else:
            show_error("Para poder continuar, debe agregar algún dataset para que los modelos se puedan entrenar.")
            return False


    def check_ListAndTabla(self):
        datos = self.datasetController.get_data()
        names = [dataset.name for dataset in datos.values()]

        # Obtener elementos actuales del listWidget
        existing_list_items = [self.ui.listWidget_datasets.item(i).text() for i in range(self.ui.listWidget_datasets.count())]

        # Obtener todos los textos de la tabla y sus posiciones
        existing_table_items = {}
        for row in range(self.ui.tableWidget_TVT.rowCount()):
            for col in range(self.ui.tableWidget_TVT.columnCount()):
                item = self.ui.tableWidget_TVT.item(row, col)
                if item is not None:
                    existing_table_items[(row, col)] = item.text()

        # Crear conjunto de elementos que no deben duplicarse
        excluded_items = set(existing_list_items + list(existing_table_items.values()))

        # Agregar nuevos datasets (que no están en la lista ni en la tabla)
        for name in names:
            if name not in excluded_items:
                self.ui.listWidget_datasets.addItem(name)

        # Eliminar del listWidget los que ya no están en datos
        for i in reversed(range(self.ui.listWidget_datasets.count())):
            item_text = self.ui.listWidget_datasets.item(i).text()
            if item_text not in names:
                self.ui.listWidget_datasets.takeItem(i)

        # Eliminar de la tabla los valores que ya no están en datos
        for (row, col), text in existing_table_items.items():
            if text not in names:
                self.ui.tableWidget_TVT.setItem(row, col, None)