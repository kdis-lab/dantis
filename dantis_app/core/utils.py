
from PyQt5.QtWidgets import QMessageBox, QFileDialog
import inspect
import numpy as np

def show_error(string: str):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(string)
        msg.setWindowTitle("Error")
        msg.exec_()

# Aplicar la función serializable a todo el diccionario
def serialize_hyperparameters(hyperparameters):
    for model_key in hyperparameters:
        if isinstance(hyperparameters[model_key], dict):  # Solo aplicar a diccionarios
            hyperparameters[model_key] = {
                k: serialize_value(v) for k, v in hyperparameters[model_key].items()
            }
        else:  # Si no es un diccionario, lo dejamos tal cual
            hyperparameters[model_key] = serialize_value(hyperparameters[model_key])
    return hyperparameters


# Función para deserializar valores guardados
def deserialize_value(value):
    if value == "empty":
        return inspect._empty
    if isinstance(value, dict) and "model_name" in value:  # Si es un modelo serializado
        try:
            module = __import__(value["module"], fromlist=[value["model_name"]])
            return getattr(module, value["model_name"])()  # Instanciamos el modelo
        except (ImportError, AttributeError):
            return value  # Si no se puede reconstruir, lo dejamos en su forma serializada
    return value


# Función para convertir valores no serializables
def serialize_value(value):
    """Convierte objetos en un formato serializable"""
    if value is inspect._empty:
        return "empty"
    elif isinstance(value, type):
        return value.__name__
    elif callable(value):
        return value.__name__
    elif hasattr(value, "__class__") and hasattr(value, "get_params"):
        # Si el objeto tiene get_params() (modelo de sklearn o pyod)
        return {
            "__model__": value.__class__.__name__,  # Nombre del modelo (ej. "LOF")
            "__module__": value.__class__.__module__,  # Módulo donde está (ej. "pyod.models.lof")
            "params": {k: serialize_value(v) for k, v in value.get_params().items()}
        }
    elif isinstance(value, list):
        return [serialize_value(v) for v in value]
    elif isinstance(value, tuple):
        return [serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif isinstance(value, np.ndarray):
        return value.tolist()
    else:
        return value
    

"""
def download_data_click(self): 
    texto_entrenamiento = self.ui.text_generate_results.toPlainText()
    texto_test = self.ui.test_generate_tests.toPlainText()

    # Abrir diálogo para seleccionar archivo a guardar
    options = QFileDialog.Options()
    archivo_txt, _ = QFileDialog.getSaveFileName(self, "Guardar archivo", "", "Archivos de texto (*.txt);;Todos los archivos (*)", options=options)

    if archivo_txt:
            with open(archivo_txt, mode='w', encoding='utf-8') as archivo:
                archivo.write("INFORME DEL ENTRENAMIENTO DE MODELOS Y TEST ESTADÍSTICOS EJECUTADOS\n")
                archivo.write("\n=== RESULTADOS ENTRENAMIENTO DE MODELOS ===\n")
                archivo.write(texto_entrenamiento + "\n")
                archivo.write("\n\n")  # Línea en blanco entre secciones
                archivo.write("\n=== RESULTADOS DE LOS TEST ESTADÍSTICOS ===\n")
                archivo.write(texto_test + "\n")
"""
    
"""
def mostrar_datos(self, datos, x_col, y_col, hiperparametros, checkboxes_marcados, validacion):
    
    self.ui.text_generate_results.clear()  # Borra todo primero (opcional)
    self.ui.text_generate_results.append("DATOS DEL ANÁLISIS: \n")
    self.ui.text_generate_results.append("→ Datasets usados: " + str(datos) + "\n")
    self.ui.text_generate_results.append("→ X_col: " + str(x_col) + "\n")
    self.ui.text_generate_results.append("→ y_col: " + str(y_col) + "\n") 
    self.ui.text_generate_results.append("→ Modelos a entrenar: ")
    for config in hiperparametros.values():
        nombre_modelo = config.get('modelo', 'No definido')
        hyperparams = config.get('hyperparameters', {})

        self.ui.text_generate_results.append(f"Modelo: {nombre_modelo}")
        self.ui.text_generate_results.append("Hiperparámetros:")

        # Recorremos los hiperparámetros para mostrarlos
        for param, valor in hyperparams.items():
            self.ui.text_generate_results.append(f"   • {param}: {valor}")

        self.ui.text_generate_results.append("")
    metricas = str(checkboxes_marcados)
    self.ui.text_generate_results.append("→ Métrica/s a estudiar: " + metricas + "\n")
    self.ui.text_generate_results.append("→ Validación: " + str(validacion) + "\n")
"""