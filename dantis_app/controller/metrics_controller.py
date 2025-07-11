

class MetricsController:
    def __init__(self):
        self.checkboxes = []

    def setChecBoxesSelected(self, checboxes): 
        self.checkboxes = checboxes

    def getChecBoxes_selected (self):
        return self.checkboxes

    """
    def simular_valor_metrica(self, nombre):
        if nombre in ['True Positive', 'True Negative', 'False positive', 'False negative']:
            return random.randint(50, 500)
        elif nombre == 'Accuracy':
            return round(random.uniform(0.7, 0.99), 4)
        elif nombre == 'Recall / Sensibilidad':
            return round(random.uniform(0.5, 1.0), 4)
        elif nombre == 'F1 Score':
            return round(random.uniform(0.5, 1.0), 4)
        elif nombre == 'Log-Loss':
            return round(random.uniform(0.1, 1.0), 4)
        elif nombre == 'Curva ROC / AUC':
            return round(random.uniform(0.6, 1.0), 4)
        elif nombre == 'Error Absoulto Medio (MAE)':
            return round(random.uniform(0.1, 2.0), 4)
        elif nombre == 'Error Cuadrático Medio (MSE)':
            return round(random.uniform(0.2, 5.0), 4)
        elif nombre == 'Raíz del Error Cuadrático Medio (RMSE)':
            return round(random.uniform(0.4, 2.2), 4)
        elif nombre == 'Coeficiente de determinación':
            return round(random.uniform(0.5, 1.0), 4)
        else:
            return "N/A"
    """