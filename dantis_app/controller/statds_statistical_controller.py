import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Callable, Union, Any

from statds.normality import shapiro_wilk_normality, d_agostino_pearson, kolmogorov_smirnov
from statds.homoscedasticity import levene_test, bartlett_test
from statds.parametrics import t_test_paired, t_test_unpaired, anova_cases, anova_within_cases
from itertools import combinations
from statds.no_parametrics import (
    wilcoxon, binomial, mannwhitneyu,
    friedman, iman_davenport, friedman_aligned_ranks, quade, kruskal_wallis,
    nemenyi, bonferroni, li, holm, holland, finner, hochberg, hommel, rom, shaffer
)

class StatisticalTestController:
    def __init__(self):
        self.alpha_default = 0.05

        self.available_tests: Dict[str, Callable] = {
            # No paramétricos
            "Wilcoxon (Por Parejas)": wilcoxon,
            "Binomial Sign (Por Parejas)": binomial,
            "Mann-Whitney U (Por Parejas)": mannwhitneyu,
            "Friedman (Grupos)": friedman,
            "Friedman + Iman Davenport (Grupos)": iman_davenport,
            "Friedman Aligned Ranks (Grupos)": friedman_aligned_ranks,
            "Quade (Grupos)" : quade,

            # Paramétricos
            "T-Test paired (Por Parejas)": t_test_paired,
            "T-Test unpaired (Por Parejas)": t_test_unpaired,
            "ANOVA between cases (Grupos)": anova_cases,
            "ANOVA within cases (Grupos)": anova_within_cases,
        }
        self.available_post_hoc = {
            "Nemenyi": nemenyi,
            "Bonferroni": bonferroni,
            "Li": li,
            "Holm": holm,
            "Holland": holland,
            "Finner": finner,
            "Hochberg": hochberg,
            "Hommel": hommel,
            "Rom": rom,
            "Shaffer": shaffer,
        }

    def list_available_tests(self) -> Dict[str, str]:
        """Devuelve listado de tests disponibles y sus nombres de función."""
        return {name: func.__name__ for name, func in self.available_tests.items()}
    
    def list_available_post_hoc(self) -> Dict[str, str]:
        """Devuelve listado de tests disponibles y sus nombres de función."""
        return {name: func.__name__ for name, func in self.available_post_hoc.items()}

    def _encode_plot(self, fig) -> str:
        """Codifica figura matplotlib a base64."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return encoded

    def run_test(
        self,
        test_name: str,
        data: Union[pd.DataFrame, Dict],
        alpha: float = None,
        post_hoc_selected: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ejecuta un test y devuelve resultados como diccionario.

        Parameters
        ----------
        test_name : str
            Nombre del test según el listado disponible.
        data : pd.DataFrame o dict
            Datos a usar (deben tener formato válido).
        alpha : float, optional
            Nivel de significación (por defecto 0.05).
        kwargs : dict
            Argumentos adicionales (ej: minimize=True, num_cases=30).

        Returns
        -------
        dict
            Resultado del test, con posibles figuras codificadas.
        """
        alpha = alpha if alpha is not None else self.alpha_default

        if test_name not in self.available_tests:
            return {"error": f"Test '{test_name}' no está disponible."}

        # try:
        func = self.available_tests[test_name]

        column = "Datasets"
        columns = list(data.columns) if isinstance(data, pd.DataFrame) else list(data.keys())

        # Casos especiales: Friedman -> rankings + args adicionales
        if "(Grupos)" in test_name:
            minimize = kwargs.get("minimize", False)
            data_ = data.reset_index()
            data_ = data_[[column] + columns]
            rankings, stat, pval, crit_val, hyp = func(data_, alpha, minimize=minimize)

            if isinstance(rankings, pd.DataFrame):
                rankings_df = rankings
            elif isinstance(rankings, dict):
                rankings_df = pd.DataFrame([rankings])
            else:
                rankings_df = pd.DataFrame(rankings).T  # asume que rankings es array-like

            # Combina los resultados principales con los rankings
            result_df = pd.DataFrame({
                "Test": [test_name],
                "Statistic": [stat],
                "P-Value": [pval],
                "Critical Value": [crit_val],
                "Hypothesis": [hyp]
            })

            # Añade columnas de rankings al DataFrame de resultados
            for col in rankings_df.columns:
                result_df[col] = rankings_df.iloc[0][col] if len(rankings_df) == 1 else rankings_df[col].values

            if post_hoc_selected and ("Friedman" in test_name or "Quade" in test_name):
                if post_hoc_selected not in self.available_post_hoc:
                    return result_df, {"error": f"Post-hoc '{post_hoc_selected}' no está disponible."}
                post_hoc_func = self.available_post_hoc[post_hoc_selected]
                if post_hoc_selected == "Nemenyi":
                    results, fig = post_hoc_func(rankings, data.shape[0], alpha)
                else:
                    results, fig = post_hoc_func(rankings, data.shape[0], alpha, type_rank=test_name.replace(" (Grupos)", ""))
            
            return result_df
        
        # Pair comparative
        results = []
        for col1, col2 in combinations(columns, 2):
            pair_data = data[[col1, col2]]
            result = func(pair_data, alpha)
            if isinstance(result, tuple):
                keys = ["statistic", "pvalue", "rejected", "hypothesis"]
                result_dict = {k: [v] for k, v in zip(keys, result)}
                result_dict["Algorithms"] = [col1 + " vs " + col2]
                results.append(result_dict)

        results = pd.concat([pd.DataFrame(i) for i in results], ignore_index=True)
        results.set_index("Algorithms", inplace=True)
        return results