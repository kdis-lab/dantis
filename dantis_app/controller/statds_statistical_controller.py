import pandas as pd
import matplotlib.pyplot as plt
import io, base64
import inspect
from typing import Dict, Callable, Union, Any

from statds.parametrics import t_test_paired, t_test_unpaired, anova_cases, anova_within_cases
from itertools import combinations
from statds.no_parametrics import (
    wilcoxon, binomial, mannwhitneyu,
    friedman, iman_davenport, friedman_aligned_ranks, quade, kruskal_wallis,
    nemenyi, bonferroni, li, holm, holland, finner, hochberg, hommel, rom, shaffer
)
from statds.normality import shapiro_wilk_normality, d_agostino_pearson, kolmogorov_smirnov
from statds.homoscedasticity import levene_test, bartlett_test

class StatisticalTestController:
    """
    A controller class for executing both parametric and non-parametric statistical tests, 
    as well as post-hoc analyses, primarily used for comparing machine learning algorithms 
    across datasets.

    This class provides methods to:
    - List available statistical and post-hoc tests.
    - Run selected statistical tests on provided data.
    - Optionally run post-hoc comparisons and generate visualizations.
    - Encode result figures as base64 strings for web integration or reporting.

    Attributes
    ----------
    alpha_default : float
        Default significance level used in all tests if not explicitly provided (default is 0.05).

    available_tests : Dict[str, Callable]
        A dictionary mapping descriptive test names to their corresponding statistical test functions.
        Includes both parametric and non-parametric tests. These functions typically come from
        `statds.parametrics` and `statds.no_parametrics`.

    available_post_hoc : Dict[str, Callable]
        A dictionary of post-hoc test functions for group-based test correction procedures.
        These include methods such as Bonferroni, Holm, Nemenyi, etc.

    Methods
    -------
    list_available_tests() -> Dict[str, str]
        Returns the names and function references for the supported statistical tests.

    list_available_post_hoc() -> Dict[str, str]
        Returns the names and function references for the supported post-hoc tests.

    run_test(test_name, data, alpha=None, post_hoc_selected=None, **kwargs) -> Dict[str, Any]
        Executes the selected statistical test with the given data and parameters.

    _encode_plot(fig) -> str
        Encodes a matplotlib figure as a base64 PNG string.

    _handle_pairwise_test(func, data, alpha) -> pd.DataFrame
        Internal method for handling tests between pairs of algorithms.

    _handle_group_test(test_name, func, data, alpha, post_hoc_selected=None, **kwargs)
        Internal method for handling group-based statistical tests and optional post-hoc analysis.

    _format_group_test_result(test_name, rankings, stat, pval, crit_val, hyp) -> pd.DataFrame
        Formats group test results into a readable dataframe.
    """
    def __init__(self):
        """
        Initializes the controller with a predefined set of statistical tests and post-hoc methods.
        Sets the default alpha value to 0.05.
        """
        self.alpha_default = 0.05
        self.available_tests: Dict[str, Callable] = {
            # Non-parametrics
            "Wilcoxon (Por Parejas)": wilcoxon,
            "Binomial Sign (Por Parejas)": binomial,
            "Mann-Whitney U (Por Parejas)": mannwhitneyu,
            "Friedman (Grupos)": friedman,
            "Friedman + Iman Davenport (Grupos)": iman_davenport,
            "Friedman Aligned Ranks (Grupos)": friedman_aligned_ranks,
            "Quade (Grupos)" : quade,

            # Parametrics
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
        """
        List all available primary statistical tests.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping test display names to their function names.
        """
        return {name: func.__name__ for name, func in self.available_tests.items()}
    
    def list_available_post_hoc(self) -> Dict[str, str]:
        """
        List all available post-hoc tests.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping post-hoc display names to their function names.
        """
        return {name: func.__name__ for name, func in self.available_post_hoc.items()}

    def _encode_plot(self, fig) -> str:
        """
        Encode a Matplotlib figure into a base64 string.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to encode.

        Returns
        -------
        str
            A base64-encoded PNG image string of the figure.
        """
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
        Run a statistical test and return results.

        Parameters
        ----------
        test_name : str
            The name of the test to run (must match a key in `available_tests`).
        data : Union[pd.DataFrame, dict]
            The input data formatted for the specific test type.
        alpha : float, optional
            Significance level to use (default is 0.05).
        post_hoc_selected : str, optional
            Name of post-hoc test to run (only applies to group-based tests).
        **kwargs : dict
            Additional keyword arguments for the test function (e.g., minimize).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the result of the test or an error message.
        """
        alpha = alpha if alpha is not None else self.alpha_default

        if test_name not in self.available_tests:
            return {"error": f"Test '{test_name}' no está disponible."}

        func = self.available_tests[test_name]

        if "(Grupos)" in test_name:
            return self._handle_group_test(test_name, func, data, alpha, post_hoc_selected, **kwargs)

        return self._handle_pairwise_test(func, data, alpha)

    def _handle_pairwise_test(self, func: Callable, data: pd.DataFrame, alpha: float) -> pd.DataFrame:
        """
        Execute pairwise comparisons between all columns using the given test.

        Parameters
        ----------
        func : Callable
            The test function to apply to each column pair.
        data : pd.DataFrame
            A dataframe where each column is a different algorithm or condition.
        alpha : float
            The significance level.

        Returns
        -------
        pd.DataFrame
            Dataframe of test statistics, p-values, and hypothesis decisions for each pair.
        """
        columns = list(data.columns)
        results = []

        for col1, col2 in combinations(columns, 2):
            pair_data = data[[col1, col2]]
            pair_data = pair_data.apply(pd.to_numeric, errors="coerce").dropna()
            result = func(pair_data, alpha)

            if isinstance(result, tuple):
                keys = ["statistic", "pvalue", "rejected", "hypothesis"]
                result_dict = {k: [v] for k, v in zip(keys, result)}
                result_dict["Algorithms"] = [f"{col1} vs {col2}"]
                results.append(result_dict)

        df = pd.concat([pd.DataFrame(r) for r in results], ignore_index=True)
        df.set_index("Algorithms", inplace=True)
        return df

    def _handle_group_test(self, test_name: str, func: Callable, data: pd.DataFrame,
                           alpha: float, post_hoc_selected: str = None, **kwargs
                            ) -> Union[pd.DataFrame, list, dict]:
        """
        Execute a group-based statistical test and optionally apply a post-hoc test.

        Parameters
        ----------
        test_name : str
            Display name of the test being run.
        func : Callable
            The group test function.
        data : pd.DataFrame
            Input data, with rows as datasets and columns as algorithms.
        alpha : float
            Significance level.
        post_hoc_selected : str, optional
            Post-hoc test to apply (e.g., Nemenyi).
        **kwargs : dict
            Additional arguments like `minimize`.

        Returns
        -------
        Union[pd.DataFrame, list, dict]
            A result dataframe or a list including post-hoc test results and plots.
        """
        minimize = kwargs.get("minimize", False)
        data_ = data.reset_index()
        columns = list(data.columns) if isinstance(data, pd.DataFrame) else list(data.keys())
        data_ = data_[["Datasets"] + columns]
        data_subset = data_[columns].apply(pd.to_numeric, errors="coerce").dropna()
        
        if "anova" in func.__name__ :
            results = func(data_subset, alpha)

            if not isinstance(results, (list, tuple)):
                raise ValueError(f"anova_cases debe devolver una tupla/lista, pero devolvió {type(results)}")

            _, anova_results, stat, pval, crit_val, hyp = results
            result_df = self._format_group_test_result(test_name, None, stat, pval, crit_val, hyp)
            return [result_df, anova_results]
        else:
            params = inspect.signature(func).parameters
            if "minimize" in params:
                rankings, stat, pval, crit_val, hyp = func(data_subset, alpha, minimize=minimize)
            else:
                rankings, stat, pval, crit_val, hyp = func(data_subset, alpha)

        result_df = self._format_group_test_result(test_name, rankings, stat, pval, crit_val, hyp)

        if post_hoc_selected and ("Friedman" in test_name or "Quade" in test_name):
            if post_hoc_selected not in self.available_post_hoc:
                return result_df, {"error": f"Post-hoc '{post_hoc_selected}' no está disponible."}
            
            post_hoc_func = self.available_post_hoc[post_hoc_selected]

            if post_hoc_selected == "Nemenyi":
                ranks_values, cd, figure = post_hoc_func(rankings, data.shape[0], alpha)
                return [result_df, figure]
            else:
                posthoc_result, fig = post_hoc_func(rankings, data.shape[0], alpha, type_rank=test_name.replace(" (Grupos)", ""))
                return [result_df, posthoc_result, fig]

        return result_df

    def _format_group_test_result(self, test_name, rankings, stat, pval, crit_val, hyp):
        """
        Format the result of a group-based test into a structured DataFrame.

        Parameters
        ----------
        test_name : str
            Name of the test executed.
        rankings : Union[pd.DataFrame, dict, list]
            Ranking results from the test.
        stat : float
            Test statistic.
        pval : float
            P-value of the test.
        crit_val : float
            Critical value.
        hyp : str
            Interpretation of hypothesis testing.

        Returns
        -------
        pd.DataFrame
            A single-row dataframe summarizing test results and rankings.
        """
        if isinstance(rankings, pd.DataFrame):
            rankings_df = rankings
        elif isinstance(rankings, dict):
            rankings_df = pd.DataFrame([rankings])
        else:
            rankings_df = pd.DataFrame(rankings).T

        result_df = pd.DataFrame({
            "Test": [test_name],
            "Statistic": [stat],
            "P-Value": [pval],
            "Critical Value": [crit_val],
            "Hypothesis": [hyp]
        })

        for col in rankings_df.columns:
            result_df[col] = rankings_df.iloc[0][col] if len(rankings_df) == 1 else rankings_df[col].values

        return result_df