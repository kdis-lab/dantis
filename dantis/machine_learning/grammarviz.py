"""
GrammarViz Anomaly Detector for DANTIS.

Este algoritmo adapta la técnica GrammarViz (originalmente para Motif Discovery)
para la detección de anomalías.

Lógica:
1. Discretización: La serie temporal se convierte en una cadena de símbolos (palabras SAX)
   usando una ventana deslizante.
2. Inferencia Gramatical: Se utiliza el algoritmo Sequitur para inferir una gramática
   libre de contexto que comprime la secuencia de palabras SAX.
3. Detección de Anomalías: Se asume que las anomalías corresponden a subsecuencias
   que no pueden ser comprimidas eficientemente (no forman parte de reglas frecuentes).
   
   El 'Anomaly Score' se calcula como la inversa de la frecuencia máxima de la regla
   que cubre cada punto temporal. Si un punto es parte de un patrón repetido 100 veces,
   su score es bajo. Si es único (frecuencia 1), su score es máximo.

Referencias:
    Senin, P., et al. (2014). GrammarViz 2.0: a tool for grammar-based pattern discovery in time series.
    Li, G., et al. (2012). Grammar-based compression of DNA sequences.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, Dict, Any, List
from scipy.stats import norm

# Importaciones relativas a dantis
from .. import algorithmbase
from .. import utils

from .sequitur import parse as sequitur_parse


def _find_sublist_indices(sequence: List[str], pattern: List[str]) -> List[int]:
    """Return start indices where pattern occurs in sequence using KMP-like prefix function.

    This is O(n + m) and operates on lists of strings.
    """
    if not pattern or not sequence or len(pattern) > len(sequence):
        return []

    # Build prefix (lps) array for pattern
    lps = [0] * len(pattern)
    length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    # Search
    res = []
    i = 0  # index for sequence
    j = 0  # index for pattern
    while i < len(sequence):
        if sequence[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                res.append(i - j)
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return res

def sax_transform(
    series: np.ndarray,
    segments: int,
    alphabet_size: int,
) -> np.ndarray:
    """
    Discretize a time series using SAX.

    Parameters
    ----------
    series : np.ndarray
        1D time series.
    segments : int
        Number of PAA segments (w).
    alphabet_size : int
        Number of symbols in the alphabet (a).
    return_symbols : bool
        If True, returns letters (a, b, c...); else integers.
    backend : {'aeon', 'native'}
        Implementation backend.

    Returns
    -------
    np.ndarray
        Symbolic word (array of str or int).
    """
    def generate_breakpoints(alphabet_size: int) -> np.ndarray:
        """
        Gaussian breakpoints for SAX: split N(0,1) into equiprobable bins.
        """
        if alphabet_size < 2:
            raise ValueError("alphabet_size must be >= 2")
        # quantiles at 1/a, 2/a, ..., (a-1)/a
        qs = np.linspace(1/alphabet_size, 1 - 1/alphabet_size, alphabet_size - 1)
        return norm.ppf(qs)
    # TODO add support to expand the ts to the original length

    sequence = (series - np.mean(series)) / (np.std(series) + 1e-6)
    n = len(sequence)
    # Robust PAA: split into (almost) equal sized blocks even when n < segments
    if segments <= 0:
        raise ValueError("segments must be positive")
    # Use array_split to handle cases where n < segments or n % segments != 0
    blocks = np.array_split(sequence, segments)
    paa = np.array([np.mean(b) if b.size > 0 else 0.0 for b in blocks])
    breakpoints = generate_breakpoints(alphabet_size)
    symbols = np.array([str(i) for i in np.digitize(paa, breakpoints)])
    return symbols


class GrammarViz(algorithmbase.AlgorithmBase):
    """
    GrammarViz Anomaly Detector.
    Detecta anomalías basándose en la densidad de reglas gramaticales (Sequitur).
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        """
        Retorna los hiperparámetros por defecto.

        - window_size: Tamaño de la ventana deslizante para generar palabras SAX.
        - paa_size: Número de segmentos PAA (debe ser divisor de window_size idealmente, o menor).
          Define la resolución de la palabra SAX.
        - alphabet_size: Tamaño del alfabeto SAX (número de símbolos distintos).
        - contamination: Proporción de anomalías esperada.
        - sax_backend: 'native' (numpy puro) o 'aeon' (si está instalado, más rápido).
        """
        return {
            "window_size": 100,
            "paa_size": 4,
            "alphabet_size": 4,
            "contamination": 0.05,
            "sax_backend": "native",
            "verbose": False
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        self._sax_words = None
        self._offsets = None

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        if hyperparameter is None:
            self._hyperparameter = defaults.copy()
            return

        def _hp_template(window_size=100, paa_size=4, alphabet_size=4, 
                         contamination=0.05, sax_backend="native", verbose=False):
            return None

        pm = utils.ParameterManagement(_hp_template)
        try:
            coerced = pm.check_hyperparameter_type(hyperparameter)
        except Exception as e:
            raise ValueError(f"GrammarViz Hyperparameter Error: {e}")

        merged = pm.complete_parameters(coerced)

        # Validaciones lógicas adicionales
        if merged["paa_size"] > merged["window_size"]:
            merged["paa_size"] = merged["window_size"]

        if int(merged.get('alphabet_size', 2)) < 2:
            raise ValueError('alphabet_size must be >= 2')

        # Add numerosity_reduction default if missing
        if 'numerosity_reduction' not in merged:
            merged['numerosity_reduction'] = False

        for k, v in hyperparameter.items():
            if k not in merged:
                merged[k] = v

        self._hyperparameter = merged

    def _validate_input(self, x) -> np.ndarray:
        """Normaliza la entrada a array numpy 1D float."""
        if isinstance(x, pd.DataFrame):
            if x.shape[1] >= 2:
                val = x.iloc[:, 1].values
            else:
                val = x.iloc[:, 0].values
        elif isinstance(x, pd.Series):
            val = x.values
        else:
            val = np.array(x)
        
        if val.ndim > 1:
            val = val.flatten()
            
        return val.astype(float)

    def _sliding_sax_words(self, ts: np.ndarray, window_size: int, paa_size: int, 
                          alphabet_size: int, backend: str):
        """
        Genera palabras SAX mediante ventana deslizante.
        Optimización: Solo guarda palabras nuevas si cambian respecto a la anterior (Run Length Encoding implícito)
        o guarda todas. Para Sequitur estándar, pasamos todas las palabras.
        """
        words = []
        # offsets guarda el índice real de inicio de cada palabra en la lista 'words'
        offsets = [] 
        
        # Validación de backend
        if backend == "aeon":
            try:
                import aeon
            except ImportError:
                backend = "native"

        # Iteración deslizante
        # Nota: Esto puede ser lento en Python puro para series muy largas (>100k).
        # Para producción real, se recomienda vectorizar PAA/SAX o usar implementación compilada.
        
        limit = len(ts) - window_size + 1
        
        # Pre-cálculo para backend nativo (más rápido que llamar a sax_transform en bucle)
        # Sin embargo, usamos la función auxiliar proporcionada para consistencia.
        
        last_word = None
        numerosity = bool(self._hyperparameter.get('numerosity_reduction', False))
        for i in range(limit):
            subseq = ts[i : i + window_size]
            
            # Manejo de subsucesiones constantes (std=0)
            if np.std(subseq) < 1e-6:
                # Asignar palabra '0'*paa_size o similar para zonas planas
                word_list = ['0'] * paa_size
                word = "".join(word_list)
            else:
                word_raw = sax_transform(
                    subseq,
                    segments=paa_size,
                    alphabet_size=alphabet_size,
                )
                word = "".join(word_raw)
            
            # Numerosity Reduction: GrammarViz a veces colapsa palabras idénticas consecutivas.
            # Aquí las mantenemos todas para que el mapeo de índices sea directo, 
            # o las colapsamos y guardamos el offset.
            # Sequitur maneja bien la repetición, así que le pasamos el flujo comprimido.
            
            if numerosity:
                if word != last_word:
                    words.append(word)
                    offsets.append(i)
                    last_word = word
            else:
                words.append(word)
                offsets.append(i)
                
        return words, offsets

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        GrammarViz suele usarse de forma no supervisada sobre el dataset de test.
        Fit aquí es passthrough.
        """
        self._validate_input(x_train)
        self._model = "GrammarViz_Ready"
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula el anomaly score basado en la densidad de reglas.
        Score[t] = 1.0 / (Frecuencia de la regla que cubre t)
        """
        series = self._validate_input(x)
        params = self._hyperparameter
        
        w_size = int(params["window_size"])
        paa_size = int(params["paa_size"])
        a_size = int(params["alphabet_size"])
        backend = str(params["sax_backend"])

        if len(series) < w_size:
            return np.zeros(len(series))

        # 1. Transformación SAX
        # Obtenemos la secuencia de palabras discretas
        sax_words, offsets = self._sliding_sax_words(series, w_size, paa_size, a_size, backend)
        
        if not sax_words:
            return np.zeros(len(series))

        # 2. Inferencia Gramatical (Sequitur)
        # grammar es un objeto que contiene las reglas producidas
        grammar = sequitur_parse(sax_words)
        
        # 3. Mapeo de Densidad
        # Inicializamos el array de "cobertura máxima" con 1 (frecuencia mínima posible)
        # covering_freq[t] almacenará la frecuencia de la regla más frecuente que abarca el instante t
        covering_freq = np.ones(len(series), dtype=int)
        
        # Obtenemos expansiones y conteos
        # expansion: Dict[RuleID, List[Symbols]] (qué palabras SAX componen la regla)
        expansions = grammar.expansions()
        
        # Iteramos sobre la secuencia original de palabras SAX para mapear reglas a tiempo
        # Esto puede ser complejo porque Sequitur es jerárquico.
        # Simplificación robusta:
        # Buscamos ocurrencias de cada regla expandida en la secuencia de palabras SAX.
        
        # Primero, calculamos cuántas veces aparece cada regla en la gramática
        # counts() devuelve cuántas veces se usa una regla en otras reglas, 
        # pero necesitamos saber cuántas veces aparece la expansión completa en la serie original.
        
        # Recorremos la secuencia de palabras SAX original y vemos qué reglas se activan.
        # Alternativa más eficiente: Usamos la lógica de 'Motif Discovery' proporcionada.
        
        rule_occurrences = defaultdict(list)
        
        # Fuerza bruta optimizada: barrido lineal sobre palabras SAX buscando matches de reglas
        # Solo nos interesan reglas que expanden a más de 1 palabra o reglas que aparecen muchas veces.
        
        full_sax_sequence = list(sax_words) # Lista de strings
        
        for rule_id, expanded_sequence in expansions.items():
            # expanded_sequence es una lista de palabras SAX ['abc', 'abd'...]
            rule_len_words = len(expanded_sequence)
            
            # Buscar dónde aparece esta secuencia en sax_words
            # Nota: Esto es O(N*M) en el peor caso. Sequitur optimizado lo haría mejor,
            # pero usamos las herramientas proporcionadas.
            
            # Para optimizar, asumimos que si Sequitur creó una regla, es porque se repite.
            # En una implementación real de GrammarViz, el árbol de Sequitur ya tiene los índices.
            # Aquí reconstruimos los índices haciendo pattern matching simple sobre la lista de palabras.
            
            # Truco para velocidad: convertir listas a tuplas para hash
            pat = tuple(expanded_sequence)
            
            # Contamos frecuencia real en la serie
            # (Sequitur garantiza que R1 sustituye todas las ocurrencias no solapadas, 
            # pero necesitamos todas las ocurrencias físicas)
            
            # Use efficient sublist search (KMP-like) to find start indices
            try:
                matches = _find_sublist_indices(full_sax_sequence, list(expanded_sequence))
            except Exception:
                matches = []
            
            freq = len(matches)
            if freq < 2: continue # No es un patrón frecuente
            
            # Actualizar covering_freq
            for match_idx in matches:
                # offset real en la serie temporal
                if match_idx < len(offsets):
                    start_time = offsets[match_idx]
                    # El patrón dura 'rule_len_words' ventanas SAX
                    # Pero cuidado: las ventanas se solapan.
                    # El final del patrón es el inicio de la última palabra + w_size
                    end_match_idx = match_idx + rule_len_words - 1
                    if end_match_idx < len(offsets):
                        end_time = offsets[end_match_idx] + w_size
                        
                        # Actualizamos la frecuencia de cobertura en este intervalo
                        # Usamos max() porque queremos saber si el punto pertenece a AL MENOS un patrón muy fuerte
                        current_slice = covering_freq[start_time:end_time]
                        if current_slice.size > 0:
                            np.maximum(current_slice, freq, out=current_slice)

        # 4. Calcular Score
        # Score = 1 / covering_freq
        # Si un punto nunca fue cubierto por una regla (freq=1), score = 1.0 (Anomalía máxima)
        # Si fue cubierto por una regla que sale 100 veces, score = 0.01 (Normal)
        
        # Suavizado logarítmico para mejorar visualización
        # scores = 1.0 / np.log1p(covering_freq) # Alternativa
        scores = 1.0 / covering_freq
        
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Etiquetas binarias.
        """
        scores = self.decision_function(x)
        contamination = float(self._hyperparameter.get("contamination", 0.05))
        contamination = max(0.001, min(0.5, contamination))
        
        threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
        labels = (scores >= threshold).astype(int)
        return labels

    def save_model(self, path: str):
        """Persist grammar artifacts (grammar, offsets, hyperparameters) using joblib."""
        try:
            import joblib
        except Exception:
            raise ImportError('save_model requires joblib')

        payload = {
            'hyperparameter': self._hyperparameter,
            'grammar': getattr(self, '_grammar', None),
            'offsets': getattr(self, '_offsets', None),
        }
        joblib.dump(payload, path)

    def load_model(self, path: str):
        try:
            import joblib
        except Exception:
            raise ImportError('load_model requires joblib')
        payload = joblib.load(path)
        self._hyperparameter = payload.get('hyperparameter', self.get_default_hyperparameters())
        self._grammar = payload.get('grammar')
        self._offsets = payload.get('offsets')
        return self