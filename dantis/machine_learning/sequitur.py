from collections import Counter, defaultdict, deque
from itertools import count


class Symbol:
    """Symbol

    Inicializa un nuevo símbolo. Si es no terminal, incrementa el recuento de referencias
    de la regla correspondiente.

    Estrechamente acoplado con Rule. No diseñado para extensibilidad.

    """

    # pylint: disable=protected-access,unidiomatic-typecheck

    def __init__(self, value, bigrams: dict):
        self.bigrams = bigrams
        self.next_symbol = None
        self.prev_symbol = None
        self.value = value
        if type(value) is Rule:
            rule: Rule = value
            rule.value += 1

    def append(self, value):
        """Inserta un valor después de este."""
        symbol = Symbol(value, self.bigrams)
        symbol.join(self.next_symbol)
        self.join(symbol)

    def join(self, right):
        """Enlaza dos símbolos, eliminando cualquier bigrama antiguo de la tabla hash."""

        if self.next_symbol is not None:
            self._remove_bigram()

            # Esto es para manejar trigramas, donde solo registramos el segundo par de los bigramas superpuestos.
            # Cuando eliminamos el segundo par, insertamos el primer par en la tabla hash para no olvidarlo.

            if (
                right.prev_symbol is not None
                and right.next_symbol is not None
                and type(right) is type(right.prev_symbol)
                and right.value == right.prev_symbol.value
                and type(right) is type(right.next_symbol)
                and right.value == right.next_symbol.value
            ):
                self.bigrams[right._bigram()] = right

            if (
                self.prev_symbol is not None
                and self.next_symbol is not None
                and type(self) is type(self.prev_symbol)
                and self.value == self.prev_symbol.value
                and type(self) is type(self.next_symbol)
                and self.value == self.next_symbol.value
            ):
                self.bigrams[self.prev_symbol._bigram()] = self.prev_symbol

        self.next_symbol = right
        right.prev_symbol = self

    def _remove_bigram(self):
        """Elimina el bigrama de la tabla hash."""
        bigram = self._bigram()
        if self.bigrams.get(bigram) is self:
            del self.bigrams[bigram]

    def check(self):
        """Verifica un nuevo bigrama. Si aparece en otra parte, lo maneja llamando a match(),
        de lo contrario lo inserta en la tabla hash."""

        if type(self) is Rule or type(self.next_symbol) is Rule:
            return False
        bigram = self._bigram()
        match: Symbol = self.bigrams.get(bigram)
        if match is None:
            self.bigrams[bigram] = self
            return False
        if match.next_symbol is not self:
            self._process_match(match)
        return True

    def _process_match(self, match):
        """Procesa el emparejamiento reutilizando una regla existente o creando una nueva.

        También verifica una regla infrautilizada."""

        if (
            type(match.prev_symbol) is Rule
            and type(match.next_symbol.next_symbol) is Rule
        ):
            # Reutiliza una regla existente.
            rule: Rule = match.prev_symbol
            self._substitute(rule)
        else:
            # Crea una nueva regla.
            rule = Rule(0, self.bigrams)
            rule.join(rule)
            rule.prev_symbol.append(self.value)
            rule.prev_symbol.append(self.next_symbol.value)
            match._substitute(rule)
            self._substitute(rule)
            self.bigrams[rule.next_symbol._bigram()] = rule.next_symbol
        # Verifica una regla infrautilizada
        if type(rule.next_symbol.value) is Rule:
            target_rule: Rule = rule.next_symbol.value
            if target_rule.value == 1:
                rule.next_symbol._expand()

    def _substitute(self, rule):
        """Sustituye el símbolo y el anterior por la regla dada."""
        prev = self.prev_symbol
        prev.next_symbol._delete()
        prev.next_symbol._delete()
        prev.append(rule)
        if not prev.check():
            prev.next_symbol.check()

    def _delete(self):
        """Limpia para la eliminación del símbolo: elimina la entrada de la tabla hash y
        decrementa el recuento de referencias de la regla."""

        self.prev_symbol.join(self.next_symbol)
        self._remove_bigram()
        if type(self.value) is Rule:
            rule: Rule = self.value
            rule.value -= 1

    def _expand(self):
        """Este símbolo es la última referencia a su regla. Se elimina y el contenido de la regla
        se sustituye en su lugar."""

        left = self.prev_symbol
        right = self.next_symbol
        value: Rule = self.value
        first = value.next_symbol
        last = value.prev_symbol
        self._remove_bigram()
        left.join(first)
        last.join(right)
        self.bigrams[last._bigram()] = last

    def _bigram(self):
        """Tupla de bigrama del valor de self y el valor del siguiente símbolo."""
        return (
            type(self),
            self.value,
            type(self.next_symbol),
            self.next_symbol.value,
        )


class Rule(Symbol):
    """Rule

    El nodo de regla es la lista enlazada de símbolos que componen la regla. Apunta
    hacia adelante al primer símbolo en la regla y hacia atrás al último símbolo en la regla.
    Su propio valor es un recuento de referencias que registra la utilidad de la regla.

    Estrechamente acoplado con Symbol. No diseñado para extensibilidad.

    """


class Parser:
    """Parser para árboles de parseo de Sequitur."""

    def __init__(self):
        self._bigrams = {}
        rule = Rule(0, self._bigrams)
        rule.join(rule)
        self._tree = rule

    @property
    def tree(self):
        """Raíz del árbol de parseo."""
        return self._tree

    @property
    def bigrams(self):
        """Bigramas del parser."""
        return self._bigrams

    def feed(self, iterable):
        """Alimenta el iterable al parser.

        Itera sobre los elementos en iterable y construye el árbol de parseo.

        """
        tree: Rule = self._tree
        for value in iterable:
            tree.prev_symbol.append(value)
            tree.prev_symbol.prev_symbol.check()


class Mark:
    """Token de marca utilizado para evitar coincidencias de bigramas."""

    # pylint: disable=too-few-public-methods

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        name = type(self).__name__
        items = vars(self).items()
        args = ', '.join(f'{key}={value!r}' for key, value in items)
        return f'{name}({args})'

    def __str__(self):
        return '|'


class Production(int):
    """Producción"""

    def __repr__(self):
        num = super().__repr__()
        return f'Production({num})'

    def __str__(self):
        return super().__repr__()


class Grammar(dict):
    """Convierte la regla inicial del árbol de parseo en gramática."""

    # pylint: disable=unidiomatic-typecheck
    value_map = {
        ' ': '_',
        '\n': chr(0x21B5),
        '\t': chr(0x21E5),
    }

    def __init__(self, tree):
        super().__init__()
        counter = count()
        rule_to_production = defaultdict(lambda: Production(next(counter)))
        self._tree = rule_to_production[tree]
        rules = deque([tree])
        while rules:
            rule = rules.popleft()
            production = rule_to_production[rule]
            if production in self:
                continue  # Ya visitado.
            symbol = rule.next_symbol
            values = []
            while type(symbol) is not Rule:
                value = symbol.value
                if type(value) is Rule:
                    rules.append(value)
                    value = rule_to_production[value]
                values.append(value)
                symbol = symbol.next_symbol
            self[production] = values

    def lengths(self):
        """Devuelve las longitudes de las producciones."""
        _lengths = {}
        stack = [(self._tree, [])]

        while stack:
            production, path = stack.pop()
            if production in _lengths:
                continue
            length = 0
            new_stack = []
            for value in self[production]:
                if isinstance(value, Production):
                    if value in _lengths:
                        length += _lengths[value]
                    else:
                        new_stack.append((value, path))
                else:
                    length += 1
            if new_stack:
                stack.append((production, path))
                stack.extend(new_stack)
            else:
                _lengths[production] = length

        return Counter(_lengths)

    def counts(self):
        """Devuelve los recuentos de las producciones."""
        _counts = Counter(
            value
            for values in self.values()
            for value in values
            if isinstance(value, Production)
        )
        _counts[self._tree] = 1
        return _counts

    def depths(self):
        """Devuelve la profundidad mínima de cada producción."""
        _depths = defaultdict(lambda: float('inf'))
        stack = [(self._tree, 0)]

        while stack:
            production, depth = stack.pop()
            if depth < _depths[production]:
                _depths[production] = depth
                for value in self[production]:
                    if isinstance(value, Production):
                        stack.append((value, depth + 1))

        return _depths

    def expansions(self):
        """Devuelve las expansiones de las producciones."""
        _expansions = {}
        stack = [self._tree]
        visited = set()

        while stack:
            production = stack[-1]
            if production in _expansions:
                stack.pop()
                continue
            if production in visited:
                # Todas las dependencias han sido procesadas
                expansion = []
                for value in self[production]:
                    if isinstance(value, Production):
                        expansion.extend(_expansions[value])
                    else:
                        expansion.append(value)
                _expansions[production] = expansion
                stack.pop()
                continue
            visited.add(production)
            # Agregar dependencias
            for value in self[production]:
                if isinstance(value, Production) and value not in _expansions:
                    stack.append(value)

        return _expansions

    def expand(self, production):
        """Generador para expandir una producción."""
        stack = [iter(self[production])]
        while stack:
            try:
                value = next(stack[-1])
                if isinstance(value, Production):
                    stack.append(iter(self[value]))
                else:
                    yield value
            except StopIteration:
                stack.pop()

    def __str__(self):
        expansions = self.expansions()
        value_map = self.value_map
        lines = []
        for production, values in sorted(self.items()):
            parts = [production, '->']
            parts.extend(value_map.get(value, value) for value in values)
            prefix = ' '.join(map(str, parts))
            if production == 0:
                lines.append(prefix)
                continue
            space = ' ' * max(1, 50 - len(prefix))
            expansion = expansions[production]
            parts = (value_map.get(value, value) for value in expansion)
            suffix = ''.join(map(str, parts))
            triple = prefix, space, suffix
            line = ''.join(triple)
            lines.append(line)
        return '\n'.join(lines)


def parse(iterable):
    """Parsea el iterable y devuelve la gramática."""
    parser = Parser()
    parser.feed(iterable)
    grammar = Grammar(parser.tree)
    return grammar