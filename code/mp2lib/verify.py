"""Exact-arithmetic checking of the calculations a candidate spells out.

A candidate solution often states its work as explicit identities, e.g.
``12 * 4 = 48``.  Recomputing such an identity with exact rational arithmetic
(:class:`fractions.Fraction`) gives direct evidence of broken arithmetic: the
solution states a number that its own operands do not produce.  A solution
that reads the *problem* wrongly while computing perfectly (the guide's bus
example) is of course invisible here — this only catches arithmetic faults.

How one line is read
--------------------
A line is split at each equals sign it contains.  On either side of the sign
the reader takes the longest run of characters that could belong to an
expression, moving outwards until it meets a letter, a second sign or any
other character that cannot appear in arithmetic.  That run is accepted only
if it is non-trivial, has balanced parentheses, ends on a number or a closing
bracket rather than a dangling operator, and — for the left side — actually
claims a calculation.  Anything doubtful is dropped: accusing a correct
solution is a worse error than overlooking a step.

The reader is a single pass over the line; :func:`audit` reports every
quantity the feature builder needs at once, so a solution is never parsed
twice for the same purpose.
"""

import re
from fractions import Fraction
from typing import NamedTuple

# How close two values must be for a step to be accepted, written as a
# numerator and a denominator so the test itself stays exact.  The band
# between 1/500 and 1/100 changes the reading of well under one percent of
# the stated steps, so this constant is not a sensitive choice.
_TOL_NUM, _TOL_DEN = 1, 500
_TOL = Fraction(_TOL_NUM, _TOL_DEN)

# A line is only worth reading if it is short enough to be a stated step
# rather than a paragraph that happens to contain an equals sign.
_MAX_LINE = 400

_FRAC_RE = re.compile(r'\\[a-zA-Z]*frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}')
_TEXT_RE = re.compile(r'\\(?:text|mathrm|mathbf|textbf|textit|operatorname)\{[^{}]*\}')
_OVERLINE_RE = re.compile(r'\\overline\s*\{([^{}]*)\}')
_PCT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*%')
_OF_RE = re.compile(r'\)\s*of\s*(\d+(?:\.\d+)?)')
_CMD_RE = re.compile(r'\\[a-zA-Z]+')
_WORDY_RE = re.compile(r'[0-9A-Za-z]*[A-Za-z][0-9A-Za-z]*')
_THOUSANDS_RE = re.compile(r'(?<=\d),(?=\d{3}(?!\d))')
_EQ_RE = re.compile(r'(?<![=!<>+\-*/])=(?![=])')
_NUM_RE = re.compile(r'\d+(?:\.\d+)?')

# Characters that may appear inside an arithmetic expression.
_EXPR_CHARS = frozenset('0123456789.+-*/() \t')

_UNICODE_OPS = (('−', '-'), ('–', '-'), ('—', '-'),
                ('×', '*'), ('÷', '/'), ('·', '*'))


# ---------------------------------------------------------------------------
# line preparation
# ---------------------------------------------------------------------------

def _clean(text):
    """Turn a solution into the list of word-free lines that state equations."""
    t = str(text)
    for a, b in _UNICODE_OPS:
        t = t.replace(a, b)
    t = t.replace('\\%', '%')
    for _ in range(4):                      # nested \frac need repeated passes
        new = _FRAC_RE.sub(lambda m: f'(({m.group(1)})/({m.group(2)}))', t)
        if new == t:
            break
        t = new
    t = _TEXT_RE.sub(' ', t)
    t = _OVERLINE_RE.sub(lambda m: m.group(1), t)
    t = t.replace('\\left', '').replace('\\right', '')
    t = t.replace('\\cdot', '*').replace('\\times', '*').replace('\\div', '/')
    t = _PCT_RE.sub(lambda m: f'(({m.group(1)})/100)', t)
    t = _OF_RE.sub(lambda m: f')*({m.group(1)})', t)
    t = _CMD_RE.sub(' ', t)
    t = t.replace('{', '(').replace('}', ')')
    t = t.replace('$', ' ').replace('#', ' ').replace('&', ' ')
    t = _THOUSANDS_RE.sub('', t)
    lines = []
    for line in t.splitlines():
        if '=' not in line or len(line) > _MAX_LINE:
            continue
        line = _WORDY_RE.sub(' ', line)     # units and prose leave gaps
        line = line.replace(',', ' ')       # remaining commas separate things
        if '=' in line:
            lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# reading one side of an equals sign
# ---------------------------------------------------------------------------

def _left_of(line, pos):
    """Longest expression run ending immediately before ``pos``."""
    i = pos - 1
    while i >= 0 and line[i] in _EXPR_CHARS:
        i -= 1
    return line[i + 1:pos].strip()


def _right_of(line, pos):
    """Longest expression run starting immediately after ``pos``."""
    i = pos
    n = len(line)
    while i < n and line[i] in _EXPR_CHARS:
        i += 1
    return line[pos:i].strip()


def _balanced(segment):
    depth = 0
    for ch in segment:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _complete(segment):
    """True when the run does not stop on a dangling operator."""
    return bool(segment) and (segment[-1].isdigit() or segment[-1] == ')')


def _claims_a_calculation(segment):
    """A left side counts only when it combines values with an operator."""
    return (sum(1 for ch in segment if ch in '+-*/') > 0
            and len(_NUM_RE.findall(segment)) >= 2)


def _no_bare_neighbours(segment):
    """Reject a run in which two numbers sit side by side with no operator.

    ``8 8 + 2`` is two numbers that happen to share a line, not one
    expression; reading it as ``88 + 2`` would invent a calculation the
    solution never made.  ``3(18)`` is fine because a bracket separates the
    two numbers.
    """
    spans = [m.span() for m in _NUM_RE.finditer(segment)]
    for (a_start, a_end), (b_start, _) in zip(spans, spans[1:]):
        if not segment[a_end:b_start].strip():
            return False
    return True


# ---------------------------------------------------------------------------
# recursive-descent parser
# ---------------------------------------------------------------------------

class _Parser:
    """Recursive descent over a whitespace-free expression string."""

    def __init__(self, s):
        self.s = s
        self.i = 0
        self.n = len(s)

    def _peek(self):
        return self.s[self.i] if self.i < self.n else ''

    def _take(self):
        ch = self.s[self.i]
        self.i += 1
        return ch

    def parse(self):
        v = self._sum()
        if self.i != self.n:
            raise ValueError('trailing characters')
        return v

    def _sum(self):
        v = self._product()
        while self._peek() in ('+', '-'):
            op = self._take()
            rhs = self._product()
            v = v + rhs if op == '+' else v - rhs
        return v

    def _product(self):
        v = self._unary()
        while True:
            op = self._peek()
            if op == '*':
                self._take()
                v = v * self._unary()
            elif op == '/':
                self._take()
                rhs = self._unary()
                if rhs == 0:
                    raise ZeroDivisionError('divide by zero')
                v = v / rhs
            elif op == '(':
                v = v * self._group()          # "3(18)" implicit product
            else:
                return v

    def _unary(self):
        op = self._peek()
        if op == '+':
            self._take()
            return self._unary()
        if op == '-':
            self._take()
            return -self._unary()
        return self._leaf()

    def _group(self):
        self._take()                           # consume '('
        v = self._sum()
        if self._peek() != ')':
            raise ValueError('unbalanced parentheses')
        self._take()
        return v

    def _leaf(self):
        ch = self._peek()
        if ch == '(':
            return self._group()
        if ch.isdigit():
            m = _NUM_RE.match(self.s, self.i)
            if m is None:
                raise ValueError('not a number')
            self.i = m.end()
            return Fraction(m.group())
        raise ValueError(f'unexpected character {ch!r}')


def _value(segment):
    """Exact value of an expression, or None when it does not read cleanly."""
    try:
        return _Parser(_tighten(segment)).parse()
    except (ValueError, ZeroDivisionError, IndexError):
        return None


def _tighten(segment):
    """Strip the spacing the reader tolerates but the parser does not need."""
    return segment.replace(' ', '').replace('\t', '')


def _read_step(line, start, end):
    """The (left, right) values claimed by the '=' spanning [start, end)."""
    left, right = _left_of(line, start), _right_of(line, end)
    if not left or not right:
        return None
    for side in (left, right):
        if not _complete(side) or not _balanced(side) or not _NUM_RE.search(side):
            return None
        if not _no_bare_neighbours(side):
            return None
    if not _claims_a_calculation(left):
        return None
    lhs, rhs = _value(left), _value(right)
    if lhs is None or rhs is None:
        return None
    return lhs, rhs


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

class StepAudit(NamedTuple):
    """What the stated calculations of one solution amount to."""

    n_steps: int        # equations the reader could check
    n_ok: int           # of those, ones that hold exactly
    n_bad: int          # of those, ones that do not
    reach_hits: int     # stated results equal to the solution's own answer
    reach_any: bool     # some stated result is the answer
    reach_last: bool    # the final stated result is the answer


def stated_steps(text):
    """Exact (left, right) values for every checkable equation in a solution."""
    out = []
    if '=' not in str(text):
        return out
    for line in _clean(text):
        for m in _EQ_RE.finditer(line):
            step = _read_step(line, m.start(), m.end())
            if step is not None:
                out.append(step)
    return out


def audit(text, answer):
    """Full arithmetic audit of one solution against its own final answer.

    ``answer`` is the string produced by ``common.parse_answer`` — a Fraction
    string such as ``'17'`` or ``'3/4'``, or None when nothing parses.
    """
    steps = stated_steps(text)
    ok = sum(1 for lhs, rhs in steps if abs(lhs - rhs) < _TOL)
    results = [rhs for _, rhs in steps]
    target = _target(answer)
    if target is None:
        hits = 0
        last = False
    else:
        hits = sum(1 for r in results if abs(r - target) < _TOL)
        last = bool(results) and abs(results[-1] - target) < _TOL
    return StepAudit(len(steps), ok, len(steps) - ok, hits, hits > 0, last)


def _target(answer):
    if answer is None:
        return None
    try:
        return Fraction(answer)
    except (ValueError, ZeroDivisionError):
        return None
