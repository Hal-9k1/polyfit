import importlib
import os
import random
import sys
import multiprocessing
from cli_util import panic, parse_args, read_points_from_csv, pos_int, pos_float

DEFAULT_TRIALS = 1
DEFAULT_PROCESSES = 1
HELP_TEXT = '''Fits a polynomial curve to a 2D data set.

Uses matrix operations to find the least squares fit. Coefficients of the fitted
curve are printed to standard output, highest degree first.

Usage:
    python fit.py --degree=DEGREE [INPUT_FILE]
    python fit.py --help

Arguments:
    DEGREE: degree of the polynomial to fit to the data. Must be an integer
        greater than 0.

    INPUT_FILE: optional; the CSV file the data will come from. The file must
        have no header and contain exactly 2 columns of real numbers. If
        specified, must be the relative path to an existing CSV file matching
        these requirements. If omitted, data will be read from standard input
        instead.

Exit code:
    0 on success, 1 on any argument parsing or data error.
'''

def fit(degree, data):
    mat = _Mat(
        degree,
        len(data),
        sum([[pow(point[0], n + 1) for n in range(degree)] for point in data], start=[])
    )
    mat_t = mat.transpose()
    # coeffs = (x^T * x)^-1 * x^T * y
    return ((mat_t * mat).invert() * mat_t * _Mat.colvec([point[1] for point in data])).as_colvec()

def poly_eval(coeffs, x):
    return sum(k * pow(x, n) for n, k in enumerate(coeffs))

class _Mat:
    def __init__(self, width, height, data=None):
        if data and len(data) != width * height:
            raise ValueError('Inconsistent dimensions with data')
        self._data = data or [0 for _ in range(width * height)]
        self._width = width
        self._height = height

    def identity(dim):
        data = [0] * (dim * dim)
        for i in range(dim):
            data[i * (dim + 1)] = 1
        return _Mat(dim, dim, data)

    def colvec(data):
        return _Mat(1, len(data), data)

    def transpose(self):
        data = [None for _ in self._data]
        for x in range(self._width):
            for y in range(self._height):
                data[x * self._height + y] = self._data[y * self._width + x]
        return _Mat(self._height, self._width, data)

    def invert(self):
        ident = _Mat.identity(self._height)
        solved = self.augment(ident).rref()
        print(solved)
        if solved._select_cols(0, self._width) != ident:
            raise ValueError('Matrix is not invertible')
        return solved._select_cols(self._width, solved._width)

    def rref(self):
        rows = [self._get_row(y) for y in range(self._height)]
        for y in range(self._height):
            pivots = [_Mat._find_pivot(row) for row in rows]
            rp_sorted = sorted(zip(rows, pivots), key=lambda rp: rp[1])
            rows = [rp[0] for rp in rp_sorted]
            pivots = [rp[1] for rp in rp_sorted]
            row = rows[y]
            pivot = pivots[y]
            if pivot == self._width:
                break
            for row2 in rows:
                if row is row2:
                    continue
                fac = row2[pivot] / row[pivot]
                for x in range(self._width):
                    row2[x] -= row[x] * fac
        for row in rows:
            pivot = _Mat._find_pivot(row)
            if pivot == self._width:
                break
            fac = 1 / row[pivot]
            for x in range(self._width):
                row[x] *= fac
        return _Mat(self._width, self._height, sum(rows, start=[]))

    def augment(self, other):
        if self._height != other._height:
            raise ValueError('Invalid matrix augment')
        data = []
        for y in range(self._height):
            data.extend(self._get_row(y))
            data.extend(other._get_row(y))
        return _Mat(self._width + other._width, self._height, data)

    def as_colvec(self):
        if self._width != 1:
            raise ValueError('_Matrix is not a column vector')
        return self._data

    def __mul__(self, other):
        if self._width != other._height:
            raise ValueError('Invalid matrix multiplication')
        width = other._width
        height = self._height
        data = [None] * (width * height)
        other_tp = other.transpose()
        for x in range(width):
            col = other_tp._get_row(x)
            for y in range(height):
                row = self._get_row(y)
                data[y * width + x] = sum(r * c for r, c in zip(row, col))
        return _Mat(width, height, data)

    def __str__(self):
        buf = ''
        for i, v in enumerate(self._data):
            buf += str(v) + ('\n' if i % self._width == self._width - 1 else '\t')
        return buf

    def __eq__(self, other):
        return (isinstance(other, type(self)) and self._width == other._width
            and self._data == other._data)

    def _find_pivot(row):
        for i, v in enumerate(row):
            if v != 0:
                return i
        return len(row)

    def _get_row(self, y):
        i = y * self._width
        return self._data[i:i + self._width]

    def _select_cols(self, start, end):
        data = [p[1] for p in enumerate(self._data) if start <= p[0] % self._width < end]
        return _Mat(end - start, self._height, data)

def _get_error(coeffs, data):
    total = 0
    for point in data:
        predicted = poly_eval(coeffs, point[0])
        total += pow(predicted - point[1], 2)
    return total

def _run_cli():
    named, positional = parse_args({
        'degree': pos_int,
        'help': None,
    })
    degree = named.get('degree')
    filename = positional[0] if len(positional) else None

    if 'help' in named:
        print(HELP_TEXT, file=sys.stderr)
        exit(0)

    if degree == None:
        panic('Missing degree')
    if len(positional) > 1:
        panic('Too many arguments')

    if filename != None:
        try:
            file = open(file, 'r')
        except FileNotFoundError:
            panic('Failed to open input file')
    else:
        file = sys.stdin

    points = read_points_from_csv(file)
    file.close()
    
    coeffs = fit(degree, points, trials, steps, max_perturb, processes)
    print(f'Error: {_get_error(coeffs, points)}')
    for coeff in reversed(coeffs):
        print(coeff)

if __name__ == '__main__':
    _run_cli()
