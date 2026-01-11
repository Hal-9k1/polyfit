'''Polynomial fitting and evaluation routines.

When run, supports a CLI to fit a polynomial to input points using least-squares.
'''

_CLI_DOC = '''Fits a polynomial curve to a 2D data set.

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

import os
import random
import sys
from cli_util import panic, parse_args, read_points_from_csv, pos_int, pos_float, print_reals

def fit(degree, data):
    '''Fits a polynomial to a list of points.

    Returns the list of coefficients for the least squares fit polynomial of the
    specified degree to the given data. The coefficient list is ordered least
    degree first.

    Args:
        degree: The degree of the polynomial to fit.
        data: The list of points (2-tuples of floats) to fit the polynomial to.

    Returns:
        The list of coefficients of the least-squares fit polynomial.
    '''
    mat = _Mat(
        degree + 1,
        len(data),
        sum([[pow(point[0], n) for n in range(degree + 1)] for point in data], start=[])
    )
    mat_t = mat.transpose()
    # coeffs = (x^T * x)^-1 * x^T * y
    # where each row in x are the ascending powers of a data point's independent variable
    # and each row in y is the corresponding dependent variable
    return (
        (mat_t * mat).invert()
        * mat_t * _Mat.colvec([point[1] for point in data])
    ).as_colvec()

def poly_eval(coeffs, x):
    '''Evaluates a polynomial at a given input.

    Evaluates the polynomial specified by the list of coefficients given in
    ascending degree at the given input value.

    Args:
        coeffs: The coefficients of the polynomial to evaluate.
        x: The input to evaluate the polynomial at.

    Returns:
        The result of the evaluation.
    '''

    return sum(k * pow(x, n) for n, k in enumerate(coeffs))

class _Mat:
    '''Represents an arbitrarily sized matrix.'''
    _EPSILON = pow(10, -5)

    def __init__(self, width, height, data=None):
        if data and len(data) != width * height:
            raise ValueError('Inconsistent dimensions with data')
        self._data = data or [0 for _ in range(width * height)]
        self._width = width
        self._height = height

    def identity(dim):
        '''Returns a dim*dim identity matrix.

        Args:
            dim: The dimension of the returned matrix.
        '''
        data = [0] * (dim * dim)
        for i in range(dim):
            data[i * (dim + 1)] = 1
        return _Mat(dim, dim, data)

    def colvec(data):
        '''Returns a single column matrix populated with data from a list.

        Args:
            data: The list of numbers to populate the matrix with.
        '''
        return _Mat(1, len(data), data)

    def transpose(self):
        data = [None for _ in self._data]
        for x in range(self._width):
            for y in range(self._height):
                data[x * self._height + y] = self._data[y * self._width + x]
        return _Mat(self._height, self._width, data)

    def invert(self):
        '''Returns a new matrix which is the inverse of this one.

        If this matrix is singular, the return value is undefined.
        '''
        ident = _Mat.identity(self._height)
        solved = self.augment(ident).rref()
        left = solved._select_cols(0, self._width)
        #if left != ident and check:
        #    raise ValueError('Matrix is not invertible; expected identity:\n' + str(left))
        return solved._select_cols(self._width, solved._width)

    def rref(self):
        '''Returns a new matrix which is the row reduced echelon form of this one.'''
        rows = [self._get_row(y) for y in range(self._height)]
        #print(self, file=sys.stderr)
        for y in range(self._height + 1): # sort one more time after processing all rows
            pivots = [_Mat._find_pivot(row) for row in rows]
            # sort by pivot, then tiebreak with magnitude at pivot
            rp_sorted = sorted(zip(rows, pivots), key=lambda rp: (rp[1], abs(rp[0][rp[1]])))
            rows = [rp[0] for rp in rp_sorted]
            pivots = [rp[1] for rp in rp_sorted]
            if y == self._height:
                # on the final iteration, stop after sorting
                break
            row = rows[y]
            pivot = pivots[y]
            if pivot == self._width:
                # the rest of the rows lack pivots
                break
            #print(f'chose row {row}')
            for row2 in rows:
                if row is row2:
                    continue
                fac = row2[pivot] / row[pivot]
                #print(f'subtracting {fac} times row from {row2}')
                for x in range(pivot, self._width):
                    # cheat a little bit to avoid crazy floating point error where
                    # bignum - (bignum / smallnum) * smallnum != 0
                    row2[x] = 0 if x == 0 else row2[x] - row[x] * fac
            #print(_Mat(self._width, self._height, sum(rows, start=[])), file=sys.stderr)
        for row in rows:
            pivot = _Mat._find_pivot(row)
            if pivot == self._width:
                break
            fac = 1 / row[pivot]
            for x in range(self._width):
                row[x] *= fac
        return _Mat(self._width, self._height, sum(rows, start=[]))

    def augment(self, other):
        '''Returns a new matrix whose rows are the concatenation of this and
        another matrices' rows.

        Args:
            other: The matrix to right-augment this matrix with.

        Raises:
            ValueError: The two matrices' do not have the same number of rows.
        '''
        if self._height != other._height:
            raise ValueError('Invalid matrix augment')
        data = []
        for y in range(self._height):
            data.extend(self._get_row(y))
            data.extend(other._get_row(y))
        return _Mat(self._width + other._width, self._height, data)

    def as_colvec(self):
        '''Checks that this matrix is a column vector and returns the contents.

        Raises:
            ValueError: This matrix is not a column vector.
        '''
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
        strs = [f'{x: 2g}' for x in self._data]
        widths = [float('-inf')] * self._width
        for i, s in enumerate(strs):
            col = i % self._width
            widths[col] = max(widths[col], len(s))
        buf = ''
        for i, s in enumerate(strs):
            col = i % self._width
            buf += s.ljust(widths[col] + 1)
            if col == self._width - 1:
                buf += '\n'
        return buf

    def __eq__(self, other):
        return (isinstance(other, type(self)) and self._width == other._width
            and all(abs(a - b) < _Mat._EPSILON for a, b in zip(self._data, other._data)))

    def _find_pivot(row):
        for i, v in enumerate(row):
            if abs(v) > _Mat._EPSILON:
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
        print(_CLI_DOC, file=sys.stderr)
        exit(0)

    if degree == None:
        panic('Missing degree')
    if len(positional) > 1:
        panic('Too many arguments')

    if filename != None:
        try:
            file = open(file, 'r')
        except OSError:
            panic('Failed to open input file for reading')
    else:
        file = sys.stdin

    points = read_points_from_csv(file)
    file.close()
    
    coeffs = fit(degree, points)
    print(f'Error: {_get_error(coeffs, points)}')
    print_reals(reversed(coeffs))

if __name__ == '__main__':
    _run_cli()
