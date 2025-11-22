import sys
import os

DEFAULT_TRIALS = 1
DEFAULT_STEPS = 

def panic(msg):
    print(msg, file=sys.stderr)
    exit(1)

def main():
    degree = None
    seed = None
    trials = DEFAULT_TRIALS
    steps = None
    file = None
    for arg in sys.argv:
        if arg.startswith('--degree='):
            try:
                degree = 

    if file != None:
        try:
            file = open(sys.argv[2], 'r')
        except FileNotFoundError:
            panic('Failed to open input file')
    else:
        file = sys.stdin

    points = []
    for line in file.readlines():
        try:
            point = (float(x) for x in line.split(','))
        except ValueError:
            panic('Found non-numeric data')
        if len(point) != 2:
            panic('Too many fields for data point')
        points.append(point)

    random.seed(os.environ.get('SEED', 0))
    
    for coeff in fit(degree, points, os.environ.get('TRIALS', 1)):
        print(coeff)

def fit(degree, data, trials, steps):
    return min(
        [_do_fit(degree, data) for _ in range(trials)],
        key=lambda x: _get_sq_error(x, data)
    )

def _do_fit(degree, data, steps):
    # least degree first
    coeffs = [0 for _ in range(degree)]
    for 

def _get_sq_error(coeffs, data):
    total = 0
    for point in data:
        predicted = [k * pow(point[0], n) for n, k in enumerate(coeffs)]
        total += pow(predicted - point[1], 2)
    return total

if __name__ == '__main__':
    main()
