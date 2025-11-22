import importlib
import os
import random
import sys
import multiprocessing
from cli_util import panic, parse_args

DEFAULT_TRIALS = 1
DEFAULT_PROCESSES = 1
DEFAULT_MAX_PERTURB = 100

def fit(degree, data, trials, steps, max_perturb, processes):
    if processes == 1:
        trials = [_spawn_fit((degree, data, steps, max_perturb), random.random())
            for _ in range(trials)]
    else:
        with multiprocessing.Pool(processes) as pool:
            trials = pool.starmap(
                _spawn_fit,
                [((degree, data, steps, max_perturb), random.random()) for _ in range(trials)]
            )
    return min(trials, key=lambda x: _get_error(x, data))

def _do_fit(degree, data, steps, max_perturb):
    # least degree first
    coeffs = [0 for _ in range(degree + 1)]
    for i in range(steps):
        temp = _get_temp(i / steps)
        new_coeffs = _perturb_coeffs(coeffs, temp, max_perturb)
        if _should_change_coeffs(temp, _get_error(coeffs, data),
                _get_error(new_coeffs, data)):
            coeffs = new_coeffs
    return coeffs

def _get_temp(time_frac):
    return 1 - time_frac

def _perturb_coeffs(coeffs, temp, max_perturb):
    return [k + (random.random() - 0.5) * 2 * temp * max_perturb for k in coeffs]

def _should_change_coeffs(temp, err_old, err_new):
    return err_new < err_old or False #random.random() > err_new / err_old * (1 - temp)

def _spawn_fit(args, seed):
    random.seed(seed)
    return _do_fit(*args)

def _get_error(coeffs, data):
    total = 0
    for point in data:
        predicted = sum(k * pow(point[0], n) for n, k in enumerate(coeffs))
        total += pow(predicted - point[1], 2)
    return total

def _run_cli():
    named, positional = parse_args({
        'degree': pos_int,
        'seed': str,
        'trials': pos_int,
        'steps': pos_int,
        'processes': int,
        'perturb': float,
    })
    degree = named.get('degree')
    seed = named.get('seed')
    trials = named.get('trials', DEFAULT_TRIALS)
    steps = named.get('steps')
    processes = named.get('processes', DEFAULT_PROCESSES)
    max_perturb = named.get('perturb', DEFAULT_MAX_PERTURB)
    filename = positional[0] if len(positional) else None

    if seed != None:
        random.seed(seed)
    if degree == None:
        panic('Missing degree')
    if steps == None:
        panic('Missing steps')
    if processes < 0:
        processes = os.process_cpu_count()
    if len(positional) > 1:
        panic('Too many arguments')

    if filename != None:
        try:
            file = open(file, 'r')
        except FileNotFoundError:
            panic('Failed to open input file')
    else:
        file = sys.stdin

    points = []
    for line in file.readlines():
        try:
            point = tuple(float(x) for x in line.split(','))
        except ValueError:
            panic('Found non-numeric data')
        if len(point) != 2:
            panic('Too many fields for data point')
        points.append(point)

    file.close()
    
    coeffs = fit(degree, points, trials, steps, max_perturb, processes)
    print(f'Error: {_get_error(coeffs, points)}')
    for coeff in reversed(coeffs):
        print(coeff)

if __name__ == '__main__':
    _run_cli()
