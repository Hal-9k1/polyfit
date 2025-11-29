import importlib
import os
import random
import sys
import multiprocessing
from cli_util import panic, parse_args, read_points_from_csv, pos_int, pos_float

DEFAULT_TRIALS = 1
DEFAULT_PROCESSES = 1
HELP_TEXT = '''Fits a polynomial curve to a 2D data set.

Uses simulated annealing to implement the least squares method. Coefficients of
the fitted curve are printed to standard output, highest degree first.

Usage:
    python fit.py --degree=DEGREE --steps=STEPS --perturb=MAX_PERTURB \
[INPUT_FILE] [--seed=SEED] [--trials=TRIALS] [--processes=PROCESSES]
    python fit.py --help

Arguments:
    DEGREE: degree of the polynomial to fit to the data. Must be an integer
        greater than 0.

    STEPS: the number of simulated annealing steps to take. The temperature
        decreases as a function of the current number of steps. Must be an
        integer greater than 0.

    MAX_PERTURB: the maximum possible perturbance to apply to the coefficients
        during simulated annealing. Perturbance naturally decreases with time,
        as does the rejection chance for increases in error. Must be a real
        number greater than 0.0.

    INPUT_FILE: optional; the CSV file the data will come from. The file must
        have no header and contain exactly 2 columns of real numbers. If
        specified, must be the relative path to an existing CSV file matching
        these requirements. If omitted, data will be read from standard input
        instead.

    SEED: optional; the random seed to use. The simulated annealing algorithm is
        probabilistic and relies on a PRNG; this sets the seed at the start of
        the program. If specified, may be any string. If omitted, the seed will
        be indeterminate and results will likely be random per-run.

    TRIALS: optional; the number of fittings to run. Because the algorithm is
        probabilistic, multiple trials may find better fittings. The
        coefficients of the curve with the smallest error are printed. If
        specified, must be an integer greater than 0. If omitted,
        defaults to 1.

    PROCESSES: optional; the number of processes to spawn to compute multiple
        trials at once. Will have no effect other than slowing startup time if
        this number exceeds the number of trials. If specified, must be an
        integer.  If less than or equal to 0, the number of logical processors
        usable by the program (from os.process_cpu_count()) will be used
        instead. If omitted, defaults to 1.

Exit code:
    0 on success, 1 on any argument parsing or data error.
'''

def fit(degree, data, trials, steps, max_perturb, processes):
    batch_params = [(degree, data, steps, max_perturb, random.random()) for _ in range(trials)]
    if processes == 1:
        trial_coeffs = map(_spawn_do_fit, batch_params)
    else:
        if processes < 1:
            processes = os.process_cpu_count()
        with multiprocessing.Pool(processes) as pool:
            try:
                trial_coeffs = pool.map(
                    _spawn_do_fit,
                    batch_params
                )
            except KeyboardInterrupt as e:
                # Workatound for apparently unfixed bpo-8296 CPython bug preventing
                # KeyboardInterrupt propagation to subprocs
                pool.terminate()
                raise e
    return min(trial_coeffs, key=lambda x: _get_error(x, data))

def poly_eval(coeffs, x):
    return sum(k * pow(x, n) for n, k in enumerate(coeffs))

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
    return err_new < err_old or random.random() > err_new / err_old * (1 - temp)

def _spawn_do_fit(args):
    try:
        random.seed(args[-1])
        return _do_fit(*(args[:-1]))
    except KeyboardInterrupt as e:
        # Workaround for bpo-8296
        raise Exception('Re-raised interrupt') from e

def _get_error(coeffs, data):
    total = 0
    for point in data:
        predicted = poly_eval(coeffs, point[0])
        total += pow(predicted - point[1], 2)
    return total

def _run_cli():
    named, positional = parse_args({
        'degree': pos_int,
        'seed': str,
        'trials': pos_int,
        'steps': pos_int,
        'processes': int,
        'perturb': pos_float,
        'help': None,
    })
    degree = named.get('degree')
    seed = named.get('seed')
    trials = named.get('trials', DEFAULT_TRIALS)
    steps = named.get('steps')
    processes = named.get('processes', DEFAULT_PROCESSES)
    max_perturb = named.get('perturb')
    filename = positional[0] if len(positional) else None

    if 'help' in named:
        print(HELP_TEXT, file=sys.stderr)
        exit(0)

    if seed != None:
        random.seed(seed)
    if degree == None:
        panic('Missing degree')
    if steps == None:
        panic('Missing steps')
    if max_perturb == None:
        panic('Missing max perturbance')
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
