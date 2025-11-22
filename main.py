import importlib
import os
import random
import sys
import multiprocessing

DEFAULT_TRIALS = 1
DEFAULT_PROCESSES = 1
DEFAULT_MAX_PERTURB = 100

def panic(msg):
    print(msg, file=sys.stderr)
    exit(1)

def parse_args(names_to_types):
    named = {}
    positional = []
    can_parse_named = True
    for arg in sys.argv[1:]:
        parts = arg.split('=')
        if parts[0].startswith('--') and can_parse_named:
            if len(parts) == 2 and parts[0][2:] in names_to_types:
                key = parts[0][2:]
                try:
                    named[key] = names_to_types[key](parts[1])
                except ValueError:
                    panic(f'Invalid {key}: {parts[1]}')
            elif parts[0] == '--':
                can_parse_named = False
            else:
                panic(f'Malformed argument {arg}')
        else:
            positional.append(arg)
    return named, positional

def pos_int(stringified):
    val = int(stringified)
    if val <= 0:
        raise ValueError
    return val

def main():
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
    file = positional[0] if len(positional) else None

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
            point = tuple(float(x) for x in line.split(','))
        except ValueError:
            panic('Found non-numeric data')
        if len(point) != 2:
            panic('Too many fields for data point')
        points.append(point)

    file.close()
    
    for coeff in reversed(fit(degree, points, trials, steps, max_perturb, processes)):
        print(coeff)

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
    return min(trials, key=lambda x: _get_sq_error(x, data))

def _get_temp(time_frac):
    return 1 - time_frac

def _perturb_coeffs(coeffs, temp, max_perturb):
    return [k + (random.random() - 0.5) * 2 * temp * max_perturb for k in coeffs]

def _should_change_coeffs(temp, err_old, err_new):
    return err_new < err_old
    #return err_new < err_old or random.random() > err_new / err_old * (1 - temp)

def _spawn_fit(args, seed):
    random.seed(seed)
    return _do_fit(*args)

def _do_fit(degree, data, steps, max_perturb):
    # least degree first
    coeffs = [0 for _ in range(degree + 1)]
    for i in range(steps):
        temp = _get_temp(i / steps)
        new_coeffs = _perturb_coeffs(coeffs, temp, max_perturb)
        if _should_change_coeffs(temp, _get_sq_error(coeffs, data),
                _get_sq_error(new_coeffs, data)):
            coeffs = new_coeffs
    return coeffs

def _get_sq_error(coeffs, data):
    total = 0
    for point in data:
        predicted = sum(k * pow(point[0], n) for n, k in enumerate(coeffs))
        total += pow(predicted - point[1], 2)
    return total

if __name__ == '__main__':
    main()
