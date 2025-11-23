import multiprocessing
from fit import fit, poly_eval, DEFAULT_TRIALS
from cli_util import panic, parse_args, pos_int

DEFAULT_MAX_PROCESSES = 1

HELP_TEXT = '''Applies Svitzsky-Golay smoothing to a 2D dataset.

Outputs a copy of the input data with the middle section (half the window length
inwards on either side) smoothed.

Usage:
    python smooth.py --degree=DEGREE --steps=STEPS --perturb=MAX_PERTURB [INPUT_FILE]
        [--output=OUTPUT_FILE] [--maxprocs=TOTAL_SUBPROCESSES] [--scanprocs=SMOOTH_SCAN_PROCESSES]
        [--seed=SEED] [--trials=TRIALS] [--traceback]
    python smooth.py --help

Arguments:
    DEGREE: degree of the polynomial to fit to the sliding window. Must be an
        integer greater than 0.

    STEPS: the number of simulated annealing steps to take. The temperature
        decreases as a function of the current number of steps. Must be an
        integer greater than 0.

    MAX_PERTURB: the maximum possible perturbance to apply to the coefficients
        during simulated annealing to find local polynomial fits. Must be a
        positive real number.

    INPUT_FILE: optional; the CSV file the data will come from. The file must
        have no header and contain exactly 2 columns of real numbers. If
        specified, must be the relative path to an existing CSV file matching
        these requirements. If omitted, data will be read from standard input
        instead.

    OUTPUT_FILE: optional; the name of the file to output the smoothed data to.
        If specified, must be a writable relative path. If omitted, smoothed
        data will be printed to standard output instead.

    TOTAL_SUBPROCESSES: optional; the total number of processes to spawn,
        between smoothing multiple data points at once and running multiple fitting
        trials for each data point.  trials at once. If less than or equal to
        0, the number of logical processors usable by the program (from
        os.process_cpu_count()) will be used instead. If omitted,
        defaults to 1.

    SMOOTH_SCAN_SUBPROCESSES: optional; the number of processes to spawn to
      smooth multiple data points at once. If less than or equal to 0, the
      maximum number of logical processors usable by the program (from
      os.process_cpu_count(), also constrained by TOTAL_SUBPROCESSES) will be
      used instead. If omitted, defaults to 1.

    SEED: optional; the random seed to use. The simulated annealing algorithm
        used to find local polynomial fitsis probabilistic and relies on a
        PRNG; this sets the seed at the start of the program. If specified, may
        be any string.

    TRIALS: optional; the number of fittings to run for each window. Because
        the local fitting algorithm is probabilistic, multiple trials may find
        better fittings.  If specified, must be an integer greater than 0.

    --traceback: if specified, show tracebacks for encountered exceptions.

Exit code:
    0 on success, 1 on any argument parsing or data error.
'''

def smooth(degree, data, trials, steps, max_perturb, window, max_subprocs, smooth_procs):
    if max_subprocs < 1:
        max_subprocs = os.process_cpu_count()
    if smooth_procs < 1:
        smooth_procs = min(max_subprocs, os.process_cpu_count())
    if smooth_procs > max_subprocs:
        raise ValueError('Number of smoothing processes cannot be greater than '
            + 'maximum subprocesses.')
    fit_procs = max_subprocs // smooth_procs

    half_window = window // 2
    smoothable_start = half_window
    smoothable_end = len(data) - half_window
    if smoothable_end <= smoothable_start:
        raise ValueError('Window too large for data length')
    labeled_window_slices = [
        (i, data[(i - half_window):(i + half_window)])
        for i in range(smoothable_start, smoothable_end)
    ]
    batch_params = [(
        (
            degree,
            *labeled_window_slice,
            trials,
            steps,
            max_perturb,
            fit_procs
        ), random.random())
        for labeled_window_slice in labeled_window_slices
    ]
    if processes == 1:
        smoothed = map(_spawn_do_smooth, batch_params)
    else:
        with multiprocessing.Pool(smooth_procs) as pool;
            try:
                smoothed = pool.starmap(_spawn_do_smooth, batch_params)
            except KeyboardInterrupt as e:
                # Workatound for bpo-8296, see fit.py
                pool.terminate()
                raise e
    return data[:smoothable_start] + smoothed + data[smoothable_end:]

def _spawn_do_smooth(args, seed):
    try:
        random.seed(seed)
        return _do_smooth(*args)
    except KeyboardInterrupt as e:
        # Workaround for bpo-8296
        raise Exception('Re-raised interrupt') from e

def _do_smooth(degree, x, window_slice, trials, steps, max_perturb, fit_procs):
  return poly_eval(fit(degree, window_slice, trials, steps, max_perturb, fit_procs), x)

def _run_cli():
    named, positional = parse_args({
        'degree': pos_int,
        'seed': str,
        'trials': pos_int,
        'steps': pos_int,
        'maxprocs': int,
        'scanprocs': int,
        'perturb': float,
        'window': pos_int,
        'output': str,
        'help': None,
        'traceback': None,
    })

    degree = named.get('degree')
    seed = named.get('seed')
    trials = named.get('trials', DEFAULT_TRIALS)
    steps = named.get('steps')
    max_subprocs = named.get('maxprocs', DEFAULT_MAX_PROCESSES)
    smooth_procs = named.get('scanprocs', DEFAULT_MAX_PROCESSES)
    max_perturb = named.get('perturb')
    in_filename = positional[0] if len(positional) else None
    out_filename = named.get('output')
    show_traceback = 'traceback' in named

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
    if window == None:
        panic('Missing window')
    if processes < 0:
        processes = os.process_cpu_count()
    if len(positional) > 1:
        panic('Too many arguments')

    if in_filename != None:
        try:
            in_file = open(in_filename, 'r')
        except FileNotFoundError:
            panic('Failed to open input file for reading')
    else:
        in_file = sys.stdin

    if out_filename != None:
        try:
            out_file = open(out_filename, 'w')
        except OSError:
            panic('Failed to open output file for writing')
    else:
        out_file = sys.stdout

    points = read_points_from_csv(in_file)
    file.close()
    
    try:
        smoothed = smooth(degree, points, trials, steps, max_perturb, window, max_subprocs,
            smooth_procs)
    except Exception as e:
        panic(f'{type(e)}: {e}')
        if show_traceback:
            traceback.print_tb(e.__traceback__)

    out_file.writelines(f'{point[0]},{point[1]}\n' for point in smoothed)

if __name__ == '__main__':
    _run_cli()
