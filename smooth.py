import multiprocessing
from fit import fit, poly_eval, DEFAULT_TRIALS
from cli_util import panic, parse_args, pos_int

DEFAULT_MAX_PROCESSES = 1

HELP_TEXT = '''Applies Svitzsky-Golay smoothing to a 2D dataset.

'''

def smooth(degree, data, trials, steps, max_perturb, window, max_subprocs, smooth_procs):
    if max_subprocs < 1:
        max_subprocs = os.process_cpu_count()
    if smooth_procs < 1:
        smooth_procs = os.process_cpu_count()
    fit_procs = max_subprocs // smooth_procs

    half_window = window // 2
    smoothable_start = half_window
    smoothable_end = len(data) - half_window
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
    
    smoothed = smooth(degree, points, trials, steps, max_perturb, window, max_subprocs,
        smooth_procs)

    out_file.writelines(f'{point[0]},{point[1]}\n' for point in smoothed)

if __name__ == '__main__':
    _run_cli()
