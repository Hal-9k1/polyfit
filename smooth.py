from fit import fit, DEFAULT_TRIALS
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

    smoothable_start = window // 2
    smoothable_end = len(data) - window // 2
    fit(degree, data, trials, steps, max_perturb, fit_procs)

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
