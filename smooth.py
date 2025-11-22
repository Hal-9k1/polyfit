from fit import fit, DEFAULT_TRIALS
from cli_util import panic, parse_args, pos_int

DEFAULT_MAX_PROCESSES = 1

HELP_TEXT = '''foobar

'''

def smooth():
    pass

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
        'help': None,
    })

    degree = named.get('degree')
    seed = named.get('seed')
    trials = named.get('trials', DEFAULT_TRIALS)
    steps = named.get('steps')
    max_subprocs = named.get('maxprocs', DEFAULT_MAX_PROCESSES)
    smooth_procs = named.get('scanprocs', DEFAULT_MAX_PROCESSES)
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
    if window == None:
        panic('Missing window')
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

    points = read_points_from_csv(file)
    file.close()
    

if __name__ == '__main__':
    _run_cli()
