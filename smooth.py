'''Applies Savitzsky-Golay smoothing to a 2D dataset.

Outputs a copy of the input data with the middle section (half the window length
inwards on either side) smoothed.

Usage:
    python smooth.py --degree=DEGREE --window=WINDOW [INPUT_FILE] [--output=OUTPUT_FILE]
    [--ends=END_BEHAVIOR] [--processes=SUBPROCESSES] [--traceback]
    python smooth.py --help

Arguments:
    DEGREE: degree of the polynomial to fit to the sliding window. Must be an
        integer greater than 0.

    WINDOW: the length of the sliding window centered on a data point on which
        to fit the smoothing polynomial. Must be an integer greater than 0 and
        less than the length of the data.

    INPUT_FILE: optional; the CSV file the data will come from. The file must
        have no header and contain exactly 2 columns of real numbers. If
        specified, must be the relative path to an existing CSV file matching
        these requirements. If omitted, data will be read from standard input
        instead.

    OUTPUT_FILE: optional; the name of the file to output the smoothed data to.
        If specified, must be a writable relative path. If omitted, smoothed
        data will be printed to standard output instead.

    END_BEHAVIOR: optional; how to process points less than half the window
        length away from either end of the data. If specified, must be one of
        'clip', 'extend', or 'preserve'.
         -> clip: do not include the end data points in the output at all.
         -> extend: use the polynomial fitting from the nearest data point with
                a full window to smooth end data points.
         -> preserve: output end data points exactly as they appear in input.
        If omitted, defaults to 'clip'.

    SUBPROCESSES: optional; the number of processes to spawn to smooth multiple
        data points at once. If less than or equal to 0, the maximum number of
        logical processors usable by the program (from os.process_cpu_count())
        will be used instead. If omitted, defaults to 1.

    --traceback: if specified, show tracebacks for encountered exceptions.

Exit code:
    0 on success, 1 on any argument parsing or data error.
'''

import multiprocessing
import os
import random
import sys
import traceback
from fit import fit, poly_eval
from cli_util import panic, parse_args, pos_int, read_points_from_csv, print_points

DEFAULT_MAX_PROCESSES = 1
DEFAULT_END_MODE = 'clip'

def smooth(degree, data, window, smooth_procs, *, end_mode=DEFAULT_END_MODE, matrix_check=True):
    if smooth_procs < 1:
        smooth_procs = _guess_cpu_count()

    data = sorted(data, key=lambda p: p[0])
    half_window = window // 2
    smoothable_start = half_window
    smoothable_end = len(data) - half_window
    if smoothable_end <= smoothable_start:
        raise ValueError('Window too large for data length')
    labeled_window_slices = [
        (data[i][0], data[(i - half_window):(i + half_window)])
        for i in range(smoothable_start, smoothable_end)
    ]
    batch_params = [
        (degree, *labeled_window_slice, matrix_check)
        for labeled_window_slice in labeled_window_slices
    ]
    if smooth_procs == 1:
        smoothed = map(_spawn_do_smooth, batch_params)
    else:
        with multiprocessing.Pool(smooth_procs) as pool:
            try:
                smoothed = pool.starmap(_do_smooth, batch_params)
            except KeyboardInterrupt as e:
                # Workatound for bpo-8296
                pool.terminate()
                raise Exception('Re-raised interrupt') from e
    smoothed_points = list(zip((params[1] for params in batch_params), smoothed))
    if end_mode == 'clip':
        return smoothed_points
    elif end_mode == 'extend':
        extended_start = [
            (x, poly_eval(fit(degree, labeled_window_slices[0][1], matrix_check=matrix_check), x))
            for x, _ in data[:smoothable_start]
        ]
        extended_end = [
            (x, poly_eval(fit(degree, labeled_window_slices[-1][1], matrix_check=matrix_check), x))
            for x, _ in data[smoothable_end:]
        ]
        return extended_start + smoothed_points + extended_end
    elif end_mode == 'preserve':
        return data[:smoothable_start] + smoothed_points + data[smoothable_end:]
    else:
        raise ValueError('Invalid end mode')

def _spawn_do_smooth(args):
    try:
        return _do_smooth(*args)
    except KeyboardInterrupt as e:
        # Workaround for bpo-8296
        raise Exception('Re-raised interrupt') from e

def _do_smooth(degree, x, window_slice, matrix_check):
    return poly_eval(fit(degree, window_slice, matrix_check=matrix_check), x)

def _guess_cpu_count():
    try:
        return os.process_cpu_count()
    except:
        pass
    try:
        return len(os.sched_getaffinity(os.getpid()))
    except:
        pass
    print('Failed to guess CPU count, using 1 process. Use a positive number for --processes to ' +
        'silence.', file=sys.stderr)
    return 1

def _typecheck_end_mode(v):
    v = v.lower()
    if v not in ('clip', 'extend', 'preserve'):
        raise ValueError
    return v

def _run_cli():
    named, positional = parse_args({
        'degree': pos_int,
        'processes': int,
        'window': pos_int,
        'output': str,
        'ends': _typecheck_end_mode,
        'help': None,
        'traceback': None,
    })

    degree = named.get('degree')
    smooth_procs = named.get('processes', DEFAULT_MAX_PROCESSES)
    window = named.get('window')
    in_filename = positional[0] if len(positional) else None
    out_filename = named.get('output')
    end_mode = named.get('ends', DEFAULT_END_MODE)
    show_traceback = 'traceback' in named

    if 'help' in named:
        print(__doc__, file=sys.stderr)
        exit(0)

    if degree == None:
        panic('Missing degree')
    if window == None:
        panic('Missing window')
    if len(positional) > 1:
        panic('Too many arguments')

    if in_filename != None:
        try:
            in_file = open(in_filename, 'r')
        except OSError:
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
    in_file.close()
    
    try:
        smoothed = smooth(degree, points, window, smooth_procs, end_mode=end_mode)
    except Exception as e:
        print(f'{type(e).__name__}: {e}', file=sys.stderr)
        if show_traceback:
            traceback.print_tb(e.__traceback__, file=sys.stderr)
        exit(1)

    print_points(smoothed, file=out_file)

if __name__ == '__main__':
    _run_cli()
