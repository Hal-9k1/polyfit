'''Analyzes Raman data.

Given Spectrum Studio CSV data, outputs a two-column Raman spectrum CSV.
- Converts wavelengths to wavenumber Stokes shifts (wavenumber/energy DECREASING,
  wavelength INCREASING) from the incident laser wavelength.
- Fits a quartic to the data to estimate fluorescence and other background
  signal, then subtracts it out.
- Applies Savitzky-Golay smoothing (fitting a quartic to a sliding window and
  using the predicted output) to most of the dataset (data points on the
  boundaries are left alone).
- Formats result data as a two-column CSV of wavenumber shift (cm^-1) and
  intensity (arbitrary units from spectrometer).
Also generates a newline-separated list of intensity peaks (given in cm^-1
wavenumber shifts) from the generated spectrum.
If none of --spectrum, --peaks, or --stdout are given, a warning will be printed
and none of the produced data will not be output.

Usage:
    python raman.py [INPUT_FILE] [--spectrum=SPECTRUM_OUTPUT_FILE]
    [--peaks=PEAKS_OUTPUT_FILE] [--stdout=STDOUT_CONTENT]
    python raman.py --help

Arguments:
    INPUT_FILE: optional; the spectrometer CSV file to read.

    SPECTRUM_OUTPUT_FILE: optional; if specified, a writable path to output the
        generated Raman spectrum CSV to. If omitted, the spectrum will not be
        output to a file. Must not be combined with --stdout=spectrum.

    PEAKS_OUTPUT_FILE: optional; if specified, a writable path to output
        the newline-separated list of intensity peaks to. If omitted, the peaks
        list will not be output to a file. Must not be combined with
        --stdout=peaks.

    STDOUT_CONTENT: optional; if specified, must be one of 'spectrum' or
        'peaks'. The selected data will be printed to standard output. If
        omitted, nothing will be written to standard output.

Exit code:
    0 on success, 1 on any argument parsing or data error.
'''

import sys

from fit import fit, poly_eval
from smooth import smooth
from cli_util import (
    panic,
    parse_args,
    read_points_from_csv,
    pos_int,
    pos_float,
    print_points,
    print_reals
)
from hilbert import hilbert_decomp

NM_PER_CM = 10**7
INCIDENT_NM = 532
MIN_WAVELENGTH_INCREASE_NM = 8
MAX_WAVELENGTH_INCREASE_NM = 140
SAVITZKY_GOLAY_DEGREE = 4
SAVITZKY_GOLAY_WINDOW = 75
PEAK_DETECTION_INTENSITY_THRESHOLD = 5
PEAK_DETECTION_WINDOW_WIDTH = 100
MAX_INLIER_INTENSITY = 600

def parse_spectrometer_csv(file):
    for _ in range(5):
        if not file.readline():
            raise ValueError('Header missing or incomplete')
    points = []
    while True:
        line = file.readline()
        if not line:
            break
        try:
            row = [float(num) for num in line.rstrip('\n').split(',')]
            points.append((row[1], row[3]))
        except ValueError as e:
            raise ValueError(f'Encountered non-numeric data: \'{line}\'') from e
        except IndexError as e:
            raise ValueError(f'Encountered row with too few columns: \'{line}\'') from e
    file.close()
    return points

def _stddev(data):
    # sqrt of mean of squares of devs from mean
    l = len(data)
    mean = sum(data) / l
    sum_sqdev = sum(pow(x - mean, 2) for x in data)
    return pow(sum_sqdev / l, 0.5)

def detect_peaks_hilbert(data, noise):
    data = sorted(data, key=lambda p: p[0])
    noise_stddev = _stddev([p[0] for p in noise])
    extrema = []
    for past, present, future in zip(data, data[1:], data[2:]):
        if (past[1] < present[1]) == (future[1] < present[1]):
            # (wavenumber_shift, intensity, is_maximum)
            extrema.append((*present, past[1] < present[1]))
    peaks = []
    for past, present, future in zip(extrema, extrema[1:], extrema[2:]):
        if present[2]:
            avg_fall = present[1] - (past[1] + future[1]) / 2
            if avg_fall > noise_stddev:
                peaks.append(present[0])
    return peaks

def detect_peaks(data):
    window = []
    peaks = []
    for point in sorted(data, key=lambda p: p[0]):
        wavenumber_shift, intensity = point
        window.append(point)
        if len(window) < 3:
            continue
        half = len(window) // 2
        left = window[0]
        midleft = window[half - 1]
        middle = window[half]
        midright = window[half + 1]
        right = window[-1]
        if right[0] - left[0] < PEAK_DETECTION_WINDOW_WIDTH:
            continue
        del window[0]
        long_check = (min(middle[1] - left[1], middle[1] - right[1])
            > PEAK_DETECTION_INTENSITY_THRESHOLD)
        short_check = max(midleft[1], middle[1], midright[1]) == middle[1]
        if long_check and short_check:
            peaks.append(middle[0])
    return peaks

def raman_process(data):
    # filter out intensities that haven't shifted much from incident because
    # those overpower the raman spectrum
    data = [
        (wavelength, intensity)
        for wavelength, intensity in data
        if MIN_WAVELENGTH_INCREASE_NM < wavelength - INCIDENT_NM < MAX_WAVELENGTH_INCREASE_NM
    ]
    # convert wavelengths to shifts in wavenumber from incident light.
    data = [
        (NM_PER_CM / INCIDENT_NM - NM_PER_CM / wavelength, intensity)
        for wavelength, intensity in data
    ]
    # fit a quartic to intensity *by wavenumber shift* to approximate fluorescent
    # interference and subtract it out
    # also clamp intensities to minimum 0
    coeffs = fit(4, data, matrix_check=False)
    data = [
        (wavenumber_shift, max(0, intensity - poly_eval(coeffs, wavenumber_shift)))
        for wavenumber_shift, intensity in data
    ]
    # apply savitzky-golay smoothing, discarding boundaries of data that can't
    # have full windows
    data = smooth(
        SAVITZKY_GOLAY_DEGREE,
        data,
        SAVITZKY_GOLAY_WINDOW,
        1,
        end_mode='clip',
        matrix_check=False
    )
    # drop (don't clamp) out-of-bounds intensities produced by smoothing
    # algorithm, which appear as crazy outliers on a graph
    data = [
        (wavenumber_shift, intensity)
        for wavenumber_shift, intensity in data
        if 0 <= intensity < MAX_INLIER_INTENSITY
    ]

    high_energy, data = hilbert_decomp(data)
    #peaks = detect_peaks_hilbert(data, high_energy)
    peaks = detect_peaks(data)
    return data, peaks

def _typecheck_stdout(v):
    v = v.lower()
    if v not in ('spectrum', 'peaks'):
        raise ValueError
    return v

def _run_cli():
    named, positional = parse_args({
        'stdout': str,
        'spectrum': str,
        'peaks': str,
        'stdout': _typecheck_stdout,
        'help': None,
    })

    in_filename = positional[0] if len(positional) else None
    spectrum_out_filename = named.get('spectrum')
    peaks_out_filename = named.get('peaks')
    stdout_content = named.get('stdout')

    if 'help' in named:
        print(__doc__, file=sys.stderr)
        exit(0)

    if len(positional) > 1:
        panic('Too many arguments')

    if in_filename != None:
        try:
            in_file = open(in_filename, 'r')
        except OSError:
            panic('Failed to open input file for reading')
    else:
        in_file = sys.stdin

    spectrum_out_file = None
    peaks_out_file = None
    if spectrum_out_filename == peaks_out_filename == stdout_content == None:
        print('Doing analysis with no output selected. Specify --spectrum, --peaks, or --stdout.',
            file=sys.stderr)
    if spectrum_out_filename != None:
        try:
            spectrum_out_file = open(spectrum_out_filename, 'w')
        except OSError:
            panic('Failed to open spectrum output file for writing')
    if peaks_out_filename != None:
        try:
            peaks_out_file = open(peaks_out_filename, 'w')
        except OSError:
            panic('Failed to open peaks output file for writing')
    if stdout_content == 'spectrum':
        spectrum_out_file = sys.stdout
    elif stdout_content == 'peaks':
        peaks_out_file = sys.stdout

    data, peaks = raman_process(parse_spectrometer_csv(in_file))

    if spectrum_out_file != None:
        spectrum_out_file.write('Wavenumber shift from 532nm (cm^-1),'
            + 'Intensity (arbitrary spectrometer units)\n')
        print_points(data, file=spectrum_out_file)
        spectrum_out_file.close()
    if peaks_out_file != None:
        print_reals(data, file=peaks_out_file)
        peaks_out_file.close()

if __name__ == '__main__':
    _run_cli()
