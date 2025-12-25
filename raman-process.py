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
    python raman-process.py [INPUT_FILE] [--spectrum=SPECTRUM_OUTPUT_FILE]
    [--peaks=PEAKS_OUTPUT_FILE] [--stdout=STDOUT_CONTENT]
    python raman-process.py --help

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
from cli_util import panic, parse_args, read_points_from_csv, pos_int, pos_float

NM_PER_CM = 10**7
INCIDENT_NM = 532
MIN_WAVELENGTH_INCREASE_NM = 8
MAX_WAVELENGTH_INCREASE_NM = 140
SAVITZKY_GOLAY_DEGREE = 4
SAVITZKY_GOLAY_WINDOW = 75

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
        except ValueError as e:
            raise ValueError(f'Encountered non-numeric data: \'{line}\'') from e
        points.append((row[1], row[3]))
    file.close()
    return points

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
    coeffs = fit(4, data, matrix_check=False)
    data = [
        (wavenumber_shift, intensity - poly_eval(coeffs, wavenumber_shift))
        for wavenumber_shift, intensity in data
    ]
    data = [
        (wavenumber_shift, max(0, intensity))
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
    data = [
        (wavenumber_shift, intensity)
        for wavenumber_shift, intensity in data
        if 0 <= intensity < 600
    ]
    data = sorted(data, key=lambda p: p[0])
    window = []
    peaks = []
    for point in data:
        wavenumber_shift, intensity = point
        window.append(point)
        if len(window) >= 3 and window[-1][0] - window[1][0] > 50:
            del window[0]
            if window[-2][1] - window[0][1] > 5 and window[-2][1] > window[-1][1]:
                peaks.append(window[-2][0])
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
            spectrum_out_file = open(out_filename, 'w')
        except OSError:
            panic('Failed to open spectrum output file for writing')
    if peaks_out_filename != None:
        try:
            peaks_out_file = open(out_filename, 'w')
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
        spectrum_out_file.writelines(f'{point[0]},{point[1]}\n' for point in data)
        spectrum_out_file.close()
    if peaks_out_file != None:
        peaks_out_file.writelines(f'{peak}\n' for peak in peaks)
        peaks_out_file.close()

if __name__ == '__main__':
    _run_cli()
