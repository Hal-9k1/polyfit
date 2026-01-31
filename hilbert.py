from cli_util import print_points
from contextlib import nullcontext
import math
import multiprocessing

# inputs less than this distance from 0 return the maximum value of the sinc
# function:
_SINC_EPSILON = pow(10, -12)
# points less than this distance from 0 are skipped when integrating for
# hilbert transformation:
_HILBERT_EPSILON = pow(10, -12)
_ORIG_IF_LOW_PASS_CUTOFF = 0.001
_ORIG_IF_AVG_WINDOW = 320
_PROJ_LOW_PASS_CUTOFF = 0.02
_TRAPEZOID_THRESHOLD = 0.0001

def hilbert(data, pool=None):
    '''Computes the Hilbert transform.

    Computes the Hilbert transform of a function given as a list of points, and
    returns the transformed points.

    Args:
        data: The list of points (2-tuples of floats) to transform.
        pool: If not None, the multiprocessing.Pool to parallelize the
            computation with.

    Returns:
        The list of transformed points, sorted by first element but otherwise
        corresponding 1-to-1 with the input points.
    '''
    return _parallelize(len(data), (sorted(data, key=lambda p: p[0]),), _hilbert_kernel, pool)

def _hilbert_kernel(i, data):
    '''Evaluates and returns the Hilbert transform of a function at a point.

    Hilbert[x(t)] = principal value integral from -oo to oo of x(t)/(t - x) dt

    Args:
        i: The datum index.
        data: The list of points.
    '''
    t = data[i][0]
    return (t, _def_integrate([
        (x, y / (t - x))
        for x, y in data
        if abs(t - x) > _HILBERT_EPSILON # principal value rule
    ]) / math.pi)

def hilbert_decomp(data, parallel=True):
    '''Extracts the "highest energy" oscillating component of a function using HVD.

    Uses Hilbert Vibration Decomposition (HVD) to extract the component of a
    function, given by a list of points, with a slowly-varying instantaneous
    frequency and amplitude. Some points may be lost during this process.

    Args:
        data: A list of points (2-tuples of floats) representing the function to
            decompose.
        parallel: Whether to parallelize the computation using all available
            virtual processors. Defaults to True.

    Returns:
        A tuple. The first element is a list of points on the extracted function
        at most of the same inputs as the given list. The second element is the
        residual component-- points representing differences between the
        extracted function and the given points at corresponding inputs.
        Returned lists are sorted by each point's first element.
    '''
    with multiprocessing.Pool() if parallel else nullcontext() as pool:
        data = sorted(data, key=lambda p: p[0])
        hilbert_data = hilbert(data, pool=pool)
        phase = _analytical_phase(data, hilbert_data)
        amp = _analytical_magnitude(data, hilbert_data)
        freq = [
            (p0[0], (p1[1] - p0[1]) % (2 * math.pi) / (p1[0] - p0[0]))
            for p0, p1 in zip(phase, phase[1:])
        ]
        #extracted_freq = _low_pass(freq, _ORIG_IF_LOW_PASS_CUTOFF, pool=pool)
        #extracted_freq = _filtered_moving_median(freq, _ORIG_IF_AVG_WINDOW, 5, pool=pool)
        extracted_freq = [(x, 0.02 + 0.00003 * x) for x, _ in freq]
        in_phase_proj, hb_phase_proj = _lock_in_amp(hilbert_data, extracted_freq, 230)
        _write_points('realproj', in_phase_proj)
        _write_points('imagproj', hb_phase_proj)
        in_phase_proj = _low_pass(in_phase_proj, _PROJ_LOW_PASS_CUTOFF, pool=pool)
        hb_phase_proj = _low_pass(hb_phase_proj, _PROJ_LOW_PASS_CUTOFF, pool=pool)
        extracted_amp = [(x, 2 * a) for x, a in _analytical_magnitude(in_phase_proj, hb_phase_proj)]
        extracted_phase = _analytical_phase(in_phase_proj, hb_phase_proj)
        signal = [(x, a * math.cos(p)) for (x, a), (_, p) in zip(extracted_amp, extracted_phase)]
        _write_points('raw', data)
        _write_points('hilbert', hilbert_data)
        _write_points('origip', phase)
        _write_points('origia', amp)
        _write_points('origif', freq)
        _write_points('extrif', extracted_freq)
        _write_points('lprealproj', in_phase_proj)
        _write_points('lpimagproj', hb_phase_proj)
        _write_points('extria', extracted_amp)
        _write_points('extrip', extracted_phase)
        _write_points('final', signal)
        return signal, []#[(x, orig - extracted) for (x, orig), (_, extracted) in zip(data, signal)]

def _parallelize(count, uniforms, func, pool):
    '''Runs a kernel function a given number of times, optionally in perallel.

    Runs a kernel function [count] times, passing the job number and the given
    list of uniforms to each invocation.  If given a multiprocessing.Pool, uses
    it to parallelize the operation.

    Args:
        count: The number of jobs to run.
        uniforms: The parameters to pass to each kernel invocation after the
            job number.
        func: The kernel function, taking an integer job number and the
            unpacked iterable of uniforms.
        pool: If not None, used to parallelize the mapping operation in multiple
            subprocesses.

    Returns:
        A list containing the return values of every invocation in order.
    '''
    args = [(i, *uniforms) for i in range(count)]
    if pool != None:
        return pool.starmap(func, args)
    else:
        return [func(*arg) for arg in args]

def _def_integrate(data):
    '''Integrates a signal.

    Integrates a function specified by a list of points from negative to
    positive infinity, assuming all values outside the specified range are 0.
    Simpson's rule is used unless the second of a triplet of points is too far
    from the mean input, in which case the trapezoid rule is used.

    Args:
        data: The list of points representing the function to be def_integrated.
    
    Returns:
        The value of the improper definite integral.
    '''
    integral = 0
    l = len(data)
    use_trapezoid = True
    for i in range(l - 1):
        if not use_trapezoid:
            # skip the middle point after using simpson's rule
            use_trapezoid = True
            continue
        use_trapezoid = i + 2 >= l
        p0 = data[i]
        p1 = data[i + 1]
        if not use_trapezoid:
            p2 = data[i + 2]
            use_trapezoid = abs(p1[1] * 2 - p0[0] - p2[0]) > _TRAPEZOID_THRESHOLD
        if use_trapezoid:
            integral += (p1[0] - p0[0]) * (p1[1] + p0[1]) * 0.5
        else:
            # simpsons
            integral += (p2[0] - p0[0]) / 6 * (p0[1] + 4 * p1[1] + p2[1])
    return integral

def _imp_integrate(data):
    integral = 0
    integrated = []
    for i in range(l - 1):
        p0 = data[i]
        p1 = data[i + 1]
        integrated.append(p0[0], integral)
        integral += (p1[0] - p0[0]) * (p1[1] + p0[1]) * 0.5
    return integrated

def _low_pass(data, cutoff, pool=None):
    '''Convolves a signal with the sinc filter function to achieve a low pass
    filtering.

    Args:
        data: The list of points representing the signal to filter.
        cutoff: The frequency cutoff to filter under.
        pool: If not None, used to parallelize the operation.

    Returns:
        The input points transformed by the low pass filter.
    '''
    return _parallelize(len(data), (data, cutoff), _low_pass_kernel, pool)

def _low_pass_kernel(i, data, cutoff):
    '''Evaluates and returns the low pass filtered version of a function at a point.

    Args:
        i: The datum index.
        data: The list of points.
    '''
    t = data[i][0]
    return (t, _def_integrate([(x, y * _sinc_filter(t - x, cutoff)) for x, y in data]))

def _lock_in_amp(signal, freq, window, pool=None):
    integrated_freq = _imp_integrate(freq)
    count = len(signal) - window
    in_phase = _parallelize(
        len(signal)
        (signal, window, integrated_freq, 0),
        _lock_in_amp_kernel,
        pool
    )
    quad_phase = _parallelize(
        len(signal)
        (signal, window, integrated_freq, -math.pi / 2),
        _lock_in_amp_kernel,
        pool
    )
    return [x for x in in_phase if x != None], [x for x in quad_phase if x != None]

def _lock_in_amp_kernel(i, signal, window, integrated_freq, phase):
    if rate[i] - rate[0]
    signal_slice = signal[window_slice]
    integrated_freq_slice = integrated_freq[window_slice]
    integrand = [
        (x, math.sin(2 * math.pi * freq_int + phase) * sig)
        for i, ((x, sig), (_, freq_int)) in enumerate(zip(signal_slice, integrated_freq_slice))
    ]
    return (window_slice[-1][0], 1 / signal * _def_integrate(integrand))

def _filtered_moving_avg(data, window, max_stdev_diff, pool=None):
    avgs = _moving_avg(data, window, pool)
    stdevs = _moving_stdev(data, window, pool)
    filtered = [
        (x, y if abs(y - avg) / stdev < max_stdev_diff else avg)
        for (x, y), (_, avg), (_, stdev) in zip(data, avgs, stdevs)
    ]
    return _moving_avg(filtered, window, pool)

def _moving_avg(data, window, pool=None):
    return _parallelize(len(data), (data, window), _moving_avg_kernel, pool)

def _moving_avg_kernel(i, data, window):
    return _moving_kernel(_mean, i, data, window)

def _moving_stdev(data, window, pool=None):
    return _parallelize(len(data), (data, window), _moving_stdev_kernel, pool)

def _moving_stdev_kernel(i, data, window):
    return _moving_kernel(_stdev, i, data, window)

def _moving_harmonic_avg(data, window, pool=None):
    return _parallelize(len(data), (data, window), _moving_harmonic_avg_kernel, pool)

def _moving_harmonic_avg_kernel(i, data, window):
    return _moving_kernel(_harmonic_mean, i, data, window)

def _filtered_moving_median(data, window, max_mad_diff, pool=None):
    meds = _moving_median(data, window, pool)
    mads = _moving_mad(data, window, pool)
    filtered = [
        (x, y if abs(y - med) / mad < max_mad_diff else med)
        for (x, y), (_, med), (_, mad) in zip(data, meds, mads)
    ]
    return _moving_avg(filtered, window, pool)

def _moving_median(data, window, pool=None):
    return _parallelize(len(data), (data, window), _moving_median_kernel, pool)

def _moving_median_kernel(i, data, window):
    return _moving_kernel(_median, i, data, window)

def _moving_mad(data, window, pool=None):
    return _parallelize(len(data), (data, window), _moving_mad_kernel, pool)

def _moving_mad_kernel(i, data, window):
    return _moving_kernel(_mad, i, data, window)

def _moving_kernel(func, i, data, window):
    l = data[max(0, i - window // 2):min(i + window // 2, len(data) - 1)]
    l = [x[1] for x in l]
    return (data[i][0], func(l))

def _mean(data):
    return sum(data) / len(data)

def _median(data):
    return sorted(data)[len(data) // 2]

def _harmonic_mean(data):
    return len(data) / sum([1 / x for x in data])

def _mad(data):
    m = _median(data)
    return _median([abs(x - m) for x in data])

def _stdev(data):
    l = len(data)
    m = sum(data) / l
    return math.sqrt(sum([(x - m)**2 for x in data]) / l)

def _sinc_filter(x, cutoff):
    '''Evaluates and returns the sinc filter function.

    Args:
        x: The input to the sinc filter function.
        cutoff: The frequency cutoff.
    '''
    if abs(x) < _SINC_EPSILON:
        return 2 * cutoff
    else:
        return math.sin(2 * cutoff * math.pi * x) / (math.pi * x)

def _pos_atan2(y, x):
    '''Returns math.atan2(y, x) in the range [0, 2pi] instead of [-pi, pi].'''
    return math.atan2(y, x) + (2 * math.pi if y < 0 else 0)

def _analytical_phase(real, imag):
    '''Computes the instantaneous phase of an analytical signal.

    Computes the instantaneous phase of the rotation of an analytical signal.
    The analytical signal is given as two list of points, the first being the
    real component of the signal and the second being the imaginary component
    (which is the Hilbert transform of the real component). If any points at
    corresponding positions in the real and imaginary signals don't have the
    same input value, the output is undefined.

    Args:
        real: A list of points (2-tuples of floats) describing the real
            component of the analytical signal.
        imag: A list of points (2-tuples of floats) describing the imaginary
            component of the analytical signal.

    Returns:
        A list of points describing the analytical signal's phase at
        corresponding inputs in the given lists.
    '''
    return [(x, _pos_atan2(i, r)) for (x, r), (_, i) in zip(real, imag)]

def _analytical_magnitude(real, imag):
    '''Computes the instantaneous magnitude of an analytical signal.

    Computes the instantaneous magnitude of an analytical signal.
    The analytical signal is given as two list of points, the first being the
    real component of the signal and the second being the imaginary component
    (which is the Hilbert transform of the real component). If any points at
    corresponding positions in the real and imaginary signals don't have the
    same input value, the output is undefined.

    Args:
        real: A list of points (2-tuples of floats) describing the real
            component of the analytical signal.
        imag: A list of points (2-tuples of floats) describing the imaginary
            component of the analytical signal.

    Returns:
        A list of points describing the analytical signal's magnitude at
        corresponding inputs in the given lists.
    '''
    return [(x, math.hypot(r, i)) for (x, r), (_, i) in zip(real, imag)]

def _write_points(fn, points):
    '''Dumps points to a file for testing purposes.'''
    with open(fn + '.csv', 'w') as f:
        print_points(points, file=f)

def _test_data_gen_func(x):
    '''Evaluates and returns the nonstationary square wave at the given input.'''
    return (1 + 0.003*x) * math.copysign(1, math.sin((0.02 + 0.00003*x)*x))

def _test_data_gen():
    size = 2048
    fac = 0.5
    return [
        (x * fac, _test_data_gen_func(x * fac))
        for x in range(int(size / fac))
    ]

def _run_cli():
    data = _test_data_gen()
    #print_points(data)
    print_points(hilbert_decomp(data)[0])
    with multiprocessing.Pool() as pool:
        #print_points(_low_pass(data, pool=pool))
        #print_points(hilbert(hilbert(data, pool=pool), pool=pool))
        pass

if __name__ == '__main__':
    _run_cli()
