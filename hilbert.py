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
_LOW_PASS_CUTOFF = 0.02
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
    return _parallelize(sorted(data, key=lambda p: p[0]), _hilbert_kernel, pool)

def _hilbert_kernel(i, data):
    '''Evaluates and returns the Hilbert transform of a function at a point.

    Hilbert[x(t)] = principal value integral from -oo to oo of x(t)/(t - x) dt

    Args:
        i: The datum index.
        data: The list of points.
    '''
    t = data[i][0]
    return (t, _integrate([
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
            (p0[0], (p1[1] - p0[1]) % (2 * math.pi)) / (p1[0] - p0[0])
            for p0, p1 in zip(phase, phase[1:])
        ]
        extracted_freq = _low_pass(freq, pool=pool)
        in_phase_proj = []
        hb_phase_proj = []
        integral = 0
        for i in range(len(freq) - 1):
            x = freq[i][0]
            x_next = freq[i + 1][0]
            freq_all = freq[i][1]
            freq_all_next = freq[i + 1][1]
            amp_all = amp[i][1]
            phase_all = phase[i][1]
            freq_ref = extracted_freq[i][1]
            freq_ref_next = extracted_freq[i + 1][1]
            in_phase_proj.append(
                (x, 0.5 * amp_all * (math.cos(phase_all) + math.cos(integral + phase_all)))
            )
            hb_phase_proj.append(
                (x, 0.5 * amp_all * (math.sin(phase_all) - math.sin(integral + phase_all)))
            )
            # trapezoid rule:
            integral += (x_next - x) * (freq_all_next + freq_ref_next + freq_all + freq_ref) * 0.5
        in_phase_proj = _low_pass(in_phase_proj, pool=pool)
        hb_phase_proj = _low_pass(hb_phase_proj, pool=pool)
        extracted_amp = [(x, 2 * a) for x, a in _analytical_magnitude(in_phase_proj, hb_phase_proj)]
        extracted_phase = _analytical_phase(in_phase_proj, hb_phase_proj)
        signal = [(x, a * math.cos(p)) for (x, a), (_, p) in zip(extracted_amp, extracted_phase)]
        return signal, [(x, orig - extracted) for (x, orig), (_, extracted) in zip(data, signal)]

def _parallelize(data, func, pool):
    '''Applies a kernel function to every element of a list.

    Transforms every element of a list with a kernel function, which takes an
    index into the list and the list itself. If given a multiprocessing.Pool,
    uses it to parallelize the operation.

    Args:
        data: The list to apply the kernel to.
        func: The kernel function, taking an integer index into data and data
            itself.
        pool: If not None, used to parallelize the mapping operation in multiple
            subprocesses.

    Returns:
        The mapped list.
    '''
    args = [(i, data) for i in range(len(data))]
    if pool != None:
        return pool.starmap(func, args)
    else:
        return [func(*arg) for arg in args]

def _integrate(data):
    '''Integrates a signal.

    Integrates a function specified by a list of points from negative to
    positive infinity, assuming all values outside the specified range are 0.
    Simpson's rule is used unless the second of a triplet of points is too far
    from the mean input, in which case the trapezoid rule is used.

    Args:
        data: The list of points representing the function to be integrated.
    
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
            integral += (p2[0] - p1[0]) / 6 * (p0[1] + 4 * p1[1] + p2[1])
    return integral

def _low_pass(data, pool=None):
    '''Convolves a signal with the sinc filter function to achieve a low pass
    filtering.

    Args:
        data: The list of points representing the signal to filter.
        pool: If not None, used to parallelize the operation.

    Returns:
        The input points transformed by the low pass filter.
    '''
    return _parallelize(data, _low_pass_kernel, pool)

def _low_pass_kernel(i, data):
    '''Evaluates and returns the low pass filtered version of a function at a point.

    Args:
        i: The datum index.
        data: The list of points.
    '''
    t = data[i][0]
    return (t, _integrate([(x, y * _sinc_filter(t - x)) for x, y in data]))

def _sinc_filter(x):
    '''Evaluates and returns the sinc filter function.

    Args:
        x: The input to the sinc filter function.
    '''
    if abs(x) < _SINC_EPSILON:
        return 2 * _LOW_PASS_CUTOFF
    else:
        return math.sin(2 * _LOW_PASS_CUTOFF * math.pi * x) / (math.pi * x)

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
