from cli_util import print_points
import math
import multiprocessing

_SINC_EPSILON = pow(10, -12)
_HILBERT_EPSILON = pow(10, -12)
_LOW_PASS_CUTOFF = 0.02
_TRAPEZOID_THRESHOLD = 0.0001

def hilbert(data, pool=None):
    return _parallelize(sorted(data, key=lambda p: p[0]), _hilbert_kernel, pool)

def _hilbert_kernel(i, data):
    t = data[i][0]
    return (t, _integrate([
        (x, y / (t - x))
        for x, y in data
        if abs(t - x) > _HILBERT_EPSILON
    ]) / math.pi)

def hilbert_decomp(data):
    with multiprocessing.Pool() as pool:
        data = sorted(data, key=lambda p: p[0])
        hilbert_data = _debug_cache('cache-hilbert.pickle', data, lambda: hilbert(data, pool=pool))
        return hilbert_data, []
        phase = _analytical_phase(data, hilbert_data)
        return phase, []
        amp = _analytical_magnitude(data, hilbert_data)
        freq = [
            (p0[0], (p1[1] - p0[1]) / (p1[0] - p0[0]))
            for p0, p1 in zip(phase, phase[1:])
        ]
        extracted_freq = _low_pass(freq, pool=pool)
        return extracted_freq, []
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
    args = [(i, data) for i in range(len(data))]
    if pool != None:
        return pool.starmap(func, args)
    else:
        return [func(*arg) for arg in args]

def _integrate(data):
    #integral = 0
    #for p0, p1 in zip(data, data[1:]):
    #    integral += (p1[0] - p0[0]) * (p1[1] + p0[1])
    #return integral * 0.5
    integral = 0
    l = len(data)
    use_trapezoid = True
    for i in range(l - 1):
        if not use_trapezoid:
            # skip one iteration after using simpson's rule
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

def _sinc_filter(x):
    if abs(x) < _SINC_EPSILON:
        return 2 * _LOW_PASS_CUTOFF
    else:
        return math.sin(2 * _LOW_PASS_CUTOFF * math.pi * x) / (math.pi * x)

def _low_pass(data, pool=None):
    return _parallelize(data, _low_pass_kernel, pool)

def _sync(input_subset, data):
    inputs = set((p[0] for p in input_subset))
    return [p for p in data if p[0] in inputs]

def _low_pass_kernel(i, data):
    t = data[i][0]
    return (t, _integrate([(x, y * _sinc_filter(t - x)) for x, y in data]))

def _analytical_phase(real, imag):
    return [(x, math.atan(i / r)) for (x, r), (_, i) in zip(real, imag)]

def _analytical_magnitude(real, imag):
    return [(x, pow(r * r + i * i, 0.5)) for (x, r), (_, i) in zip(real, imag)]

def _debug_cache(fn, deps, gen):
    import pickle
    try:
        f = open(fn, 'rb')
        stored_deps, stored_data = pickle.load(f)
        f.close()
    except:
        stored_deps = object()
    if deps == stored_deps:
        return stored_data
    else:
        data = gen()
        f = open(fn, 'wb')
        pickle.dump((deps, data), f)
        f.close()
        return data

def _sel(l, i):
    return [p[i] for p in l]

def _test_data_gen_func(x):
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
