from cli_util import print_points
import math

_SINC_EPSILON = pow(10, -8)
_HILBERT_EPSILON = pow(10, -8)
_LOW_PASS_CUTOFF = 0.05

def hilbert(data):
    return [
        (t, _integrate([
            (x, y / (t - x))
            for x, y in data
            if abs(t - x) > _HILBERT_EPSILON
        ]) / math.pi)
        for (t, _) in data
    ]

def hilbert_decomp(data):
    data = sorted(data, key=lambda p: p[0])
    hilbert_data = _debug_cache('cache-hilbert.pickle', data, lambda: hilbert(data))
    phase = _analytical_phase(data, hilbert_data)
    amp = _analytical_magnitude(data, hilbert_data)
    freq = [
        (p0[0], (p1[1] - p0[1]) / (p1[0] - p0[0]))
        for p0, p1 in zip(phase, phase[1:])
    ]
    extracted_freq = _low_pass(freq)
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
    in_phase_proj = _low_pass(in_phase_proj)
    hb_phase_proj = _low_pass(hb_phase_proj)
    extracted_amp = [(x, 2 * a) for x, a in _analytical_magnitude(in_phase_proj, hb_phase_proj)]
    extracted_phase = _analytical_phase(in_phase_proj, hb_phase_proj)
    signal = [(x, a * math.cos(p)) for (x, a), (_, p) in zip(extracted_amp, extracted_phase)]
    return signal, [(x, orig - extracted) for (x, orig), (_, extracted) in zip(data, signal)]

def _integrate(data):
    integral = 0
    for p0, p1 in zip(data, data[1:]):
        integral += (p1[0] - p0[0]) * (p1[1] + p0[1])
    return integral * 0.5

def _sinc_filter(x):
    if abs(x) < _SINC_EPSILON:
        return 2 * _LOW_PASS_CUTOFF
    else:
        return math.sin(2 * _LOW_PASS_CUTOFF * math.pi * x) / (math.pi * x)

def _low_pass(data):
    return [(t, _integrate([(x, y * _sinc_filter(t - x)) for x, y in data])) for (t, _) in data]

def _analytical_phase(real, imag):
    return [(x, math.atan2(i, r)) for (x, r), (_, i) in zip(real, imag)]

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

def _test_data_gen():
    freq = 0.125
    fac = 0.5
    rider_amp = 0.25
    return [
        (x * fac, math.sin(freq * (x * fac)) + rider_amp * math.sin(freq * 8 * (x * fac)))
        for x in range(int(1000 / fac))
    ]

def _run_cli():
    data = _test_data_gen()
    #print_points(data)
    print_points(hilbert_decomp(data)[0])
    #print_points(_low_pass(data))
    #print_points(hilbert(hilbert(data)))

if __name__ == '__main__':
    _run_cli()
