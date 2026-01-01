import math

_HILBERT_EPSILON = pow(10, -4)

def hilbert(data):
    tfmd = []
    half_rpi = 0.5 / math.pi
    data = sorted(data, key=lambda p: p[0])
    for p in data:
        integral = 0
        for p0, p1 in zip(data, data[1:]):
            if min(abs(p[0] - p0[0]), abs(p[0] - p1[0]), _HILBERT_EPSILON) != _HILBERT_EPSILON:
                # not sure how to evaluate a cauchy principal value numerically
                # so we're skipping the undefined cases
                continue
            dx = p1[0] - p0[0]
            y0 = p0[1] / (p[0] - p0[0])
            y1 = p1[1] / (p[0] - p1[0])
            # trapezoid rule, but multiplying by 0.5 is factored out
            integral += dx * (y0 + y1)
        tfmd.append((p[0], half_rpi * integral))
    return tfmd

def debug(msg, expr):
    print(msg)
    return expr

def hilbert_decomp(data):
    # instantaneous phase is atan(hilbert[x(t)] / x(t))
    phase = [(x, math.atan2(tfmd, orig)) for (x, tfmd), (_, orig) in zip(hilbert(data), data)]
    # instantaneous frequency is d/dt phase(t)
    freq = [(p0[0], (p1[1] - p0[1]) / (p1[0] - p0[0])) for p0, p1 in zip(phase, phase[1:])]
    return freq
    # use moving average to find slow-varying component of IF
    return [], []

def _run_cli():
    freq = 0.125
    fac = 0.5
    data = [(x * fac, math.sin(freq * (x * fac))) for x in range(int(1000 / fac))]
    for x, y in hilbert_decomp(data):
        print(f'{x},{y}')

if __name__ == '__main__':
    _run_cli()
