from cli_util import print_points
import math

_HILBERT_EPSILON = pow(10, -4)
_AVG_WINDOW_SIZE = 10

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

def _extended_moving_avg(data):
    res = []
    half_winsize = math.ceil(_AVG_WINDOW_SIZE / 2)
    extended = [data[0]] * half_winsize + data + [data[-1]] * half_winsize
    for i in range(len(data)):
        res.append(sum(extended[i:i + _AVG_WINDOW_SIZE]) / _AVG_WINDOW_SIZE)
    return res

def hilbert_decomp(data):
    hilbert_data = hilbert(data)
    # instantaneous phase is atan(hilbert[x(t)] / x(t))
    phase = [
        (x, math.atan2(tfmd, orig))
        for (x, tfmd), (_, orig) in zip(hilbert_data, data)
    ]
    # instantaneous amplitude is sqrt(x^2(t) + hilbert[x(t)]^2)
    amp = [
        (x, pow(pow(orig, 2) + pow(tfmd, 2), 0.5))
        for (x, tfmd), (_, orig) in zip(hilbert_data, data)
    ]
    # instantaneous frequency is d/dt phase(t)
    freq = [
        (p0[0], (p1[1] - p0[1]) / (p1[0] - p0[0]))
        for p0, p1 in zip(phase, phase[1:])
    ]
    # hilbert decomposition supposes that the input is the sum of one signal with slow-varying
    # instantaneous amplitude and frequency and another that is hot garbage (which may actually be
    # the sum of many other signals with slow-varying properties). by smoothing the instantaneous
    # amplitude and frequency of the input, we get the properties of the first slow-varying
    # component.
    # TODO: word that better
    smooth_amp = list(zip([p[0] for p in amp], _extended_moving_avg([p[1] for p in amp])))
    smooth_freq = list(zip([p[0] for p in freq], _extended_moving_avg([p[1] for p in freq])))
    return smooth_amp
    return [], []

def _run_cli():
    freq = 0.125
    fac = 0.5
    data = [
        (x * fac, math.sin(freq * (x * fac)) + 0.25 * math.sin(freq * 4 * (x * fac)))
        for x in range(int(1000 / fac))
    ]
    print_points(data)

if __name__ == '__main__':
    _run_cli()
