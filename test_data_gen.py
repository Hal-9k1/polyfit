from fit import poly_eval
from cli_util import parse_args, print_points
import random

seed = 42
coeffs = (200, -39375, -425, 55, 1)
noise_size = 150000
step_size = 3/16/2
data_length = 400

def main():
    random.seed(seed)
    points = []
    for i in range(data_length):
        x = (i - 60) * step_size
        poly = poly_eval(coeffs, x)
        spike = (-100*(4*x - 62)*(4.2*x - 63)*(7*x - 64)*(3*x - 65) + 200000) / ((x - 15)**4 + 1)
        noise = (random.random() * 2 - 1) * noise_size
        points.append((x, poly + spike + noise))
    print_points(points)

if __name__ == '__main__':
    main()
