from fit import poly_eval
from cli_util import parse_args
import random

def main():
    seed = 42
    coeffs = (200, -39375, -425, 55, 1)
    noise_size = 200000
    step_size = 3/16/2
    data_length = 400

    named, positional = parse_args({
        'info': None,
    })
    if 'info' in named:
        print(f'Coefficients (0th degree first): {coeffs}\nNoise size: {noise_size}\n'
            + f'Step size: {step_size}\nData length: {data_length}')
        return

    random.seed(seed)

    for i in range(data_length):
        x = (i - 60) * step_size
        noise = (random.random() * 2 - 1) * noise_size
        print(f'{x},{poly_eval(coeffs, x) + noise}')

if __name__ == '__main__':
    main()
