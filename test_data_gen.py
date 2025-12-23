from fit import poly_eval
from cli_util import parse_args
import random

def main():
    seed = 42
    coeffs = (0, -39375, -425, 55, 1)
    noise_size = 4
    data_length = 100

    named, positional = parse_args({
        'info': None,
    })
    if 'info' in named:
        print(f'Coefficients (0th degree first): {coeffs}\nNoise size: {noise_size}\n'
            + f'Data length: {data_length}')
        return

    random.seed(seed)

    for i in range(data_length):
        x = (i - 60) * 0.75
        noise = (random.random() * 2 - 1) * 4
        print(f'{x},{poly_eval(coeffs, x) + noise}')

if __name__ == '__main__':
    main()
