from fit import poly_eval
import random

def main():
    random.seed(42)
    coeffs = (2000, -0.5, 0, 1, -2.75)
    noise_size = 4
    for i in range(100):
        x = (i - 60) * 0.75
        noise = (random.random() * 2 - 1) * 4
        print(f'{x},{poly_eval(coeffs, x) + noise}')

if __name__ == '__main__':
    main()
