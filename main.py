import sys

def panic(msg):
    print(msg, file=sys.stderr)
    exit(1)

def main():
    if len(sys.argv) < 2:
        panic('Missing polynomial degree')
    try:
        degree = int(sys.argv[2])
    except ValueError:
        panic('Invalid polynomial degree')
    if len(sys.argv) > 3:
        panic('Too many arguments')
    if len(sys.argv) == 3:
        try:
            file = open(sys.argv[2], 'r')
        except FileNotFoundError:
            panic('Failed to open input file')
    else:
        file = sys.stdin
    points = []
    for line in file.readlines():
        try:
            point = (float(x) for x in line.split(','))
        except ValueError:
            panic('Found non-numeric data')
        if len(point) != 2:
            panic('Too many fields for data point')
        points.append(point)
    
    for coeff in fit(degree, points):
        print(coeff)

if __name__ == '__main__':
    main()
