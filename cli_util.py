import sys
from decimal import Decimal

def panic(msg):
    print(msg, file=sys.stderr)
    exit(1)

def parse_args(names_to_types):
    named = {}
    positional = []
    can_parse_named = True
    for arg in sys.argv[1:]:
        parts = arg.split('=')
        if parts[0].startswith('--') and can_parse_named:
            key = parts[0][2:]
            if key in names_to_types:
                if len(parts) == 2:
                    if names_to_types[key] == None:
                        panic(f'Argument {key} does not accept a value')
                    try:
                        named[key] = names_to_types[key](parts[1])
                    except ValueError:
                        panic(f'Invalid {key}: {parts[1]}')
                elif names_to_types[key] == None:
                    named[key] = None
                else:
                    panic(f'Argument missing value {arg}')
            elif parts[0] == '--':
                can_parse_named = False
            else:
                panic(f'Malformed argument {arg}')
        else:
            positional.append(arg)
    return named, positional

def pos_int(s):
    return _require_positive(s, int)

def pos_float(s):
    return _require_positive(s, float)

def _require_positive(s, t):
    val = t(s)
    if val <= 0:
        raise ValueError
    return val

def read_points_from_csv(file):
    points = []
    for line in file.readlines():
        try:
            point = tuple(float(x) for x in line.split(','))
        except ValueError:
            panic('Found non-numeric data')
        if len(point) != 2:
            panic('Too many fields for data point')
        points.append(point)
    return points

def print_points(points, file=sys.stdout):
    file.writelines(f'{Decimal(x)},{Decimal(y)}\n' for x, y in points)
