'''General CLI utilities.

Contains routines for command line argument parsing, reading 2-column CSVs, and
properly formatted data output.
'''
import sys
from decimal import Decimal

def panic(msg):
    '''Prints a message to standard error and exits the program.

    Args:
        msg: The message to print to standard error.

    Returns:
        Never.
    '''
    print(msg, file=sys.stderr)
    exit(1)

def parse_args(names_to_types):
    '''Parses GNU-style long-form command line options.

    Parses options in sys.argv of the form --arg and --arg=value,
    automatically typechecking and converting values to their specified types.
    If an option value cannot be converted to the specified type, exits the
    program.
    Arguments are specified by a dictionary mapping option names to parser
    functions. Parser functions accept a string and return the parsed value or
    raise ValueError if the input string is invalid. Built-in type conversion
    functions like str and int are valid parser functions. The special value
    None, when used as a parser function, indicates that an option does not
    accept a value.
    As is GNU convention, any command line arguments following the argument '--'
    (without an option name) disables subsequent parsing of options.

    Args:
        names_to_types: A dictionary of option names to parser functions.

    Returns:
        A tuple. The first element is a dictionary of encountered option names
        to parsed values (or None if the option was not configured to accept a
        value). The second element is a list of the unparsed positional
        arguments.
    '''
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
    '''A parser function for positive integers.

    Args:
        s: The string to parse.

    Returns:
        The positive integer represented by the string in base 10.

    Raises:
        ValueError: The string does not represent a positive base 10 integer.
    '''
    return _require_positive(s, int)

def pos_float(s):
    '''A parser function for positive real numbers.

    Args:
        s: The string to parse.

    Returns:
        The positive real number represented by the string in base 10.

    Raises:
        ValueError: The string does not represent a positive base 10 real number.
    '''
    return _require_positive(s, float)

def _require_positive(s, t):
    '''A parser function wrapper that constrains the parsed value to be positive.

    Args:
        s: The string to parse.
        t: The type conversion function to use to parse the string.

    Returns:
        The positive parsed value of the string.

    Raises:
        ValueError: The type conversion function failed to parse the string, or
        the parsed value was not positive.
    '''

    val = t(s)
    if val <= 0:
        raise ValueError
    return val

def read_points_from_csv(file):
    '''Reads a numeric two-column CSV.

    Reads an open CSV file whose lines are of the format '%d,%d\\n' (where %d
    represents the base-10 representation of a real number) into a list of
    two-element tuples. If invalid data is encounered, exits the program.

    Args:
        file: The open CSV file to read points from.

    Returns:
        A list of tuples containing two floats.
    '''
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
    '''Prints formatted points to a CSV file.

    Prints a list of points (2-tuples of floats) to an open file. Tuple elements
    are comma-separated and points are newline-separated. Each float is output
    to the precision that is stored by Python.

    Args:
        points: The list of points to print.
        file: The file to print the points to.
    '''
    file.writelines(f'{Decimal(x)},{Decimal(y)}\n' for x, y in points)

def print_reals(nums, file=sys.stdout):
    '''Prints newline-separated real numbers.

    Prints a list of floats to an open file. Floats are newline-separated. Each
    float is output to the precision that is stored by Python.

    Args:
        nums: The list of numbers to print.
        file: The file to print the numbers to.

    '''
    file.writelines(f'{Decimal(x)}\n' for x in nums)
