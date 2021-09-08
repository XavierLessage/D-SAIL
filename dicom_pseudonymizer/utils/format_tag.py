
'''
Utility for printing the tags in the original hex format.

This code was taken and adapted from https://github.com/KitwareMedical/dicom-anonymizer
'''

def hex_to_string(x: hex):
    '''
    Convert a tag number to it's original hex string.
    E.g. if a tag has the hex number 0x0008, it becomes 8,
    and we then convert it back to 0x0008 (as a string).

    Parameters
    ----------
    x : hex
        The hex number to be converted.

    Returns
    -------
    s : str
        The hex tag converted to hex number string.
    '''
    x = str(hex(x))
    left = x[:2]
    right = x[2:]
    num_zeroes = 4 - len(right)
    return left + ('0'*num_zeroes) + right

def tag_to_hex_strings(tag: tuple):
    '''
    Convert a tag tuple to a tuple of full hex number strings.

    E.g. (0x0008, 0x0010) is evaluated as (8, 16) by python. So
    we convert it back to a string '(0x0008, 0x0010)' for pretty printing.

    Parameters
    ----------
    tag : tuple
        The tuple to be converted from hex numbers to hex number strings.

    Returns
    -------
    s : tuple
        The hex tag converted to hex number string.
    '''
    return tuple([hex_to_string(tag_element) for tag_element in tag])
