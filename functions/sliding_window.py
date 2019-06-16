from pprint import pprint
from typing import List, Iterable, Any
from collections import deque
from itertools import repeat

from Bio.PDB import PDBParser

from functions import inline_distance


def sliding_window(sequence: List, window_size: int, padding: bool = True, padding_el: Any = None) \
        -> Iterable:
    """
    Compute a sliding window over a list. If step is not compatible
    with size it drops the last cells. (E.g. doesn't perform padding)
    :param padding_el: The element to use as padding. Required if padding is True.
    :param padding: if True the function will return as many chunks as the elements in sequence.
        If false function will return len(sequence)-windows_size + 1 chunks
    :param sequence: list of values
    :param window_size: the size of the sliding window. If this size is greater than sequence lenght,
        window_size is adjusted to match sequence lenght.
    :return: generator of chunks of sequence
    """
    # Verify the inputs
    assert isinstance(sequence, List), "ERROR sequence must be a list."
    assert isinstance(window_size,
                      int), "ERROR window_size and step must be integers."
    # assert window_size < len(sequence), "ERROR window_size must not be larger than sequence length."
    assert (padding and padding_el is not None) or (not padding), "ERROR if padding is set to true, " \
                                                                  "padding element must be provided."

    # If there are not enough values reduce window size
    if window_size > len(sequence):
        window_size = len(sequence)
    # Pre-compute number of chunks to emit (drop last elements if size is not suitable)
    num_chunks = (len(sequence) - window_size) + 1

    if padding:
        # Update number of chunks
        num_chunks = len(sequence)

        # Compute padding for both ends of the list
        if window_size % 2 == 0:  # if size if even
            left_padding = (window_size // 2) - 1
            right_padding = window_size // 2
        else:  # if win size is odd
            left_padding = right_padding = window_size // 2

        # Convert list to deque for easier prepend and add padding to both ends
        sequence = deque(sequence)
        sequence.extendleft(repeat(padding_el, left_padding))
        sequence.extend(repeat(padding_el, right_padding))

        # print(sequence)

    # Return a generator
    for i in range(0, num_chunks, 1):
        yield list(sequence)[i:i + window_size]


# To test
if __name__ == '__main__':

    a = [1, 2, 5, 6, 7, 8, 5]
    chunks = sliding_window(a, window_size=3, padding=True, padding_el=0)

    result = []
    for chunk in chunks:
        blob_sum = 0
        for el in chunk:
            blob_sum += el
        result.append(blob_sum)

    print(result)

    # pdb_id = "1cee"
    # structure = PDBParser(QUIET=True).get_structure(pdb_id, "../dataset/entities/pdb" + pdb_id +
    # ".ent")
    # prova = list(sliding_window(structure[0]["A"], 2))
    # pprint(prova)
