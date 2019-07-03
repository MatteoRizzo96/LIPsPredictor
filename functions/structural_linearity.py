from typing import List

from functions.sliding_window import sliding_window


def structural_linearity(entries: List, inter_sc_col: int, intra_col: int, intra_long_col: int) -> List:
    """
    Computes structural linearity of each residue using a sliding window over 11 consecutive
    residues. SL is computed as inter_sc / (intra + intra_long * 4.0), where each parameter
    is relative to the sum of contacts inside each blob of residues

    :param entries: a list of list with intra, intra_long and inter_sc contacts for each residue
    :param inter_sc_col: index of column with inter_sc values
    :param intra_col: index of column with intra values
    :param intra_long_col: index of column with intra_long values
    :return: a list of values fo r structural linearity for each residue
    """

    intra = [row[intra_col] for row in entries]
    intra_long = [row[intra_long_col] for row in entries]
    inter_sc = [row[inter_sc_col] for row in entries]

    features = [[a, b, c] for a, b, c in zip(intra, intra_long, inter_sc)]
    # features = [
    #               [a=intra, b=intra_long, c=inter_sc],
    #               ...
    #            ]

    struct_lin = []
    chunks = sliding_window(features, 11, padding=True, padding_el=[0, 0, 0])

    for blob in chunks:
        intra_total, inter_total, intra_long_total = 0, 0, 0

        for res in blob:
            intra_total += res[0]
            intra_long_total += res[1]
            inter_total += res[2]

        # Final +1 avoids dividing by 0)
        blob_sl = inter_total / ((intra_total + intra_long_total * 4.0) + 1)
        struct_lin.append(blob_sl)

    return struct_lin
