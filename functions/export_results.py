import os

from Bio.PDB import is_aa

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
     'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
     'TPO': 'T', 'MSE': 'M'}


def export_results(results: list, structure, path):
    filename = "{}/out_{}.txt".format(path, structure.id)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w+') as f:

        f.write(">{}\n".format(structure.id))
        i = 0

        for chain in structure[0]:

            aa_residue = [residue for residue in chain if is_aa(residue)]

            for residue in aa_residue:
                f.write("0/{}/{}/{}/{} {:.3f} {}\n"
                        .format(chain.id,
                                residue.id[1],
                                residue.id[2].replace(" ", ""),
                                d[residue.resname] if residue.resname in d.keys() else residue.resname[0],
                                results[i][1],
                                1 if results[i][1] >= 0.5 else 0))
                i += 1
