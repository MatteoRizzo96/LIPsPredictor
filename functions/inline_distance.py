from pprint import pprint

import numpy as np
from Bio.PDB import is_aa, Chain, PDBParser


def dist(a, b):
    return np.linalg.norm(a - b)


def inline_distance(chain: Chain):
    residues_distance = dict()

    # Filter out the residues that are not AA from the chain
    chain_only_aa = [residue for residue in chain if is_aa(residue)]
    # Index of the "window"
    sequence_index = 0
    # Index of the first element of the "window"
    start = 0
    # Index of the last element + 1 of the "window"
    end = 3

    # Construct the first window from the start index to the end - 1
    window_of_chain = list(chain_only_aa[start:end])

    # Initialize the previous sum of distances from the line constructed in the window
    previous_sum_of_distances = 0

    # Continue compute new windows and distances until reaching the end of the chain
    while end - 1 < len(chain_only_aa):
        # Initialize the sum of distances for the new window
        sum_of_distances = 0.0
        # List of all the coordinates of the CA atoms for every AA in the window
        ca_atoms_coordinates = []
        distances_buffer = dict()
        for residue in window_of_chain:
            # Initialize the list of distances (one for each time the residue is in a window) of a residue to the line
            if residue not in residues_distance.keys():
                residues_distance[residue] = []

            # Find the CA in the residue and get the coordinates of it
            ca_coordinates = np.ndarray([])
            found_ca = False
            for atom in residue:
                if atom.name == "CA" and not found_ca:
                    ca_coordinates = atom.get_coord()
                    found_ca = True

            # Apparently there are some AA that are composed by only one atom that is not a CA
            if found_ca:
                ca_atoms_coordinates.append(ca_coordinates)
            else:
                ca_atoms_coordinates.append(np.array([0., 0., 0.]))

        # If the window is composed by at least 6 residue, try to skew the line in a more "linear way from the start
        # to the end of the window
        if len(ca_atoms_coordinates) > 5:
            x = (ca_atoms_coordinates[0] + ca_atoms_coordinates[1] + ca_atoms_coordinates[2]) / 3
            y = (ca_atoms_coordinates[-1] + ca_atoms_coordinates[-2] + ca_atoms_coordinates[-3]) / 3
        else:
            x = ca_atoms_coordinates[0]
            y = ca_atoms_coordinates[-1]

        # Calculate the distances for each residue from the line drawn from the first to the last CA of the window
        for point, residue in zip(ca_atoms_coordinates, window_of_chain):
            ap = point - x
            ab = y - x
            if np.dot(ap, ab) <= 0.0:
                result = x
            elif np.dot(point - y, ab) >= 0.0:
                result = y
            else:
                result = x + np.dot(ap, ab) / np.dot(ab, ab) * ab
            distance = dist(point, result)
            distances_buffer[residue] = distance
            sum_of_distances += distance

        # The window is enlarged only if the last residue that was added to the window modify the sum of distances for
        # a maximum of 4.6 angstrom
        factor = previous_sum_of_distances + 4.6

        # If the sum is too far from the previous, compute a new window, with start = end and width 3
        if previous_sum_of_distances != 0 and sum_of_distances > factor:
            previous_sum_of_distances = 0
            start = end - 1
            end = start + 3

            # If the remaining residues of the chain are less than 3, the assign a distance=0 and a sequence length=1
            # to the remaining residues
            if end >= len(chain_only_aa):
                for residue in chain_only_aa[start:]:
                    sequence_index += 1
                    residues_distance[residue] = [(0.0, 1, sequence_index)]
            else:
                # Move to the next window and move forward the sequence index
                window_of_chain = list(chain_only_aa[start:end])
                sequence_index += 1
        else:
            # Enlarge this window, save the current sum of distances and append the tuple
            # (distance from line; length of window; index of the window) to the results of each residue.
            end += 1
            window_of_chain = list(chain_only_aa[start:end])
            previous_sum_of_distances = sum_of_distances
            for key in distances_buffer.keys():
                residues_distance[key].append((distances_buffer[key], len(window_of_chain) - 1, sequence_index))

    # Get the index of the window with the maximum length of the window for each residue
    index = []
    for k, v in residues_distance.items():
        index.append(max(v, key=lambda x: x[1])[2])

    # Compute the results, which for each residue, is the distance from the line drawn from the first residue to the
    # last of the window of maximum size, multiplied for the length of this window.
    result = dict()
    for k, v in residues_distance.items():
        max_sequence = max(v, key=lambda x: x[1])
        if max_sequence[2] in index:
            result[k] = max_sequence[0] * max_sequence[1] if max_sequence[0] != 0.0 else 1.0 * max_sequence[1]

    return result


if __name__ == "__main__":
    pdb_id = "1un0"
    structure = PDBParser(QUIET=True).get_structure(pdb_id, "../dataset/entities/pdb" + pdb_id + ".ent")
    target_chain = "A"
    distances = inline_distance(structure[0][target_chain])
