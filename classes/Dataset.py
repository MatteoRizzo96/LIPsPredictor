import csv
import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from Bio.PDB import DSSP, HSExposureCB, PDBList, Structure, is_aa
from Bio.PDB.PDBParser import PDBParser
from pandas import DataFrame
from sklearn.utils import shuffle

from classes.Ring import Ring
from functions.inline_distance import inline_distance
from functions.structural_linearity import structural_linearity


class Dataset:
    def __init__(self,
                 dssp_path: str,
                 specs_file_name: str = "dataset_specifications.csv",
                 dataset_file_name: str = "dataset.csv",
                 rewrite: bool = True,
                 use_ring_api=False,
                 delete_edges=False,
                 delete_entities=False):
        """
        Build the dataset from the specifications in "dataset/dataset_specifications.csv".
        If the rewrite flag is True, the existing dataset will be deleted and regenerated.

        :param dssp_path: the path to the dssp executable
        :param rewrite: boolean flag to regenerate the whole dataset
        :param use_ring_api: boolean flag to use RING APIs or executable
        :param delete_edges: boolean flag to delete all RING edges files
        :param delete_entities: boolean flag to delete all RING entities files
        """

        # Initialize paths
        self.__dataset_folder_path = os.path.join(os.getcwd(), "dataset")
        self.__dataset_specs_path = os.path.join(self.__dataset_folder_path, specs_file_name)
        self.__dataset_path = os.path.join(self.__dataset_folder_path, dataset_file_name)
        self.__entities_path = os.path.join(self.__dataset_folder_path, "entities")
        self.__edges_path = os.path.join(self.__dataset_folder_path, "edges")
        self.__dssp_path = dssp_path
        self.__delete_edges = delete_edges
        self.__delete_entities = delete_entities

        # Initialize parameters
        self.__use_ring_api = use_ring_api

        if dataset_file_name == "test_set.csv" and not os.path.isfile(
                os.path.join(self.__dataset_folder_path, "test_set_specifications.csv")):
            self.__reorganize_test_set_specs()

        if rewrite or (not rewrite and not os.path.isfile(self.__dataset_path)):
            logging.info("Generating dataset...")
            self.__clear_dataset()
            self.__build_dataset()
        else:
            logging.info("Dataset already exists")

        # Initialize info
        self.__residues_info = self.__read_residues_info()

        # Initialize the features
        self.__features = self.__read_features()

        # Initialize the labels
        self.__labels = self.__read_labels()

    def get_residues_info(self) -> DataFrame:
        return self.__residues_info

    def shuffle_dataset(self):
        df = shuffle(self.get_dataset())
        self.__labels = list(df.pop('is_lip'))
        self.__features = df

    def get_features(self, best_features: List[str] = "all") -> pd.DataFrame:
        """
        Return all the best features of the dataset.

        :param best_features: some selected features obtain via features selection
        :return: a DataFrame of features
        """

        if best_features != "all":
            for feature in self.__features:
                if feature not in best_features:
                    self.__features.drop(feature, axis=1, inplace=True)

        return self.__features

    def get_features_names(self) -> List[str]:
        return [feature_name for feature_name in self.__features]

    def get_labels(self) -> List[float]:
        return self.__labels

    def get_dataset(self) -> pd.DataFrame:
        return self.__features.assign(is_lip=self.__labels)

    def update_features(self, features: pd.DataFrame):
        """
        Update the features of the dataset with the given set of features.

        :param features: a new set of features for the dataset
        """

        self.__features = features

    def balance(self, balance_ratio: int = 50):
        """
        Approximately balance the percentage of positive and negative examples
        based on the given ratio of positive examples.

        :param balance_ratio: a ratio of positive examples to balance the dataset
        """

        num_labels = len(self.__labels)
        num_positive = self.__labels.count(1)
        positive_ratio = num_positive / num_labels * 100

        logging.info("The dataset has {pr}% of positive examples out of {nl} entries before balancing"
                     .format(pr=positive_ratio, nl=num_labels))

        logging.info("Balancing dataset for positive examples ratio approximately equal to {}%"
                     .format(balance_ratio))

        # If the dataset needs a greater percentage of positive examples,
        # the number of negative examples must be lowered
        if balance_ratio > positive_ratio:

            while positive_ratio < balance_ratio:
                # Remove a negative example
                i = self.__labels.index(0)
                del self.__labels[i]
                self.__features.drop(self.__features.index[i], inplace=True)

                # Recalculate the ratio of positive examples
                num_labels -= 1
                positive_ratio = num_positive / num_labels * 100

        # If the dataset needs a lower percentage of positive examples
        # the number of positive examples must be lowered
        else:

            while positive_ratio > balance_ratio:
                # Remove a positive example
                i = self.__labels.index(1)
                del self.__labels[i]
                self.__features.drop(self.__features.index[i], inplace=True)

                # Recalculate the ratio of positive examples
                num_labels -= 1
                num_positive -= 1
                positive_ratio = num_positive / num_labels * 100

        logging.info("The dataset has {pr}% of positive examples out of {nl} entries after balancing"
                     .format(pr=positive_ratio, nl=num_labels))

        logging.info("Dataset balanced successfully!")

    def __reorganize_test_set_specs(self):

        logging.info("Reorganizing test set specifications...")

        entries = []

        with open(self.__dataset_specs_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            pdbl = PDBList()

            for dataset_specs_entry in csv_reader:

                pdb_id = dataset_specs_entry["pdb"]
                target_chain = dataset_specs_entry["chain"]
                start = "neg" if dataset_specs_entry["start"] == "#" else int(dataset_specs_entry["start"])
                end = "neg" if dataset_specs_entry["start"] == "#" else int(dataset_specs_entry["end"])

                if start != "neg":
                    pdbl.retrieve_pdb_file(pdb_id, pdir=self.__entities_path, file_format='pdb')

                    # Load the structure from the ent file
                    structure = PDBParser(QUIET=True).get_structure(pdb_id,
                                                                    os.path.join(self.__entities_path,
                                                                                 "pdb{}.ent".format(pdb_id)))
                    # Initialize a model
                    model = structure[0]

                    # Initialize the list of amino acids filtering hetero groups
                    residues = [residue for residue in model[target_chain] if is_aa(residue)]

                    # Initialize the start of the LIP residues sequence
                    _, seq_id_start, _ = residues[start - 1].id
                    start = seq_id_start

                    # Initialize the end of the LIP residues sequence
                    _, seq_id_end, _ = residues[end - 1].id
                    end = seq_id_end

                entries.append([
                    pdb_id,
                    target_chain,
                    start,
                    end
                ])

            df = pd.DataFrame(entries, columns=["pdb", "chain", "start", "end"])
            df.to_csv(os.path.join(self.__dataset_folder_path, "test_set_specifications.csv"), header=True)
            self.__dataset_specs_path = os.path.join(self.__dataset_folder_path, "test_set_specifications.csv")

    def __build_dataset(self):
        """
        Generate a new dataset in dataset/dataset.csv from
        the dataset specifications in dataset/dataset_specifications.csv.
        """

        # Read the dataset specifications
        dataset_specs = self.__read_specifications()

        pdbl = PDBList()

        # Iterate over the dataset specifications
        for specs in dataset_specs:
            # Get the pdb id of the current protein
            pdb_id = specs["pdb"]

            # Save dataset proteins to dataset/entities/pdbXXXX.ent files
            pdbl.retrieve_pdb_file(pdb_id, pdir=self.__entities_path, file_format='pdb')

            # Load the structure from the ent file
            structure = PDBParser(QUIET=True).get_structure(pdb_id,
                                                            os.path.join(self.__entities_path,
                                                                         "pdb{}.ent".format(pdb_id)))

            # Write the dataset entries for the current specs
            self.__write_entries(self.__calculate_entries_data(structure, specs))

        # If RING APIs were used, delete al zip files downloaded
        if self.__use_ring_api:
            for item in os.listdir(os.getcwd()):
                if item.endswith(".zip"):
                    os.remove(os.path.join(os.getcwd(), item))

        logging.info('Dataset generated successfully!')

    def __read_specifications(self) -> List[Dict]:
        """
        Read the dataset specifications from the specifications file "dataset/dataset_specifications.csv".

        :return: a list of specifications as dicts
        """

        logging.info("Reading dataset specifications at {}...".format(self.__dataset_specs_path))

        dataset_specs = []

        with open(self.__dataset_specs_path) as csv_file:

            csv_reader = list(csv.DictReader(csv_file, delimiter=','))
            i = 0

            while i < len(csv_reader):

                # Get the current row
                dataset_specs_entry = csv_reader[i]

                # Get the current row id formed by  pbd_id + chain_id
                dataset_specs_id = dataset_specs_entry["pdb"] + dataset_specs_entry["chain"]
                current_id = dataset_specs_id

                # Initialize a list of LIP ranges
                ranges = []

                # While the same chain is being considered
                while current_id == dataset_specs_id:

                    # Append the current LIP range to the list
                    ranges.append({
                        "start": int(csv_reader[i]["start"])
                        if csv_reader[i]["start"] != "neg" and csv_reader[i]["start"] != "#" else None,
                        "end": int(csv_reader[i]["end"])
                        if csv_reader[i]["start"] != "neg" and csv_reader[i]["start"] != "#" else None
                    })

                    # Increment the index and calculate the next id
                    i += 1

                    if i < len(csv_reader):
                        current_id = csv_reader[i]["pdb"] + csv_reader[i]["chain"]
                    else:
                        break

                dataset_specs.append({
                    "pdb": dataset_specs_entry["pdb"],
                    "chain": dataset_specs_entry["chain"],
                    "ranges": ranges
                })

        return dataset_specs

    def __read_residues_info(self) -> DataFrame:
        """
        Read the info from the dataset.

        :return: a list of info as dicts
        """

        logging.info("Reading residues info at {}...".format(self.__dataset_path))

        return pd.read_csv(self.__dataset_path, usecols=['pdb_id', 'chain', 'res_seq_id'])

    def __read_features(self) -> pd.DataFrame:
        """
        Read the features from the dataset.

        :return: a list of features as lists
        """

        logging.info("Reading features at {}...".format(self.__dataset_path))

        # Loading features from dataset.csv
        df = pd.read_csv(self.__dataset_path,
                         usecols=['HSE_up',
                                  'HSE_down',
                                  'ASA',
                                  'sec_struct',
                                  'contacts_ratio',
                                  'contacts_intra',
                                  'contacts_inter',
                                  'contacts_energy_ratio',
                                  'amino_type',
                                  'phi',
                                  'psi',
                                  'chain_len',
                                  'nho1relidx',
                                  'nho1energy',
                                  'onh1relidx',
                                  'onh1energy',
                                  'nho2relidx',
                                  'nho2energy',
                                  'onh2relidx',
                                  'onh2energy',
                                  'distance_from_line',
                                  'structural_linearity'],
                         na_values='-')

        df['sec_struct'] = df['sec_struct'].replace(np.nan, "struct_-")
        df['sec_struct'] = df['sec_struct'].replace('B', "struct_B")
        df['sec_struct'] = df['sec_struct'].replace('S', "struct_S")
        df['sec_struct'] = df['sec_struct'].replace('T', "struct_T")
        df['sec_struct'] = df['sec_struct'].replace('E', "struct_E")
        df['sec_struct'] = df['sec_struct'].replace('H', "struct_H")
        df['sec_struct'] = df['sec_struct'].replace('I', "struct_I")
        df['sec_struct'] = df['sec_struct'].replace('G', "struct_G")

        return df

    def __read_labels(self) -> List[float]:
        """
        Read the labels from the dataset.

        :return: a list of labels
        """

        logging.info("Reading labels at {}...".format(self.__dataset_path))

        with open(self.__dataset_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            return [float(dataset_entry["is_lip"]) for dataset_entry in csv_reader]

    def __write_entries(self, entries: List):
        """
        Write the given entries to the dataset file.
        If a dataset file already exists, append the new entries.

        :param entries: the entries to be written
        """

        logging.info("Writing entries at {}...".format(self.__dataset_path))

        if os.path.isfile(self.__dataset_path):
            df = pd.DataFrame(entries)
            df.to_csv(self.__dataset_path, mode='a', header=False)
        else:
            df = pd.DataFrame(entries,
                              columns=['pdb_id',
                                       'chain',
                                       'res_seq_id',
                                       'HSE_up',
                                       'HSE_down',
                                       'ASA',
                                       'sec_struct',
                                       'contacts_ratio',
                                       'contacts_intra',
                                       'contacts_inter',
                                       'contacts_energy_ratio',
                                       'amino_type',
                                       'phi',
                                       'psi',
                                       'chain_len',
                                       'nho1relidx',
                                       'nho1energy',
                                       'onh1relidx',
                                       'onh1energy',
                                       'nho2relidx',
                                       'nho2energy',
                                       'onh2relidx',
                                       'onh2energy',
                                       'distance_from_line',
                                       'structural_linearity',
                                       'is_lip'])

            df.to_csv(self.__dataset_path, header=True)

    def __calculate_entries_data(self, structure: Structure, specs: Dict) -> List:
        """
        Calculate the set of dataset entries for the current specifications.
        Each entry corresponds to a residue in the specs and is characterized by:
        - Residue information
        - Residue features
        - Residue label

        :param structure: a protein structure
        :param specs: the current specifications
        :return: a list of entries data
        """

        # Get the specs data from the current specs
        pdb_id = specs["pdb"]
        target_chain = specs["chain"]
        ranges = specs["ranges"]
        no_lip = True if ranges[0]["start"] is None else False

        if no_lip:
            logging.info("Extracting features from {id} chain {chain}, where no LIPs are present"
                         .format(id=pdb_id,
                                 chain=target_chain))
        else:
            for r in ranges:
                logging.info("Searching LIP residues from {start} to {end} in chain {chain}"
                             .format(start=r["start"],
                                     end=r["end"],
                                     chain=target_chain))

        # Create an empty entries list
        entries = []

        # Create an empty contacts list
        contacts = []

        # Initialize a model
        model = structure[0]

        # Initialize a DSSP for the current model
        dssp = dict(DSSP(model,
                         os.path.join(self.__entities_path, "pdb{}.ent".format(pdb_id)),
                         dssp=self.__dssp_path))

        # Initialize RING to calculate contacts ratios
        ring = Ring(pdb_id, use_api=self.__use_ring_api)
        logging.info("Getting Ring features for protein {}".format(pdb_id))
        ring_features = ring.get_features()

        # Initialize HSE to calculate surface exposure
        hse = HSExposureCB(model)

        # Compute the distance of each CA from the line generated between
        # the starting residue of LIP and the last one
        logging.info(
            "Getting the Inline Distance feature for protein {} chain {}".format(pdb_id, target_chain))
        distances = inline_distance(model[target_chain])

        residues = [residue for residue in model[target_chain] if is_aa(residue)]

        # Iterate over residues of the target chain filtering the hetero groups (returns only amino acids)
        for residue in residues:
            # Residue ID structure:
            #   - hetero flag
            #   - sequence id
            #   - insertion code

            # Get the residue sequence id
            _, seq_id, _ = residue.id

            # Calculate the features for the current residue (entry)

            # Accessible Surface Area
            dssp_res = dssp.get((target_chain, residue.id))
            asa = float(dssp_res[3]) if dssp_res else None

            # Half sphere exposure
            exp_up = hse[(target_chain, residue.id)][0] if residue.id[0] == " " and (
                target_chain, residue.id) in hse.keys() else None
            exp_down = hse[(target_chain, residue.id)][1] if residue.id[0] == " " and (
                target_chain, residue.id) in hse.keys() else None

            # Amino type
            amino_type = dssp_res[1] if dssp_res else '-'

            # Secondary structure
            secondary_structure = dssp_res[2] if dssp_res else '-'

            # Phi and psi
            phi = dssp_res[4] if dssp_res else None
            psi = dssp_res[5] if dssp_res else None

            # Energies
            nho1relidx = dssp_res[6] if dssp_res else None
            nho1energy = dssp_res[7] if dssp_res else None
            onh1relidx = dssp_res[8] if dssp_res else None
            onh1energy = dssp_res[9] if dssp_res else None
            nho2relidx = dssp_res[10] if dssp_res else None
            nho2energy = dssp_res[11] if dssp_res else None
            onh2relidx = dssp_res[12] if dssp_res else None
            onh2energy = dssp_res[13] if dssp_res else None

            # Chain length
            chain_len = len(residues)

            # Contacts ratios
            contacts_ratio = ring_features.get_contacts_ratio(seq_id, target_chain)
            contacts_energy_ratio = ring_features.get_contacts_energy_ratio(seq_id, target_chain)

            # Contacts features (for structural linearity)
            contacts_inter_sc = ring_features.get_contacts_inter_sc(seq_id, target_chain)
            contacts_intra_long = ring_features.get_contacts_intra_long(seq_id, target_chain)
            contacts_intra = ring_features.get_contacts_intra(seq_id, target_chain)
            contacts_inter = ring_features.get_contacts_inter(seq_id, target_chain)

            contacts.append([contacts_inter_sc, contacts_intra, contacts_intra_long])

            # Calculate the label
            is_lip = 0

            # If the residue can be LIP because some range has been calculated for the current chain
            if not no_lip:

                # The index spans the ranges
                idx = 0

                # While there are ranges to be checked
                while is_lip == 0 and idx < len(ranges):
                    if ranges[idx]["start"] <= seq_id <= ranges[idx]["end"]:
                        is_lip = 1
                    else:
                        idx += 1

            entries.append([pdb_id,
                            target_chain,
                            seq_id,
                            exp_up,
                            exp_down,
                            asa,
                            secondary_structure,
                            contacts_ratio,
                            contacts_intra,
                            contacts_inter,
                            contacts_energy_ratio,
                            amino_type,
                            phi,
                            psi,
                            chain_len,
                            nho1relidx,
                            nho1energy,
                            onh1relidx,
                            onh1energy,
                            nho2relidx,
                            nho2energy,
                            onh2relidx,
                            onh2energy,
                            distances[residue],
                            is_lip])

        # Compute structural linearity (0, 1, 2 tell the positions of required values)
        struct_lin = structural_linearity(contacts, 0, 1, 2)

        # Append structural linearity as second-last element of each entry
        for entry, sl in zip(entries, struct_lin):
            entry.insert(-1, sl)

        logging.info("Features for protein {} and chain {} extracted".format(pdb_id, target_chain))

        return entries

    def __clear_dataset(self):
        """
        Delete all data concerning the previous dataset,
        possibly including entities and edges files.
        """

        # Delete the previous dataset file
        if os.path.isfile(self.__dataset_path):
            logging.info("Deleting old dataset file...")
            os.remove(self.__dataset_path)
            logging.info("Old dataset file deleted successfully!")

        # Delete all entities files
        if os.path.isdir(self.__entities_path) and self.__delete_entities:
            logging.info("Deleting old entities files...")
            for file in os.listdir(self.__entities_path):
                file_path = os.path.join(self.__entities_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
            logging.info("Old entities files deleted successfully!")

        # Delete all edges files
        if os.path.isdir(self.__edges_path) and self.__delete_edges:
            logging.info("Deleting old edges files...")
            for file in os.listdir(self.__edges_path):
                file_path = os.path.join(self.__edges_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
            logging.info("Old edges files deleted successfully!")
