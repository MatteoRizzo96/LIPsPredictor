import csv
import logging
import os
from typing import List, Dict

import pandas as pd
from Bio.PDB import PDBList, is_aa, DSSP, Structure, HSExposureCB
from Bio.PDB.PDBParser import PDBParser

from classes.Ring import Ring
from functions.inline_distance import inline_distance
from functions.structural_linearity import structural_linearity


class Dataset:

    def __init__(self,
                 dssp_path: str,
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
        self.__dataset_specs_path = os.path.join(self.__dataset_folder_path,
                                                 "dataset_specifications.csv")
        self.__dataset_path = os.path.join(self.__dataset_folder_path, "dataset.csv")
        self.__entities_path = os.path.join(self.__dataset_folder_path, "entities")
        self.__edges_path = os.path.join(self.__dataset_folder_path, "edges")
        self.__dssp_path = dssp_path
        self.__delete_edges = delete_edges
        self.__delete_entities = delete_entities

        # Initialize parameters
        self.__use_ring_api = use_ring_api

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

    def get_residues_info(self) -> List[Dict]:
        return self.__residues_info

    def get_features(self) -> pd.DataFrame:
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
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for dataset_specs_entry in csv_reader:
                dataset_specs.append({"pdb": dataset_specs_entry["pdb"],
                                      "chain": dataset_specs_entry["chain"],
                                      "start": int(dataset_specs_entry["start"])
                                      if dataset_specs_entry["start"] != "neg" else None,
                                      "end": int(dataset_specs_entry["end"])
                                      if dataset_specs_entry["end"] != "neg" else None,
                                      "type": dataset_specs_entry["type"]
                                      })
        return dataset_specs

    def __read_residues_info(self) -> List[Dict]:
        """
        Read the info from the dataset.

        :return: a list of info as dicts
        """

        logging.info("Reading residues info at {}...".format(self.__dataset_path))

        info = []

        with open(self.__dataset_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for dataset_entry in csv_reader:
                info.append({"pdb_id": dataset_entry["pdb_id"],
                             "chain_id": dataset_entry["chain"],
                             "residue_id": int(dataset_entry["res_seq_id"])
                             })

        return info

    def __read_features(self) -> pd.DataFrame:
        """
        Read the features from the dataset.

        :return: a list of features as lists
        """

        logging.info("Reading features at {}...".format(self.__dataset_path))

        # Loading features from dataset.csv
        return pd.read_csv(self.__dataset_path,
                           usecols=["HSE_up",
                                    "ASA",
                                    "HSE_down",
                                    "sec_struct",
                                    "contacts_ratio",
                                    "distance_from_line",
                                    "contacts_energy_ratio",
                                    "structural_linearity"])

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
                                       'contacts_energy_ratio',
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
        start = specs["start"]
        end = specs["end"]

        if start is not None:
            logging.info("Searching residues from {start} to {end} in chain {chain}"
                         .format(start=start,
                                 end=end,
                                 chain=target_chain))
        else:
            logging.info("Extracting features from {id}, chain {chain} where no LIPs are present"
                         .format(id=pdb_id,
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

        # Iterate over residues of the target chain filtering the hetero groups
        # (returns only amino acids)
        for residue in [residue for residue in model[target_chain] if is_aa(residue)]:
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

            # Secondary structure
            secondary_structure = (1 if dssp_res[2] != '-' else 0) if dssp_res else None

            # Contacts ratios
            contacts_ratio = ring_features.get_contacts_ratio(seq_id, target_chain)
            contacts_energy_ratio = ring_features.get_contacts_energy_ratio(seq_id, target_chain)

            # Contacts features (for structural linearity)
            contacts_inter_sc = ring_features.get_contacts_inter_sc(seq_id, target_chain)
            contacts_intra_long = ring_features.get_contacts_intra_long(seq_id, target_chain)
            contacts_intra = ring_features.get_contacts_intra(seq_id, target_chain)

            contacts.append([contacts_inter_sc, contacts_intra, contacts_intra_long])

            # Calculate the label
            is_lip = 1 if start is not None and start <= seq_id <= end else 0

            entries.append([pdb_id,
                            target_chain,
                            seq_id,
                            exp_up,
                            exp_down,
                            asa,
                            secondary_structure,
                            contacts_ratio,
                            contacts_energy_ratio,
                            distances[residue],
                            is_lip])

        # Compute structural linearity (0, 1, 2 tell the positions of required values)
        struct_lin = structural_linearity(contacts, 0, 1, 2)

        # Append struct lin as second-last element of each entry
        for entry, sl in zip(entries, struct_lin):
            entry.insert(-1, sl)

        logging.info("Features for protein {} and chain {} extracted".format(pdb_id, target_chain))

        return entries

    def __clear_dataset(self):
        """
        Delete alla data concerning the previous dataset,
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
