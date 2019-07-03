import logging
import os
from typing import List

import numpy as np
import pandas as pd
from Bio.PDB import DSSP, HSExposureCB, PDBList, Structure, is_aa
from Bio.PDB.PDBParser import PDBParser
from pandas import DataFrame

from classes.Ring import Ring
from functions.inline_distance import inline_distance
from functions.structural_linearity import structural_linearity


class Example:

    def __init__(self, pdb_id: str, dssp_path: str, use_ring_api=False, delete_previous_example=False):
        """
        Build an input example.

        :param dssp_path: the path to the dssp executable
        :param use_ring_api: boolean flag to use RING APIs or executable
        :param delete_previous_example: boolean flag to delete previous example data
        """

        logging.info("Initializing the features of the input example...")

        # Initialize paths
        self.__example_folder_path = os.path.join(os.getcwd(), "input")
        self.__example_path = os.path.join(self.__example_folder_path, "example.csv")
        self.__entities_path = os.path.join(self.__example_folder_path, "entities")
        self.__edges_path = os.path.join(self.__example_folder_path, "edges")
        self.__dssp_path = dssp_path

        # Initialize parameters
        self.__pdb_id = pdb_id
        self.__use_ring_api = use_ring_api

        if delete_previous_example or (not delete_previous_example and not os.path.isfile(self.__example_path)):
            # Delete old example data
            self.__clear_example()

            # Initialize structure
            pdbl = PDBList()
            pdbl.retrieve_pdb_file(pdb_id, pdir=self.__entities_path, file_format='pdb')
            self.__structure = PDBParser(QUIET=True).get_structure(self.__pdb_id,
                                                                   os.path.join(self.__entities_path,
                                                                                "pdb{}.ent".format(self.__pdb_id)))

            # Write examples features
            self.__write_entries(self.__calculate_entries())

            # If RING APIs were used, delete al zip files downloaded
            if self.__use_ring_api:
                for item in os.listdir(os.getcwd()):
                    if item.endswith(".zip"):
                        os.remove(os.path.join(os.getcwd(), item))

        # Set residues info
        self.__residues_info = self.__read_residues_info()

        # Set example features
        self.__features = self.__read_features()

        logging.info("Features of the input example successfully initialized!")

    def get_features(self, best_features: List[str] = "all") -> pd.DataFrame:
        """
        Return all the best features of the example.

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

    def get_structure(self) -> Structure:
        return self.__structure

    def update_features(self, features: pd.DataFrame):
        """
        Update the features of the example with the given set of features.

        :param features: a new set of features for the example
        """

        self.__features = features

    def __calculate_entries(self) -> List:
        """
        Calculate the set of entries for the features of the example.

        :return: a list of entries data
        """

        # # Save dataset proteins to dataset/entities/pdbXXXX.ent files
        # pdbl = PDBList()
        # pdbl.retrieve_pdb_file(self.__pdb_id, pdir=self.__entities_path, file_format='pdb')

        # Load the structure from the ent file
        structure = self.__structure

        # Create a model
        model = structure[0]

        # Create a dssp for the current model
        dssp = dict(DSSP(model,
                         os.path.join(self.__entities_path, "pdb{}.ent".format(self.__pdb_id)),
                         dssp=self.__dssp_path))

        # Initialize RING to calculate contacts ratios
        ring = Ring(self.__pdb_id, use_api=self.__use_ring_api, working_dir="input")
        ring_features = ring.get_features()

        # Init hse to calculate surface exposure
        hse = HSExposureCB(model)

        # Init the list of entries to be returned
        entries = []

        # Iterate over residues of the target chain filtering the hetero groups
        # (returns only amino acids)
        for chain in model:

            # Init an empty list of entries for the current chain
            chain_entries = []

            # Init an empty list of contacts for the current chain
            contacts = []

            # Compute the distance of each CA from the line generated
            # between the starting residue of LIP and the last one
            distances = inline_distance(chain)

            residues = [residue for residue in chain if is_aa(residue)]

            for residue in residues:
                # Residue ID structure:
                #   - hetero flag
                #   - sequence id
                #   - insertion code

                # Get the residue sequence id
                _, seq_id, _ = residue.id

                # Calculate the features for the current residue (entry)

                # Accessible Surface Area
                dssp_res = dssp.get((chain.id, residue.id))
                asa = float(dssp_res[3]) if dssp_res else None

                # Half sphere exposure
                exp_up = hse[(chain.id, residue.id)][0] if residue.id[0] == " " and (
                    chain.id, residue.id) in hse.keys() else None
                exp_down = hse[(chain.id, residue.id)][1] if residue.id[0] == " " and (
                    chain.id, residue.id) in hse.keys() else None

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
                contacts_ratio = ring_features.get_contacts_ratio(seq_id, chain.id)
                contacts_energy_ratio = ring_features.get_contacts_energy_ratio(seq_id, chain.id)

                # Contacts feature (for structural linearity)
                contacts_inter_sc = ring_features.get_contacts_inter_sc(seq_id, chain.id)
                contacts_intra_long = ring_features.get_contacts_intra_long(seq_id, chain.id)
                contacts_intra = ring_features.get_contacts_intra(seq_id, chain.id)
                contacts_inter = ring_features.get_contacts_inter(seq_id, chain.id)

                contacts.append([contacts_inter_sc, contacts_intra, contacts_intra_long])

                chain_entries.append([self.__pdb_id,
                                      chain.id,
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
                                      distances[residue]])

            # Compute structural linearity (9, 10, 11 are the indexes of relevant values)
            struct_lin = structural_linearity(contacts, 0, 1, 2)

            # Append sl as second-last element of each entry
            for entry, sl in zip(chain_entries, struct_lin):
                entry.append(sl)

            entries.append(chain_entries)

        return [residue_entry for chain_entries in entries for residue_entry in chain_entries]

    def __read_residues_info(self) -> DataFrame:
        """
        Read the info from the dataset.

        :return: a list of info as dicts
        """

        logging.info("Reading residues info at {}...".format(self.__example_path))

        return pd.read_csv(self.__example_path, usecols=['pdb_id', 'chain', 'res_seq_id'])

    def __read_features(self) -> pd.DataFrame:
        """
        Read the features from the dataset.

        :return: a list of features as lists
        """

        # Loading features from dataset.csv
        df = pd.read_csv(self.__example_path,
                         usecols=["HSE_up",
                                  "HSE_down",
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

    def __write_entries(self, entries: List[List]):
        """
        Write the given entries to the dataset file.
        If a dataset file already exists, append the new entries.

        :param entries: the entries to be written
        """

        if os.path.isfile(self.__example_path):
            df = pd.DataFrame(entries)
            df.to_csv(self.__example_path, mode='a', header=False)
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
                                       'structural_linearity'])

            df.to_csv(self.__example_path, header=True)

    def __clear_example(self):
        """
        Delete alla data concerning the previous example,
        including entities and edges files
        """

        # Delete the previous example file
        if os.path.isfile(self.__example_path):
            logging.info("Deleting old example file...")
            os.remove(self.__example_path)
            logging.info("Old example file deleted successfully!")

        # Delete all entities files
        if os.path.isdir(self.__entities_path):
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
        if os.path.isdir(self.__edges_path):
            logging.info("Deleting old edges files...")
            for file in os.listdir(self.__edges_path):
                file_path = os.path.join(self.__edges_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
            logging.info("Old edges files deleted successfully!")

    def get_residues_info(self) -> DataFrame:
        return self.__residues_info
