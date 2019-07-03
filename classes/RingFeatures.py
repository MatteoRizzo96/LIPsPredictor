from typing import Dict

from Bio.PDB import is_aa

from classes.RingContact import RingContact
from classes.RingNode import RingNode


class RingFeatures:

    def __init__(self, structure, contact_file_path):
        """
        Build a RING features object storing all features retrievable from RING
        """

        self.__contacts = self.__init_contacts(contact_file_path, contact_threshold=3.5)
        self.__features = self.__init_features(structure)

    @staticmethod
    def __init_contacts(contact_file_path: str, contact_threshold: float) -> Dict:
        """
        Parse the edge file and generate the network.

        :param contact_threshold: a distance threshold for contacts
        :return: a dict of contacts
        """

        contacts = {}

        with open(contact_file_path) as f:
            next(f)
            for line in f:
                contact = RingContact(line)

                if contact.get_distance() <= contact_threshold:
                    contacts.setdefault(contact.get_node1().get_id(), [])
                    contacts.setdefault(contact.get_node2().get_id(), [])

                    contacts[contact.get_node1().get_id()].append(contact)
                    contacts[contact.get_node2().get_id()].append(contact)

        return contacts

    def __init_features(self, structure):
        residues_features = {}

        for chain in structure[0]:
            residues_features[chain.id] = {}
            for residue in chain:
                if is_aa(residue):
                    # Create a RING node object from a RING string
                    node = RingNode("{}:{}:{}:{}".format(chain.id,
                                                         residue.id[1],
                                                         residue.id[2],
                                                         residue.get_resname()))

                    # Calculate contacts features
                    contacts_inter, contacts_intra, contacts_ratio, contacts_intra_long, \
                    contacts_inter_sc, contacts_energy_ratio = self.__calculate_contacts(node)

                    residue_features = {
                        "contacts_inter": contacts_inter,
                        "contacts_intra": contacts_intra,
                        "contacts_ratio": contacts_ratio,
                        "contacts_intra_long": contacts_intra_long,  # used in structural linearity
                        "contacts_inter_sc": contacts_inter_sc,  # used in structural linearity
                        "contacts_energy_ratio": contacts_energy_ratio
                    }
                    residues_features[node.get_chain()][node.get_residue_number()] = residue_features

        return residues_features

    def __calculate_contacts(self, node: RingNode) -> (int, int, float, int, int):
        contacts_inter, contacts_intra, contacts_inter_sc, contacts_intra_long, intra_energy, inter_energy = 0, 0, 0, 0, 0, 0

        # Calculate inter and intra contacts
        for contact in self.__contacts.get(node.get_id(), []):
            contact_node = contact.get_contact_node(node.get_id())

            # Inter-chain contacts
            if contact_node.get_chain() != node.get_chain():
                contacts_inter += 1
                # Get the energy of the contact
                inter_energy += contact.get_energy()

                # Inter-chain contacts where at least one of the involved res is in side chain
                inter = contact.get_interaction()
                if inter['node1_type'] == 'SC' or inter['node2_type'] == 'SC':
                    contacts_inter_sc += 1
            # Intra-chain contacts
            else:
                contacts_intra += 1
                # Get the energy of the contact
                intra_energy += contact.get_energy()

                # Intra chain contacts with sequence separation > 7
                res1_pos = contact.get_node1().get_residue_number()
                res2_pos = contact.get_node2().get_residue_number()
                if abs(res1_pos - res2_pos) > 7:
                    contacts_intra_long += 1

        # Calculate contacts ratio (inter/intra)
        contacts_ratio = float(contacts_intra / (contacts_inter + 1))
        # Calculate the ratio of the contacts w.r.t. the energy of each bond
        contacts_energy_ratio = float(intra_energy / (inter_energy + 1))
        return contacts_inter, contacts_intra, contacts_ratio, contacts_intra_long, contacts_inter_sc, contacts_energy_ratio

    def get_all_features(self, target_residue_id: int, target_chain: str) -> Dict:
        return self.__features[target_chain][target_residue_id]

    def get_contacts_ratio(self, target_residue_id: int, target_chain: str) -> float:
        return self.__features[target_chain][target_residue_id]["contacts_ratio"]

    def get_contacts_inter(self, target_residue_id: int, target_chain: str) -> int:
        return self.__features[target_chain][target_residue_id]["contacts_inter"]

    def get_contacts_intra(self, target_residue_id: int, target_chain: str) -> int:
        return self.__features[target_chain][target_residue_id]["contacts_intra"]

    def get_contacts_inter_sc(self, target_residue_id: int, target_chain: str) -> int:
        return self.__features[target_chain][target_residue_id]["contacts_inter_sc"]

    def get_contacts_intra_long(self, target_residue_id: int, target_chain: str) -> int:
        return self.__features[target_chain][target_residue_id]["contacts_intra_long"]

    def get_contacts_energy_ratio(self, target_residue_id: int, target_chain: str) -> float:
        return self.__features[target_chain][target_residue_id]["contacts_energy_ratio"]
