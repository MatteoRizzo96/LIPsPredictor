from typing import Dict

from classes.RingNode import RingNode


class RingContact:
    def __init__(self, contact: str):
        n1, inter, n2, dist, angle, energy, a1, a2 = contact.split()[0:8]

        # inter = type(IAC, VDW,...):n1_localization_n2_localization (SC = side chain,
        # (MC = Main chain, LIG if it is a ligand)
        inter_type, inter_loc = inter.split(":")

        # node type can be: MC, SC, LIG
        n1_loc, n2_loc = inter_loc.split("_")

        self.__node1 = RingNode(n1)
        self.__node2 = RingNode(n2)

        self.interaction = {
            "type": inter_type,
            "node1_type": n1_loc,
            "node2_type": n2_loc
        }

        self.__distance = float(dist) if dist.replace('.', '', 1).isdigit() else 0.0
        self.__angle = float(angle) if angle != '-999.9' and angle.replace('.', '', 1).isdigit() else None
        self.__energy = float(energy) if energy.replace('.', '', 1).isdigit() else 0.0

        self.__atom1 = a1
        self.__atom2 = a2

    def get_node1(self) -> RingNode:
        return self.__node1

    def get_node2(self) -> RingNode:
        return self.__node2

    def get_interaction(self) -> Dict:
        return self.interaction

    def get_distance(self) -> float:
        return self.__distance

    def get_angle(self):
        return self.__angle

    def get_energy(self):
        return self.__energy

    def get_atom1(self):
        return self.__atom1

    def get_atom2(self):
        return self.__atom2

    def get_contact_node(self, node_id: str) -> RingNode:
        return self.__node1 if self.__node1.get_id() != node_id else self.__node2
