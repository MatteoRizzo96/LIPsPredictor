class RingNode:

    def __init__(self, node: str):
        # node = chain : residue_number : insertion_code : residue_name
        chain, residue_number, insertion_code, residue_name = node.split(":")

        self.__id = chain + residue_number
        self.__chain = chain
        self.__residue_number = int(residue_number)
        self.__insertion_code = insertion_code if insertion_code != ' ' else '_'
        self.__residue_name = residue_name

    def get_id(self) -> str:
        return self.__id

    def get_chain(self) -> str:
        return self.__chain

    def get_residue_number(self) -> int:
        return self.__residue_number

    def get_insertion_code(self) -> str:
        return self.__insertion_code

    def get_residue_name(self) -> str:
        return self.__residue_name
