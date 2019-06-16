import argparse
import logging
import os
import sys

from classes.Params import Params


class Initializer:

    def __init__(self):
        """
        Initialize logger and retrieve the input from terminal.
        """

        # Initialize the logger
        self.__init_logger()

        # Initialize the parser
        parser = self.__init_parser()
        args = parser.parse_args()

        # Read json configuration file
        self.__params = self.__create_params_object(args.configuration)

        # Get pdb file and trim filename
        pdb_name = args.pdbfile.strip()
        self.__pdb_id = str(pdb_name[:-4])

        self.__dssp_path = os.path.join(os.getcwd(), args.dssp_path)

        self.__out_dir = args.out_dir

    def get_pdb_id(self) -> str:
        return self.__pdb_id

    def get_params(self) -> Params:
        return self.__params

    def get_dssp_path(self) -> str:
        return self.__dssp_path

    def get_out_dir(self) -> str:
        return self.__out_dir

    @staticmethod
    def __init_parser() -> argparse.ArgumentParser:
        """
        Initialize the arguments parser and read the input JSON file.

        The program takes as input a PDB file in the PDB format, optional parameters are:
            - configuration file with algorithm parameters;
            - output directory.

        Example of console input:
        $ python initializer.py -configuration parameters.json -out_dir results 1jsu.pdb

        Example of working parameters with absolute path to xssp:
        -dssp_path /home/[your_name]/opt/xssp/mkdssp -configuration configuration/parameters.json 2o8a.pdb

        Example of working parameters with relative path to xssp in the repo:
        -dssp_path libraries/xssp/mkdssp -configuration configuration/parameters.json 2o8a.pdb
        """

        parse_obj = argparse.ArgumentParser()

        parse_obj.add_argument('-dssp_path', help='Required absolute path to xssp executable', type=str)

        parse_obj.add_argument('-configuration',
                               default=os.path.join(os.getcwd(), 'configuration', 'parameters.ini'),
                               type=str,
                               help='Optional file with algorithm parameters to use')

        parse_obj.add_argument('-out_dir', default=os.path.join(os.getcwd(), 'results'),
                               type=str,
                               help='Optional output folder (default to results/')

        parse_obj.add_argument('pdbfile', help='Input pdb file name (not path)', type=str)

        return parse_obj

    @staticmethod
    def __init_logger():
        """
        Initialize the logger.
        """

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S',
                            handlers=[
                                logging.FileHandler('app.log', mode='w'),
                                logging.StreamHandler(sys.stdout)
                            ])

        logging.info('Logger initialized')

    @staticmethod
    def __create_params_object(path: str) -> Params:
        """
        Create a Params object to store parameters.

        :param path: the path to the params.json file containing the parameters
        :return: a Param object based on the params.json
        """

        assert os.path.isfile(path), "No json configuration file found at {}".format(path)
        return Params(path)
