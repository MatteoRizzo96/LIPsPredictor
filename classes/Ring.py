import json
import logging
import os
import shlex
import subprocess
import time
import zipfile

import requests
from Bio.PDB import PDBParser

from classes.RingFeatures import RingFeatures


class Ring:

    def __init__(self,
                 pdb_id: str,
                 use_api=True,
                 working_dir="dataset"):
        """
        Initialize RING and create the graph.

        :param pdb_id: a pdb id as string
        """

        self.__pdb_id = pdb_id
        self.__entities_path = os.path.join(os.getcwd(), working_dir, "entities",
                                            "pdb{}.ent".format(self.__pdb_id))
        self.__edges_path = os.path.join(os.getcwd(), working_dir, "edges",
                                         "pdb{}.edges".format(self.__pdb_id))

        if os.path.isfile(self.__edges_path):
            logging.info("RING already initialized for protein {}".format(self.__pdb_id))
        else:
            logging.info("Initializing RING for protein {}...".format(self.__pdb_id))

            if use_api:
                self.__init_ring_from_api()
            else:
                self.__init_ring_from_executable()

            logging.info('Initializing RING...')

    def get_features(self) -> RingFeatures:
        """
        Use Ring to compute features
        :return: Ring features object
        """
        structure = PDBParser(QUIET=True).get_structure(self.__pdb_id, self.__entities_path)
        return RingFeatures(structure, self.__edges_path)

    def __init_ring_from_executable(self):
        """
        Run RING software to calculate residue contacts.
        """

        logging.info("Initializing RING from executable...")

        # Set the path to the RING library
        path_to_ring = os.path.join(os.getcwd(), "libraries/ring/bin/Ring")

        # The tailing "/" is mandatory
        os.environ["VICTOR_ROOT"] = os.path.abspath("libraries/ring/") + "/"
        command = '{} -i {} --no_energy --all -g 5 -E {} -N /dev/null'.format(path_to_ring,
                                                                              self.__entities_path,
                                                                              self.__edges_path)
        try:
            subprocess.check_output(shlex.split(command))
        except subprocess.CalledProcessError as e:
            print("Error RING {}".format(self.__pdb_id))

        logging.info('Successfully initialized RING from executable!')

    def __init_ring_from_api(self):
        """
        Initialize RING via web API.
        """

        logging.info("Initializing RING from web API...")

        req = {"pdbName": self.__pdb_id,
               "chain": "all",
               "seqSeparation": "5",
               "networkPolicy": "closest",
               "nowater": "true",
               "ringmd": "false",
               "allEdges": "true",
               "thresholds":
                   '{"hbond": 3.5,'
                   ' "vdw": 0.5,'
                   ' "ionic": 4,'
                   ' "pipi": 6.5,'
                   ' "pication": 5,'
                   ' "disulphide": 2.5}'}

        r = requests.post('http://protein.bio.unipd.it/ringws/submit',
                          data=json.dumps(req),
                          headers={'content-type': 'application/json'})
        job_id = json.loads(r.text)['jobid']

        logging.info("Sending request with job ID {}...".format(job_id))

        # Check job status and wait until done
        status = None
        while status != 'complete':
            status = json.loads(
                requests.get('http://protein.bio.unipd.it/ringws/status/{}'.format(job_id)).text).get(
                "status", None)
            logging.info("Job status:{}".format(status))
            time.sleep(5)

        # Retrieve and write the web result to a file
        logging.info("Downloading archive...")

        archive_file = "{}_network.zip".format(self.__pdb_id)
        r = requests.get("http://protein.bio.unipd.it/ring_download/{}/{}".format(job_id, archive_file))
        with open(archive_file, "wb") as file_out:
            file_out.write(r.content)

        logging.info("Archive downloaded successfully!")

        # Extract the contact file from the archive
        logging.info("Extracting archive...")

        archive = zipfile.ZipFile(archive_file, 'r')
        with open(self.__edges_path, "wb") as file_out:
            file_out.write(archive.read("{}_edges.txt".format(self.__pdb_id)))
        logging.info("Archive extracted successfully!")

        logging.info('Successfully initialized RING from web API!')
