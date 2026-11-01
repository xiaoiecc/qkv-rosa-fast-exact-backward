"""Independent causal-join and suffix-index oracles for exact acceleration."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import unittest
from tests._support import BUILD, CSRC
class ExactRoundAuxiliaryTests(unittest.TestCase):
    def test_lazy_indexes_and_constant_coordinate_join(self):
        BUILD.mkdir(parents=True,exist_ok=True)
        source=Path(__file__).parent/'cpp/exact_round_auxiliary.cpp'
        binary=BUILD/'exact_round_auxiliary'
        subprocess.run(shlex.split(os.environ.get('CXX','c++'))+['-O2','-std=c++17','-I',str(CSRC),str(source),'-o',str(binary)],check=True)
        result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
        report=json.loads(result.stdout)
        self.assertEqual(report['status'],'PASS')
        self.assertGreater(report['ancestor_queries'],100000)
        self.assertGreater(report['endpoint_queries'],100000)
if __name__=='__main__':unittest.main()
