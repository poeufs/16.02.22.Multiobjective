# Catchment class
"""Catchments have a name and inflow (from data file)"""

from utils import utils, MyFile


class CatchmentParam:
    # The CatchmentParam class defines CM
    def __init__(self) -> None:
        self.CM = int(0)
        self.inflow_file = MyFile()  # myFile type from utils. Contains the inflow trajectory


class Catchment:
    def __init__(self, pCM):
        cModel = pCM.CM
        self.inflow = utils.loadVector(pCM.inflow_file.filename, pCM.inflow_file.row)

        if cModel == 0:
            self.inflow = utils.loadVector(pCM.inflow_file.filename, pCM.inflow_file.row)

    def get_inflow(self, pt):
        q = float(self.inflow[pt])
        return q
