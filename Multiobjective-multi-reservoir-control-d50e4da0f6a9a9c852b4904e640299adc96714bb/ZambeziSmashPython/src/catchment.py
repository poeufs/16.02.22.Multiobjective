# Catchment class
"""Catchments have a name and inflow (from data file)"""

from utils import utils, MyFile


class CatchmentParam:
    # The CatchmentParam class defines CM
    def __init__(self) -> None:
        self.CM = int(0)  # type of catchment model_redriver (0 = historical trajectory, 1 = HBV)
        # HBV = Hydrologiska Byråns Vattenbalansavdelning model. A hydrological transport model to measure the
        # streamflow (Arnold et. al, 2023). However, this is not implemented in the code (neither in c++ code).
        self.inflow_file = MyFile()  # myFile type from utils. Contains the inflow trajectory


class Catchment:
    # loads the inflow of the catchment.
    def __init__(self, pCM):
        cModel = pCM.CM
        self.inflow = utils.loadVector(pCM.inflow_file.filename, pCM.inflow_file.row) # overwritten in

        if cModel == 0:
            self.inflow = utils.loadVector(pCM.inflow_file.filename, pCM.inflow_file.row)

    def get_inflow(self, pt):
        """ function to retrieve the inflow for day "pt" """
        q = float(self.inflow[pt])
        return q
