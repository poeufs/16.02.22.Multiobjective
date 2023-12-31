# Alternative policy structures
import numpy as np


class IrrigationPolicy:
    """irrigation policy class represents different irrigation policies for all the irrigation districts
    Contains the class that allows user to specify desired policy function"""

    def __init__(self, n_inputs, n_outputs, kw_dict):
        """
        The constructor initializes the irrigation policy classes with a number of inputs (n_inputs), a number of
        outputs (n_outputs) and information from the keyword dictionary (kw_dict) on the number of irrigation districts
        stored in self.I
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.I = kw_dict['n_irr_districts']  # number of irrigation districts (== irrigation diversions).
        self.irr_parab_param = np.empty(0)
        self.irr_input_min = np.empty(0)
        self.irr_input_max = np.empty(0)

    def set_parameters(self, IrrTheta):
        self.irr_parab_param = IrrTheta

    def clear_parameters(self):
        self.irr_parab_param = np.empty(0)

    def get_output(self, input):
        """Calculates the irrigation output"""
        input_inflow, input_w, irr_district, irr_district_idx = tuple(input)
        y = float()
        hdg, hdg_dn, m = tuple(3 * [float()])
        start_param_idx = int(irr_district_idx[
                                  irr_district - 2]) # [irr_district-2] need to be changed when additional
                                                        # irrigation districts are added

        hdg = self.irr_parab_param[
            start_param_idx]  # hdg is the first policy parameter for irrigation district i (=threshold)
        m = self.irr_parab_param[start_param_idx + 1]

        hdg_dn = hdg * (self.irr_input_max[irr_district - 2] - self.irr_input_min[irr_district - 2]) \
                 + self.irr_input_min[irr_district - 2]  # hdg_dn = denormalized hdg

        if input_inflow <= hdg_dn:
            import numpy as np
            np.seterr(divide='ignore', invalid='ignore')
            y = min(input_inflow, input_w * (pow(input_inflow / hdg_dn, m)))
        else:
            y = min(input_inflow, input_w)

        return y

    def get_free_parameter_number(self):
        return 2 * self.I

    def set_min_input(self, pV):
        self.irr_input_min = np.array(pV)

    def set_max_input(self, pV):
        self.irr_input_max = np.array(pV)
