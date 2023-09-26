# Reservoir class and ReservoirParam struct (class in Python)
# encapsulating the parameters of the reservoir

from utils import MyFile, utils
import numpy as np


class ReservoirParam:
    def __init__(self):
        self.EV = int()
        self.evap_rates = MyFile()
        self.rating_curve = MyFile()
        self.rating_curve_minmax = MyFile()
        self.rule_curve = MyFile()
        self.lsv_rel = MyFile()
        self.A = float()  # reservoir surface (assumed to be constant)
        self.initCond = float()
        self.tailwater = MyFile()
        self.minEnvFlow = MyFile()


class Reservoir:
    """
    A class used to represent reservoirs of the problem

    Attributes
    ----------
    LakeName : str
        lowercase non-spaced name of the reservoir
    ReservoirParam : miscellaneous
        parameters specified in above construct (will be elaborated!!!!)

    Methods
    -------
    storage_to_level(h=float)
        Returns the level(height) based on volume
    level_to_storage(s=float)
        Returns the volume based on level(height)
    level_to_surface(h=float)
        Returns the surface area based on level
    """

    def __init__(self, name):
        self.LakeName = name  # assigning the name of the reservoir in constructor

    def storage_to_level(self, s):
        # interpolation when lsv_rel exists
        if (self.lsv_rel.size > 0):
            h = utils.interp_lin(self.lsv_rel[2], self.lsv_rel[0], s)
        # approximating with volume and cross section
        else:
            h = s / self.A
        return h

    def level_to_storage(self, h):
        # interpolation when lsv_rel exists
        if (self.lsv_rel.size > 0):
            s = utils.interp_lin(self.lsv_rel[0], self.lsv_rel[2], h)
        # approximating with level and cross section
        else:
            s = h * self.A
        return s

    def level_to_surface(self, h):
        # interpolation when lsv_rel exists      
        if (self.lsv_rel.size > 0):
            S = utils.interp_lin(self.lsv_rel[0], self.lsv_rel[1], h)
        # approximating with cross section
        else:
            S = self.A
        return S

    def min_release(self, s):
        if (self.LakeName == "kafuegorgelower"):
            q = 0.0
            if (s <= self.rating_curve_minmax[0]):
                q = 0
            elif (s >= self.rating_curve_minmax[1]):
                q = self.rating_curve_minmax[2]
            else:
                q = 0
            return q

        else:
            # no time-dependent (cmonth=0 not used)
            q = 0.0
            if (self.rating_curve.size > 0):
                q = utils.interp_lin(self.rating_curve[0], self.rating_curve[1], self.storage_to_level(s))
            else:
                print(self.LakeName, " rating curve not defined")
            return q

    def max_release(self, s):
        if (self.LakeName == "kafuegorgelower"):
            q = 0.0
            if (s <= self.rating_curve_minmax[0]):
                q = 0
            elif (s >= self.rating_curve_minmax[1]):
                q = self.rating_curve_minmax[2]
            else:
                q = self.rating_curve_minmax[2]
            return q
        else:
            q = 0.0
            if (self.rating_curve.size > 0):
                q = utils.interp_lin(self.rating_curve[0], self.rating_curve[2], self.storage_to_level(s))
            else:
                print(self.LakeName, " rating curve not defined")
            return q

    def actual_release_MEF(self, uu, s, cmonth, n_sim, MEF):

        if (self.LakeName == "itezhitezhi"):
            # min-Max storage-discharge relationship
            qm = self.min_release(s)
            qM = self.max_release(s)

            # actual release
            rr = min(qM, max(qm, uu))

            # compute actual release - NO FLOW AUGMENTATION for Itezhitezhi
            rr_MEF = 0.0

            if (MEF <= 40):  # Itezhitezhi MUST release a MF of 40 m3/sec all year round
                rr_MEF = max(rr, MEF)
            else:
                if (n_sim <= 40):
                    rr_MEF = max(rr, 40)
                elif (n_sim > 40 and n_sim < MEF):
                    rr_MEF = max(rr, n_sim)
                elif (n_sim >= MEF):
                    rr_MEF = max(rr, MEF)

            return rr_MEF

        else:
            # min-Max storage-discharge relationship
            qm = self.min_release(s)
            qM = self.max_release(s)

            # actual release
            rr = min(qM, max(qm, uu))
            rr_MEF = max(rr, MEF)

            return rr_MEF

    def integration(self, HH, tt, s0, uu, n_sim, cmonth):  # returns double vector!
        # HH = number of days in the current month
        # tt = current month

        self.sim_step = 3600 * 2 * HH / HH
        HH = int(HH)

        self.s = np.full(HH + 1, -999.0).astype('float')
        self.r = np.full(HH, -999)
        self.stor_rel = np.empty(0)

        self.MEF = self.getMEF(cmonth - 1)

        # initial conditions
        self.s[0] = s0

        for i in range(HH):
            # compute actual release - NO FLOW AUGMENTATION
            self.r[i] = self.actual_release_MEF(uu, self.s[i], cmonth, n_sim, self.MEF)

            # compute evaporation
            if (self.EV == 1):
                self.S = self.level_to_surface(self.storage_to_level(self.s[i]))
                self.E = self.evap_rates[cmonth - 1] / 1000 * self.S / (3600 * 2 * HH)
            # One elif to be implemented
            else:
                self.E = 0.0

            # system transition
            self.s[i + 1] = self.s[i] + self.sim_step * (n_sim - self.r[i] - self.E)

        self.stor_rel = np.append(self.stor_rel, self.s[HH])
        self.stor_rel = np.append(self.stor_rel, np.mean(self.r))

        return self.stor_rel

    def integration_daily(self, HH, tt, s0, uu, n_sim, cmonth):
        # HH = number of days in the current month
        # tt = current month

        self.sim_step = 3600 * 24 * HH / HH

        HH = int(HH)

        self.s = np.full(HH + 1, -999, dtype=float)

        self.r = np.full(HH, -999)
        self.stor_rel = np.empty(0)

        self.MEF = self.getMEF(cmonth - 1)

        # initial conditions
        self.s[0] = s0

        for i in range(HH):
            # compute actual release - NO FLOW AUGMENTATION
            self.r[i] = self.actual_release_MEF(uu, self.s[i], cmonth, n_sim, self.MEF)

            # compute evaporation
            if (self.EV == 1):
                self.S = self.level_to_surface(self.storage_to_level(self.s[i]))
                self.E = self.evap_rates[cmonth - 1] / 1000 * self.S / (86400 * HH)
            # One elif to be implemented!!
            else:
                self.E = 0.0

            # system transition
            self.s[i + 1] = self.s[i] + self.sim_step * (n_sim - self.r[i] - self.E)

        self.stor_rel = np.append(self.stor_rel, self.s[HH])
        self.stor_rel = np.append(self.stor_rel, np.mean(self.r))

        return self.stor_rel

    def actual_release(self, uu, s, cmonth):
        # min-Max storage-discharge relationship
        qm = self.min_release(s)
        qM = self.max_release(s)

        # actual release
        rr = min(qM, max(qm, uu))

        return rr

    def relToTailwater(self, r):
        hd = 0.0
        if (self.tailwater.size > 0):
            hd = utils.interp_lin(self.tailwater[0], self.tailwater[1], r)

        return hd

    def setInitCond(self, ci):
        self.init_condition = ci

    def getInitCond(self):
        return self.init_condition

    def setEvap(self, pEV):
        self.EV = pEV

    def setEvapRates(self, pEvap):
        self.evap_rates = utils.loadVector(pEvap.file_name, pEvap.row)

    def setRatCurve(self, pRatCurve):
        self.rating_curve = utils.loadMatrix(pRatCurve.file_name, pRatCurve.row, pRatCurve.col)

    def setRatCurve_MinMax(self, pRatCurve_MinMax):
        self.rating_curve_minmax = utils.loadVector(pRatCurve_MinMax.file_name, pRatCurve_MinMax.col)

    def setRuleCurve(self, pRuleCurve):
        self.rule_curve = utils.loadMatrix(pRuleCurve.file_name, pRuleCurve.row, pRuleCurve.col)

    def setLSV_Rel(self, pLSV_Rel):
        self.lsv_rel = utils.loadMatrix(pLSV_Rel.file_name, pLSV_Rel.row, pLSV_Rel.col)

    def setSurface(self, pA):
        self.A = pA

    def setTailwater(self, pTailWater):
        self.tailwater = utils.loadMatrix(pTailWater.file_name, pTailWater.row, pTailWater.col)

    def setMEF(self, pMEF):
        self.minEnvFlow = utils.loadVector(pMEF.file_name, pMEF.row)

    def getMEF(self, pMoy):
        return self.minEnvFlow[pMoy]
