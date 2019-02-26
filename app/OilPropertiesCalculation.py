from math import *

air = 28.96
R = 10.73

a = 185.843208
b = 1.877840
c = -3.1437
d = -1.32657
e = 1.398441

a1 = 0.816
b1 = 0.172
c1 = -0.989

a2 = 5.38088e-3 
b2 = 0.715082
c2 = -1.87784 
d2 = 3.1437
e2 = 1.32657

class GasSolubility(object):
    """docstring for GasSolubility"""
    def __init__(self, pressure, temperature, API, sg_oil, sg_gas, correlation):
        self.pressure = pressure
        self.temperature = temperature
        self.API = API
        self.sg_oil = sg_oil
        self.sg_gas = sg_gas
        self.correlation = correlation

    def Standing(self):
        x = 0.0125 * self.API - 0.00091 * self.temperature
        return self.sg_gas * ((self.pressure / 18.2 + 1.4) * 10 ** x) ** 1.2048

    def Glaso(self):
        x = 2.8869 - (14.1811 - 3.3093 * log10(self.pressure)) ** 0.5
        pb = 10 ** x
        return self.sg_gas * ((self.API ** 0.989 / self.temperature ** 0.172) * pb ) ** 1.2255

    def Marhoun(self):
        return (a * self.sg_gas ** b * self.sg_oil ** c * self.temperature ** d * self.pressure) ** e

    def Petrosky_Farshad(self):
        x = 7.916e-4 * self.API ** 1.5410 - 4.561e-5 * self.temperature ** (-1.3911)
        return ((self.pressure / 112.727 + 12.34) * self.sg_gas ** 0.8439 * 10 ** x) ** 1.73184

    def Rs(self):
        if self.correlation == 'Standing':
            return self.Standing()
        elif self.correlation == 'Glaso':
            return self.Glaso()
        elif self.correlation == 'Marhoun':
            return self.Marhoun()
        elif self.correlation == 'Petrosky-Farshad':
            return self.Petrosky_Farshad()

class BubblePointPressure(object):
    """docstring for BubblePointPressure"""
    def __init__(self, pressure, temperature, API, sg_oil, sg_gas, Rs_correlation, Pb_correlation):
        self.pressure = pressure
        self.temperature = temperature
        self.API = API
        self.sg_oil = sg_oil
        self.sg_gas = sg_gas
        self.Rs_correlation = Rs_correlation
        self.Pb_correlation = Pb_correlation

    def Rs(self):
        R_solubility = GasSolubility(self.pressure, self.temperature, self.API, self.sg_oil, self.sg_gas, self.Rs_correlation)
        return R_solubility.Rs()

    def Standing(self):
        a = 0.00091 * self.temperature - 0.0125 * self.API
        return 18.2 * ((self.Rs() / self.sg_gas ) ** 0.83 * 10 ** a - 1.4)
        
    def Glaso(self):
        pb = (self.Rs() / self.sg_gas ** a1) * (self.temperature - 460) ** b1 * self.API ** c1
        return 10 ** (1.7669 + 1.7447 * log10(pb) - 0.30218 * (log10(pb)) ** 2)

    def Marhoun(self):
        return a2 * self.Rs() ** b2 * self.sg_oil ** c2 * self.sg_oil ** d2 * self.temperature ** e2

    def Petrosky_Farshad(self):
        x = 4.561e-5 * self.temperature ** 1.3911 - 7.916e-4 * self.API ** 1.541
        return (112.727 * self.Rs() ** 0.577421 / (self.sg_gas ** 0.8439 * 10 ** x)) - 1391.051

    def Pb(self):
        if self.Pb_correlation == 'Standing':
            return self.Standing()
        elif self.Pb_correlation == 'Glaso':
            return self.Glaso()
        elif self.Pb_correlation == 'Marhoun':
            return self.Marhoun()
        elif self.Pb_correlation == 'Petrosky-Farshad':
            return self.Petrosky_Farshad()    

class OilFVF(object):
    """docstring for OilFVF"""
    def __init__(self, pressure, temperature, API, sg_oil, sg_gas, Rs_correlation, Bo_correlation):

        self.pressure = pressure
        self.temperature = temperature
        self.API = API
        self.sg_oil = sg_oil
        self.sg_gas = sg_gas
        self.Rs_correlation = Rs_correlation
        self.Bo_correlation = Bo_correlation

    def Rs(self):
        R_solubility = GasSolubility(self.pressure, self.temperature, self.API, self.sg_oil, self.sg_gas, self.Rs_correlation)
        return R_solubility.Rs()

    def Standing(self):
        return 0.9759 + 0.00012 * (self.Rs() * (self.sg_gas / self.sg_oil) ** 0.5 + 1.25 * self.temperature) ** 1.2

    def Glaso(self):
        Bob = self.Rs() * (self.sg_gas / self.sg_oil) ** 0.526 + 0.968 * self.temperature
        A =  -6.58511 + 2.91329 * log10(Bob) - 0.27683 * log10(Bob) ** 2
        return 1 + 10 ** A

    def Marhoun(self):
        a = 0.742390
        b = 0.323294
        c = -1.202040
        F = self.Rs() ** a * self.sg_gas ** b * self.sg_oil ** c
        return 0.497069 + 0.862963e-3 * self.temperature + 0.182594e-2 * F +  0.318099e-5 * F ** 2

    def Petrosky_Farshad(self):
        return 1.0113 + 7.2046e-5 * (self.Rs() ** 0.3738 * (self.sg_gas ** 0.2914 / self.sg_oil ** 0.6265) + 0.24626 * self.temperature ** 0.5371) ** 3.0936

    def Bo(self):
        if self.Bo_correlation == 'Standing':
            return self.Standing()
        elif self.Bo_correlation == 'Glaso':
            return self.Glaso()
        elif self.Bo_correlation == 'Marhoun':
            return self.Marhoun()
        elif self.Bo_correlation == 'Petrosky-Farshad':
            return self.Petrosky_Farshad()

class IsothermalCompressibility(object):
    """docstring for IsothermalCompressibility"""
    def __init__(self, pressure, temperature, API, sg_oil, sg_gas, Rs_correlation, Pb_correlation):
        self.pressure = pressure
        self.temperature = temperature
        self.API = API
        self.sg_oil = sg_oil
        self.sg_gas = sg_gas
        self.Rs_correlation = Rs_correlation
        self.Pb_correlation = Pb_correlation

    def Rs(self):
        R_solubility = GasSolubility(self.pressure, self.temperature, self.API, self.sg_oil, self.sg_gas, self.Rs_correlation)
        return R_solubility.Rs()

    def Pb(self):
        P_bubblepoint = BubblePointPressure(self.pressure, self.temperature, self.API, self.sg_oil, self.sg_gas, self.Rs_correlation, self.Pb_correlation)
        return P_bubblepoint.Pb()

    def Petrosky_Farshad(self):
        return 1.705e-7 * self.Rs() ** 0.69357 * self.sg_gas ** 0.1885 * self.API ** 0.3272

    def McCain(self):
        A = -7.573 - 1.45 * log(self.pressure) - 0.383 * log(self.Pb()) + 1.402 * log(self.temperature) + 0.256 * log(self.API) + 0.449 * log(self.Rs()) 
        return exp(A)

    def Co(self):
        if self.pressure > self.Pb():
            return self.Petrosky_Farshad()
        else:
            return self.McCain()

class OilFVFforUnderSaturated(object):
    """docstring for OilFVFforUnderSaturated"""
    def __init__(self, 
                 pressure,
                 temperature, 
                 API, 
                 sg_oil, 
                 sg_gas, 
                 Rs_correlation,
                 Pb_correlation,
                 Bo_correlation):
        self.pressure = pressure
        self.temperature = temperature
        self.API = API
        self.sg_oil = sg_oil
        self.sg_gas = sg_gas
        self.Rs_correlation = Rs_correlation
        self.Pb_correlation = Pb_correlation
        self.Bo_correlation = Bo_correlation
        self.Rs, self.Pb, self.Bob = self.params()

    def params(self):
        R_solubility = GasSolubility(self.pressure, self.temperature, self.API, self.sg_oil, self.sg_gas, self.Rs_correlation)
        Rs =  R_solubility.Rs()

        P_bubblepoint = BubblePointPressure(self.pressure, self.temperature, self.API, self.sg_oil, self.sg_gas, self.Rs_correlation, self.Pb_correlation)
        Pb = P_bubblepoint.Pb()

        B_bubblepoint = OilFVF(Pb, self.temperature, self.API, self.sg_oil, self.sg_gas, self.Rs_correlation, self.Bo_correlation)
        Bob = B_bubblepoint.Bo()

        return Rs, Pb, Bob

    def Petrosky_Farshad(self):
        A = 4.1646e-7 * self.Rs ** 0.69357 * self.API ** 0.3272 * self.temperature ** 0.6729
        return self.Bob * exp(-A * (self.pressure ** 0.4094 - self.Pb ** 0.4094))

class OilDensity(object):
    """docstring for OilDensity"""
    def __init__(self, 
                 pressure,
                 temperature, 
                 API, 
                 sg_oil, 
                 sg_gas, 
                 Rs_correlation,
                 Pb_correlation,
                 Bo_correlation):
        self.pressure = pressure
        self.temperature = temperature
        self.API = API
        self.sg_oil = sg_oil
        self.sg_gas = sg_gas
        self.Rs_correlation = Rs_correlation
        self.Pb_correlation = Pb_correlation
        self.Bo_correlation = Bo_correlation
        self.Rs, self.Pb, self.Bob = self.params()

    def params(self):
        R_solubility = GasSolubility(self.pressure, self.temperature, self.API, self.sg_oil, self.sg_gas, self.Rs_correlation)
        Rs =  R_solubility.Rs()

        P_bubblepoint = BubblePointPressure(self.pressure, self.temperature, self.API, self.sg_oil, self.sg_gas, self.Rs_correlation, self.Pb_correlation)
        Pb = P_bubblepoint.Pb()

        B_bubblepoint = OilFVF(Pb, self.temperature, self.API, self.sg_oil, self.sg_gas, self.Rs_correlation, self.Bo_correlation)
        Bob = B_bubblepoint.Bo()

        return Rs, Pb, Bob

    def saturated_density(self):
        B_bubblepoint = OilFVF(self.pressure, self.temperature, self.API, self.sg_oil, self.sg_gas, self.Rs_correlation, self.Bo_correlation)
        Bo = B_bubblepoint.Bo()

        return (62.4 * self.sg_oil + 0.0136 * self.Rs * self.sg_gas) / Bo

    def under_density(self):
        bp_density = (62.4 * self.sg_oil + 0.0136 * self.Rs * self.sg_gas) / self.Bob
        A = 4.1646e-7 * self.Rs ** 0.69357 * self.API ** 0.3272 * self.temperature ** 0.6729

        return bp_density * exp(A * (self.pressure ** 0.4094 - self.Pb ** 0.4094))
        
class OilVisCosity(object):
    """docstring for OilVisCosity"""
    def __init__(self, 
                 pressure,
                 temperature, 
                 API, 
                 sg_oil, 
                 sg_gas,
                 Rs_correlation,
                 Pb_correlation,
                 dead_oil_correlation, 
                 saturated_oil_correlation):
        self.pressure = pressure
        self.temperature = temperature
        self.API = API
        self.sg_oil = sg_oil
        self.sg_gas = sg_gas
        self.Rs_correlation = Rs_correlation
        self.Pb_correlation = Pb_correlation
        self.dead_oil_correlation = dead_oil_correlation
        self.saturated_oil_correlation = saturated_oil_correlation
        self.Rs, self.Pb = self.params()

    def params(self):
        R_solubility = GasSolubility(self.pressure, self.temperature, self.API, self.sg_oil, self.sg_gas, self.Rs_correlation)        
        Rs = R_solubility.Rs()

        P_bubblepoint = BubblePointPressure(self.pressure, self.temperature, self.API, self.sg_oil, self.sg_gas, self.Rs_correlation, self.Pb_correlation)
        Pb = P_bubblepoint.Pb()

        return Rs, Pb

    def DeadOil_Beal(self):
        a = 10 ** (0.43 + 8.33 / self.API)
        return (0.32 + 1.8e7 / self.API ** 4.53) * (360 / (self.temperature)) ** a

    def DeadOil_Beggs_Robinson(self):
        z = 3.0324 - 0.02023 * self.API
        y = 10 ** z
        x = y * self.temperature ** (-1.163)
        return 10 ** x - 1

    def DeadOil_Glaso(self):
        a = 10.313 * log10(self.temperature) - 36.447
        return 3.141e10 * self.temperature ** (-3.444) * log10(self.API) ** a

    def V_dead_oil(self):
        if self.dead_oil_correlation == 'Beal':
            return self.DeadOil_Beal()
        elif self.dead_oil_correlation == 'Beggs-Robinson':
            return self.DeadOil_Beggs_Robinson()
        elif self.dead_oil_correlation == 'Glaso':
            return self.DeadOil_Glaso()

    def SOil_Chew_Connally(self):
        e = 3.74e-3 * self.Rs
        d = 1.1e-3 * self.Rs
        c = 8.62e-5 * self.Rs
        b = 0.68 / 10 ** c + 0.25 / 10 ** d + 0.062 / 10 ** e
        a = self.Rs * (2.2e-7 * self.Rs - 7.4e-4)
        return 10 ** a * self.V_dead_oil() ** b

    def SOil_Beggs_Robinson(self):
        a = 10.715 * (self.Rs + 100) ** (-0.515)
        b = 5.44 * (self.Rs + 150) ** (-0.338)
        return a * self.V_dead_oil() ** b

    def S_oil(self):
        if self.saturated_oil_correlation == 'Beggs-Robinson':
            return self.SOil_Beggs_Robinson()
        elif self.saturated_oil_correlation == 'Chew-Connally':
            return self.SOil_Chew_Connally()

    def Under_oil(self):
        a = -3.9e-5 * self.pressure - 5
        m = 2.6 * self.pressure ** 1.187 * 10 ** a
        return self.S_oil() * (self.pressure / self.Pb) ** m
        
        







