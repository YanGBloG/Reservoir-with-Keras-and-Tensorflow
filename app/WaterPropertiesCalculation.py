import math

# Gas-Free Water coefficient
a11 = 0.9947
a12 = -4.228*10**(-6)
a13 = 1.3*10**(-10)
a21 = 5.8*10**(-6)
a22 = 1.8376*10**(-8)
a23 = -1.3855*10**(-12)
a31 = 1.02*10**(-6)
a32 = -6.77*10**(-11)
a33 = 4.285*10**(-15)

# Gas-Saturated Water
b11 = 0.9911
b12 = -1.093*10**(-6)
b13 = -5.0*10**(-11)
b21 = 6.35*10**(-5)
b22 = -3.497*10**(-9)
b23 = 6.429*10**(-13)
b31 = 8.5*10**(-7)
b32 = 4.57*10**(-12)
b33 = -1.43*10**(-15)

class WaterDensity():
    def __init__(self, pressure, temperature, gravity, nacl, gas_type):
        self.pressure = pressure
        self.temperature = temperature
        self.gravity = gravity
        self.nacl = nacl
        self.gas_type = gas_type

    def standard_condition(self):
        return 62.368 + 0.438603 * (self.nacl / 100) + 1.60074e-3 * (self.nacl / 100) ** 2

    def reservoir_condition(self):
        waterFVF = WaterFVF(self.pressure, self.temperature, self.gravity, self.gas_type)
        Bw = waterFVF.Bwater()
        return self.standard_condition() / Bw

class WaterFVF():
    def __init__(self, pressure, temperature, gravity, gas_type):
        self.pressure = pressure
        self.temperature = temperature
        self.gravity = gravity
        self.gas_type = gas_type

    def free_gas(self):
        A1 = a11 + a21 * (self.temperature - 460) + a31 * (self.temperature - 460)**2
        A2 = a12 + a22 * (self.temperature - 460) + a32 * (self.temperature - 460)**2
        A3 = a13 + a23 * (self.temperature - 460) + a33 * (self.temperature - 460)**2
        return A1, A2, A3

    def saturated_gas(self):
        A1 = b11 + b21 * (self.temperature - 460) + b31 * (self.temperature - 460)**2
        A2 = b12 + b22 * (self.temperature - 460) + b32 * (self.temperature - 460)**2
        A3 = b13 + b23 * (self.temperature - 460) + b33 * (self.temperature - 460)**2
        return A1, A2, A3

    def Bwater(self):
        if self.gas_type == 'Gas-Free':
            A1, A2, A3 = self.free_gas()
        elif self.gas_type == 'Gas-Saturated':
            A1, A2, A3 = self.saturated_gas()
        return A1 + A2 * self.pressure + A3 * self.pressure ** 2

class WaterViscosity():
    def __init__(self, pressure, temperature, gravity, nacl, correlation):
        self.pressure = pressure
        self.temperature = temperature
        self.gravity = gravity
        self.nacl = nacl
        self.correlation = correlation

    def brine_viscosity(self):
        D = 1.12166 - 0.0263951 * (self.nacl / 100) + 6.79461e-4 * (self.nacl / 100)**2 + 5.47119e-5 * (self.nacl / 100)**3 - 1.55586e-6 * (self.nacl / 100)**4
        return (109.574 - 8.40564 * (self.nacl / 100) + 0.313314 * (self.nacl / 100)**2 + 8.72213e-3 * (self.nacl / 100)**3) * self.temperature**(-D)

    def waterViscosity(self):
        if self.correlation == 'McCain':
            return self.brine_viscosity() * (0.9994 + 4.0295e-5 * self.pressure + 3.1062e-9 * self.pressure**2)
        elif self.correlation == 'Beggs-Brill':
            return math.exp(1.003 - 1.479e-2 * (self.temperature - 460) + 1.982e-5 * (self.temperature - 460)**2)

class GasSolubilityinWater():
    def __init__(self, pressure, temperature, gravity):
        self.pressure = pressure
        self.temperature = temperature
        self.gravity = gravity

    def Rsw(self):
        A = 2.12 + 3.45e-3 * (self.temperature - 460) - 3.59e-5 * (self.temperature - 460) ** 2
        B = 0.0107 - 5.26e-5 * (self.temperature - 460) + 1.48e-7 * (self.temperature - 460) ** 2
        C = 8.75e-7 + 3.9e-9 * (self.temperature - 460) - 1.02e-11 * (self.temperature - 460) ** 2
        return A + B * self.pressure + C * self.pressure**2

class WaterIsothermalCompressibity():
    def __init__(self, pressure, temperature, gravity):
        self.pressure = pressure
        self.temperature = temperature
        self.gravity = gravity

    def Cw(self):
        C1 = 3.8546 - 0.000134 * self.pressure
        C2 = -0.01052 + 4.77e-7 * self.pressure
        C3 = 3.9267e-5 - 8.8e-10 * self.pressure
        return  (C1 + C2 * (self.temperature - 460) + C3 * (self.temperature - 460)**2 ) / 1e6
