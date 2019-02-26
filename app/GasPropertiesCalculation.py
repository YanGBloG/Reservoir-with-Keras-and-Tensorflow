import math

air = 28.96
R = 10.73

A1 = 0.3265
A2 = -1.0700
A3 = -0.5339
A4 = 0.01569
A5 = -0.05165
A6 = 0.5475
A7 = -0.7361
A8 = 0.1844
A9 = 0.1056
A10 = 0.6134
A11 = 0.7210

B1 = 0.31506237 
B2 = -1.0467099 
B3 = -0.57832720 
B4 = 0.53530771
B5 = -0.61232032
B6 = -0.10488813
B7 = 0.68157001
B8 = 0.68446549

C0 = -2.46211820
C1 = 2.970547414
C2 = -2.86264054*10**(-1)
C3 = 8.05420522*10**(-3)
C4 = 2.80860949
C5 = -3.49803305
C6 = 3.60373020*10**(-1)
C7 = -1.044324*10**(-2)
C8 = -7.93385648*10**(-1)
C9 = 1.39643306
C10 = -1.49144925*10**(-1)
C11 = 4.41015512*10**(-3)
C12 = 8.39387178*10**(-2)
C13 = -1.86408848*10**(-1)
C14 = 2.03367881*10**(-2)
C15 = -6.09579263*10**(-4)

class zFactor():
    def __init__(self, getPressure, getTemperature, getGravity, co2, h2s, n2, correlation):
        self.getGravity = getGravity
        self.getTemperature = getTemperature
        self.getPressure = getPressure
        self.h2s = h2s
        self.co2 = co2
        self.n2 = n2
        self.correlation = correlation

    def epsilon(self):
        A = self.h2s + self.co2
        B = self.h2s
        return 120*(A**0.9 - A**1.6) + 15*(B**0.5 - B**4)

    def Tpc(self):
        if all([self.co2 == 0, self.h2s == 0, self.n2 == 0]):
            return 168 + 325*self.getGravity - 12.5*self.getGravity**2
        elif self.n2 == 0:
            Tpc = 168 + 325*self.getGravity - 12.5*self.getGravity**2
            return Tpc - self.epsilon()
        else:
            Tpc = 168 + 325*self.getGravity - 12.5*self.getGravity**2
            return Tpc - 80*self.co2 + 130*self.h2s - 250*self.n2

    def Tpr(self):
        return self.getTemperature/self.Tpc()

    def Ppc(self):
        if all([self.co2 == 0, self.h2s == 0, self.n2 == 0]):
            return 677 + 15*self.getGravity - 37.5*self.getGravity**2
        elif self.n2 == 0:
            Ppc = 677 + 15*self.getGravity - 37.5*self.getGravity**2
            return (Ppc*self.Tpc()) / (self.Tpc() + self.h2s*(1 - self.h2s)*self.epsilon())
        else:
            Ppc = 677 + 15*self.getGravity - 37.5*self.getGravity**2
            return Ppc + 440*self.co2 + 600*self.h2s - 170*self.n2

    def Ppr(self):
        return self.getPressure/self.Ppc()

    ### Hall-Yarborough correlation parameters
    def t(self):
        return self.Tpc()/self.getTemperature

    def X1(self):
        X1 = -0.06125*self.Ppr()*self.t()*math.exp(-1.2*(1-self.t())**2)
        return X1

    def X2(self):
        X2 = 14.76*self.t() - 9.76*self.t()**2 + 4.58*self.t()**3
        return X2

    def X3(self):
        X3 = 90.7*self.t() - 242.2*self.t()**2 + 42.3*self.t()**3
        return X3

    def X4(self):
        X4 = 2.18 + 2.82*self.t()
        return X4

    def Hall_Yarborough(self):
        y = 0.0125*self.Ppr()*self.t()*math.exp(-1.2*(1-self.t())**2)
        for i in range(1000):
            fy = self.X1() + (y + y**2 + y**3 + y**4)/(1-y)**3 - self.X2()*y**2 + self.X3()*y**(self.X4())
            dfy = ((1+2*y+3*y**2+4*y**3)*(1-y)**3+3*(1-y)**2*(y+y**2+y**3+y**4))/(1-y)**6-2*self.X2()*y+self.X3()*self.X4()*y**(self.X4()-1)
            y1 = y - fy/dfy
            if abs(y1 - y) < 10**(-12):
                break
            else:
                y = y1
        return y1

    ### Dranchuk-Purvis-Robinson correlation parameters
    def T1(self):
        T1 = B1 + B2/self.Tpr() + B3/self.Tpr()**3
        return T1

    def T2(self):
        T2 = B4 + B5/self.Tpr()
        return T2

    def T3(self):
        T3 = B5*B6/self.Tpr()
        return T3

    def T4(self):
        T4 = B7/self.Tpr()**3
        return T4

    def T5(self):
        T5 = 0.27*self.Ppr()/self.Tpr()
        return T5

    def Dranchuk_Purvis_Robinson(self):
        y = 0.27*self.Ppr()/self.Tpr()
        for i in range(1000):
            dy = 1 + self.T1()*y + self.T2()*y**2 + self.T3()*y**5 + (self.T4()*y**2*(1+B8*y**2)*math.exp(-B8*y**2)) - self.T5()/y
            dfy = self.T1() + 2*self.T2()*y + 5*self.T3()*y**4 + 2*self.T4()*y*math.exp(-B8*y**2)*(B8**2*y**4 + B8*y**2 + 1) + self.T5()/y**2
            y1 = y - dy/dfy
            if abs(y1 - y) < 10**(-12):
                break
            else:
                y = y1
        return y1

    ### Dranchuk-Abu-Kassem correlation parameters
    def R1(self):
        R1 = A1 + A2/self.Tpr() + A3/self.Tpr()**3 + A4/self.Tpr()**4 + A5/self.Tpr()**5
        return R1

    def R2(self):
        R2 = 0.27*self.Ppr()/self.Tpr()
        return R2

    def R3(self):
        R3 = A6 + A7/self.Tpr() + A8/self.Tpr()**2
        return R3

    def R4(self):
        R4 = A9*(A7/self.Tpr() + A8/self.Tpr()**2)
        return R4

    def R5(self):
        R5 = A10/self.Tpr()**3
        return R5

    def Dranchuk_Abu_Kassem(self):
        y = 0.27*self.Ppr()/self.Tpr()
        for i in range(1000):
            fy = self.R1()*y - self.R2()/y + self.R3()*y**2 - self.R4()*y**5 + self.R5()*(1+A11)*y**2*math.exp(-A11*y**2)+1
            dfy = y + self.R2()/y**2 + 2*y*self.R3() - 5*y**4*self.R4() + (1 - y**2*A11)*(2*self.R5()*(1+A11)*y*math.exp(-A11*y**2))
            y1 = y - fy/dfy
            if abs(y1 - y) < 10**(-12):
                break
            else:
                y = y1
        return y1

    # Calculate final result
    def calculate_z(self):
        if self.correlation == 'Hall-Yarborough':
            return ((0.06125*self.Ppr()*self.t())/self.Hall_Yarborough())*math.exp(-1.2*(1-self.t())**2)
        elif self.correlation == 'Dranchuk-Purvis-Robinson':
            return self.T5()/self.Dranchuk_Purvis_Robinson()
        elif self.correlation == 'Dranchuk-Abu-Kassem':
            return self.R2()/self.Dranchuk_Abu_Kassem()

class gasViscosity():
    def __init__(self, getPressure, getTemperature, getGravity, z, co2, h2s, n2, correlation):
        self.getPressure = getPressure
        self.getTemperature = getTemperature
        self.getGravity = getGravity
        self.z = z
        self.co2 = co2
        self.h2s = h2s
        self.n2 = n2
        self.correlation = correlation

    def Tpc(self):
        if all([self.co2 == 0, self.h2s == 0, self.n2 == 0]):
            return 168 + 325*self.getGravity - 12.5*self.getGravity**2
        else:
            Tpc = 168 + 325*self.getGravity - 12.5*self.getGravity**2
            return Tpc - 80*self.co2 + 130*self.h2s - 250*self.n2

    def Tpr(self):
        return self.getTemperature/self.Tpc()

    def Ppc(self):
        if all([self.co2 == 0, self.h2s == 0, self.n2 == 0]):
            return 677 + 15*self.getGravity - 37.5*self.getGravity**2
        else:
            Ppc = 677 + 15*self.getGravity - 37.5*self.getGravity**2
            return Ppc + 440*self.co2 + 600*self.h2s - 170*self.n2

    def Ppr(self):
        return self.getPressure/self.Ppc()

    # Standing correlation
    def uncorrected_viscosity(self):
        return 8.188*10**(-3) + (1.709*10**(-5) - 2.062*10**(-6)*self.getGravity)*self.getTemperature - 6.15*10**(-3)*math.log(self.getGravity)

    def n2_viscosity(self):
        return self.n2*(8.48*10**(-3)*math.log(self.getGravity) + 9.59*10**(-3))

    def co2_viscosity(self):
        return self.co2*(9.08*10**(-3)*math.log(self.getGravity) + 6.24*10**(-3))

    def h2s_viscosity(self):
        return self.h2s*(8.49*10**(-3)*math.log(self.getGravity) + 3.73*10**(-3))

    def viscosity_1(self):
        return self.uncorrected_viscosity() + self.n2_viscosity() + self.co2_viscosity() + self.h2s_viscosity()

    # Lee-Gonzalez-Eakin correlation
    def Ma(self):
        return self.getGravity*air

    def K(self):
        K = ((9.4+0.02*self.Ma())*self.getTemperature**1.5)/(209+19*self.Ma()+self.getTemperature)
        return K

    def X(self):
        return 3.5 + 986/self.getTemperature + 0.01*self.Ma()

    def Y(self):
        return 2.4-0.2*self.X()

    def gasDensity(self):
        return self.getPressure * self.Ma() / (R * self.getTemperature * self.z)
        
    def calculate_viscosity(self):
        if self.correlation == 'Standing-Dempsey':
            A = C0 + C1*self.Ppr() + C2*self.Ppr()**2 + C3*self.Ppr()**3 +self.Tpr()*(C4 + C5*self.Ppr() + C6*self.Ppr()**2 + C7*self.Ppr()**3) + self.Tpr()**2*(C8 + C9*self.Ppr() + C10*self.Ppr()**2 + C11*self.Ppr()**3) + self.Tpr()**3*(C12 + C13*self.Ppr() + C14*self.Ppr()**2 + C15*self.Ppr()**3)
            return (math.exp(A) / self.Tpr()) * self.viscosity_1()
        elif self.correlation == 'Lee-Gonzalez-Eakin':
            return 10**(-4)*self.K()*math.exp(self.X()*(self.gasDensity()/62.4)**self.Y())

class gasFVF():
    def __init__(self, getPressure, getTemperature, getGravity, co2, h2s, n2, correlation):
        self.getGravity = getGravity
        self.getTemperature = getTemperature
        self.getPressure = getPressure
        self.h2s = h2s
        self.co2 = co2
        self.n2 = n2
        self.correlation = correlation
    def Bg(self):
        parent = zFactor(self.getPressure, self.getTemperature, self.getGravity, self.co2, self.h2s, self.n2, self.correlation)
        z = parent.calculate_z()
        return 0.02827 * z * self.getTemperature / self.getPressure

class gasCompressibility():
    def __init__(self, getPressure, getTemperature, getGravity, co2, h2s, n2, correlation):
        self.getGravity = getGravity
        self.getTemperature = getTemperature
        self.getPressure = getPressure
        self.h2s = h2s
        self.co2 = co2
        self.n2 = n2
        self.correlation = correlation

    def Cp(self):
        parent = zFactor(self.getPressure, self.getTemperature, self.getGravity, self.co2, self.h2s, self.n2, self.correlation)
        Tpr = parent.Tpr()
        Ppc = parent.Ppc()
        Ppr = parent.Ppr()
        z = parent.calculate_z()
        pr_density = 0.27 * Ppr / (z * Tpr)
        T1 = A1 + A2/Tpr + A3/Tpr**3 + A4/Tpr**4 + A5/Tpr**5
        T2 = 2 * (A6 + A7/Tpr + A8/Tpr**2) * pr_density
        T3 = -5 * A9 * (A7/Tpr + A8/Tpr**2) * pr_density ** 4
        T4 = (2 * A10 * (1 + A11*pr_density**2 - A11*pr_density**4) * (pr_density*math.exp(-A11*pr_density**2))) / Tpr ** 3
        partial_derivative = T1 + T2 + T3 + T4
        x = (pr_density / z) * partial_derivative
        Cpr = (1 - x / (1 + x)) / Ppr
        return Cpr / Ppc