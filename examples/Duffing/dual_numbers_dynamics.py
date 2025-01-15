import numpy as np
from collections import defaultdict
from scipy.sparse import diags, csr_matrix, csc_matrix

class Dual(object):
    def __init__(self, value: csr_matrix | np.ndarray | float | int = 0.0, dq: csr_matrix | np.ndarray | float | int = 0.0, dqdot: csr_matrix| np.ndarray | float | int = 0.0, dOm: csr_matrix| np.ndarray | float | int = 0.0):
        self.value = value
        self.dq = dq
        self.dqdot = dqdot
        self.dOm = dOm
    
    @classmethod
    def new(cls, value: csr_matrix | np.ndarray | float | int = 0.0 , dq: csr_matrix | np.ndarray | float | int = 0.0, dqdot: csr_matrix | np.ndarray | float | int = 0.0, dOm: csr_matrix | np.ndarray | float | int = 0.0):
        assert isinstance(value, csr_matrix| np.ndarray | float | int), "value needs to be of type csr_matrix, np.ndarray, float, int"
        assert isinstance(dq, csr_matrix| np.ndarray | float | int), "dq needs to be of type csr_matrix, np.ndarray, float, int"
        assert isinstance(dqdot, csr_matrix | np.ndarray | float | int), "dqdot needs to be of type csr_matrix, np.ndarray, float, int"
        assert isinstance(dOm, csr_matrix | np.ndarray| float | int), "dOm needs to be of type csr_matrix, np.ndarray, float, int"
         
        return Dual(value=value, dq=dq, dqdot=dqdot, dOm=dOm)
    
    def __str__(self):
        text = f"{self.value} + {self.dq}*dq + {self.dqdot}*dqdot + {self.dOm}*dOm"
        return text

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.value+other.value, self.dq+other.dq, self.dqdot+other.dqdot, self.dOm+other.dOm)

        return Dual(self.value+other, self.dq, self.dqdot, self.dOm)
                
    def __radd__(self, other):
        return Dual(self.value + other, self.dq, self.dqdot, self.dOm)
    
    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.value-other.value, self.dq-other.dq, self.dqdot-other.dqdot, self.dOm-other.dOm)
        
        return Dual(self.value-other, self.dq, self.dqdot, self.dOm)
        
    def __rsub__(self, other):
        return Dual(other - self.value, -1*self.dq, -1*self.dqdot, -1*self.dOm)
    
    def __mul__(self, other):
        if isinstance(other, Dual):

            value = np.dot(self.value,other.value)
            diff_q = np.dot(self.dq,other.value) + np.dot(self.value,other.dq)
            diff_qdot = np.dot(self.dqdot,other.value) + np.dot(self.value,other.dqdot)
            diff_Om = np.dot(self.dOm, other.value) + np.dot(self.value, other.dOm)

            return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)
        
        elif isinstance(other, float | int):
            value = self.value * other
            diff_q = self.dq * other
            diff_qdot = self.dqdot * other
            diff_Om = self.dOm * other

            return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)
        
        elif isinstance(other, csr_matrix | np.ndarray):
            value = np.dot(self.value, other)
            diff_q = np.dot(self.dq, other)
            diff_qdot = np.dot(self.dqdot, other)
            diff_Om = np.dot(self.dOm, other)

            return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)

        else:
            exit("Dual: Method __mul__ expects parameter other to be instance of Dual | float | int | np.ndarray | csr_matrix")

    def __rmul__(self, other):
        if isinstance(other, csr_matrix | np.ndarray):

            value = np.dot(other, self.value)
            diff_q = np.dot(other, self.dq)
            diff_qdot = np.dot(other, self.dqdot)
            diff_Om = np.dot(other, self.dOm)

            return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)
        
        else:
            return self*other

    def exp(self):
        value = np.exp(self.value)

        return Dual(value=value, dq=self.dq*value, dqdot=self.dqdot*value, dOm=self.dOm*value)

    def __pow__(self, power):
        if isinstance(power, Dual):
            return Dual.exp(Dual.log(self) * power)
        
        elif isinstance(power, float | int):
            aux = self.value**(power-1)
            value = self.value*aux
            diff = power * aux

            return Dual(value=value, dq=self.dq*diff, dqdot=self.dqdot*diff, dOm=self.dOm*diff)
        
        else:
            exit("Dual: Method __pow__ expects parameter power to be instance of Dual | float | int")

    def __rpow__(self, base):
        return Dual.exp(self * np.log(base))
    
    def power(self, power):
        if isinstance(power, Dual):
            return Dual.exp(Dual.log(self)*power)
        
        elif isinstance(power, float | int):
            aux = self.value.power(power-1)
            value = self.value.multiply(aux)
            diff = power * diags(aux.A.ravel()).tocsr()

            return Dual(value=value, dq=diff.dot(self.dq), dqdot=diff.dot(self.dqdot), dOm=diff.dot(self.dOm))
        
        else:
            exit("Dual: Method __pow__ expects parameter power to be instance of Dual | float | int")

    def __truediv__(self, other):
        if isinstance(other, Dual):
            value = self.value/other.value
            diff_q = (other.value*self.dq - self.value*other.dq)/(other.value**2)
            diff_qdot = (other.value*self.dqdot - self.value*other.dqdot)/(other.value**2)
            diff_Om = (other.value*self.dOm - self.value*other.dOm)/(other.value**2)
            
            return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)
        
        elif isinstance(other, float | int):
            value = self.value/other
            diff_q = self.dq/other
            diff_qdot = self.dqdot/other
            diff_Om = self.dOm/other

            return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)
        
        else:
            exit("Dual: Method __truediv__ expects parameter other to be instance of Dual | float | int")


    def __rtruediv__(self, other):
        if isinstance(other, float | int):
            value = other/self.value
            diff_q = self.dq * (-((other)/(self.value**2)))
            diff_qdot = self.dqdot * (-((other)/(self.value**2)))
            diff_Om = self.dOm * (-((other)/(self.value**2)))

            return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)
        
        else:
            exit("Dual: Method __rtruediv__ expects parameter other to be instance of float | int")
                                
    def multiply(self, other):
        assert isinstance(other, Dual), "Dual: Method multiply expects parameter other to be of instance Dual"

        value = np.multiply(self.value,other.value)
        diff_q = np.multiply(self.dq,other.value) + np.multiply(other.dq,self.value)
        diff_qdot = np.multiply(self.dqdot,other.value) + np.multiply(other.dqdot,self.value)
        diff_Om = np.multiply(self.dOm,other.value) + np.multiply(other.dOm,self.value)

        return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)
    
    def dot(self, other):
        if isinstance(self, csr_matrix | np.ndarray):
            if isinstance(other, Dual):
                value = np.dot(self, other.value)
                diff_q = np.dot(self, other.dq)
                diff_qdot = np.dot(self, other.dqdot)
                diff_Om = np.dot(self, other.dOm)

                return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)
            
            else:
                return np.dot(self, other)
        
    
    def cos(self):
        value = np.cos(self.value)
        diff_q = -1*np.sin(self.value) * self.dq
        diff_qdot = -1*np.sin(self.value) * self.dqdot
        diff_Om = -1*np.sin(self.value) * self.dOm

        return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)

    def sin(self):
        value = np.sin(self.value)
        diff_q = np.cos(self.value) * self.dq
        diff_qdot = np.cos(self.value) * self.dqdot
        diff_Om = np.cos(self.value) * self.dOm

        return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)
    
    def tan(self):
        value = np.tan(self.value)
        diff_q = 1/(np.cos(self.value)**2) * self.dq
        diff_qdot = 1/(np.cos(self.value)**2) * self.dqdot
        diff_Om = 1/(np.cos(self.value)**2) * self.dOm

        return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)

    def log(self):
        value = np.log(self.value)
        diff_q = self.dq/self.value
        diff_qdot = self.dqdot/self.value
        diff_Om = self.dOm/self.value

        return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)
    
    def arctan(self):
        value = np.arctan(self.value)
        diff_q = 1/(self.value**2 + 1)*self.dq
        diff_qdot = 1/(self.value**2 + 1)*self.dqdot
        diff_Om = 1/(self.value**2 + 1)*self.dOm


        return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)

    def tanh(self):
        value = np.tanh(self.value)
        derivative = (1 - value**2)
        diff_q = derivative*self.dq
        diff_qdot = derivative*self.dqdot
        diff_Om = derivative*self.dOm

        return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)
    
    def sqrt(self):
        value = np.sqrt(self.value)
        diff_q = (1/(2*np.sqrt(self.value)))*self.dq
        diff_qdot = (1/(2*np.sqrt(self.value)))*self.dqdot
        diff_Om = (1/(2*np.sqrt(self.value)))*self.dOm

        return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)

    def __abs__(self):
        value = abs(self.value)
        sign = np.sign(self.value)
        diff_q = self.dq*sign
        diff_qdot = self.dqdot*sign
        diff_Om = self.dOm*sign

        return Dual(value=value, dq=diff_q, dqdot=diff_qdot, dOm=diff_Om)
    
    def __lt__(self, other):
        if isinstance(other, Dual):
            return self.value <= other.value
        elif isinstance(other, float | int):
            return self.value <= other
        else: exit("Dual: Method __le__ expects parameter 'other' to be instance of 'Dual | float | int'")

    def __le__(self, other: float | int):
        if isinstance(other, Dual):
            return self.value <= other.value
        elif isinstance(other, float | int):
            return self.value <= other
        else: exit("Dual: Method __le__ expects parameter 'other' to be instance of 'Dual | float | int'")   

    def __gt__(self, other: float | int):
        if isinstance(other, Dual):
            return self.value > other.value
        elif isinstance(other, float | int):
            return self.value > other
        else: exit("Dual: Method __gt__ expects parameter 'other' to be instance of 'Dual | float | int'")   
        
    def __ge__(self, other: float | int):
        if isinstance(other, Dual):
            return self.value >= other.value
        elif isinstance(other, float | int):
            return self.value >= other
        else: exit("Dual: Method __ge__ expects parameter 'other' to be instance of 'Dual | float | int'") 

    def sign(self):
        return Dual(value=np.sign(self.value), dq=0, ddot=0, dOm=0)











