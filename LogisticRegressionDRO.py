from DroCounterfactual import  DROCounterfactual
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.optimize import minimize_scalar
import math

class LogisticRegressionDRO(DROCounterfactual):

    def train(self,X_s,y_s,seed):
        model = LogisticRegression(random_state=seed)
        model.fit(X_s, y_s)
        return model
    
    def v_D(self,x,p,theta):
        if p<1:
            raise ValueError("p must be >= 1")
        if theta<0:
            raise ValueError("theta cannot be negative")
        
        if theta==0:
            return self.nominalValue(x)
        q = 1 / (1 - (1/p))

    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))

    def Phi(self,lam,modelo,x,p):
        if p<1:
            raise ValueError("p must be >= 1")
        q = 1 / (1 - (1/p))
        z_i = modelo.decision_function([x])[0]
        x_norm = (np.sum(np.abs(x)**q) +1)

        if lam==0:
            return 0
        
        if p==1:
            if lam>= (1/4):
                return self.sigmoid(z_i)
            zMin = math.log((1-2*lam - math.sqrt(1-4*lam))/(2*lam))
            if z_i < zMin:
                return self.sigmoid(z_i)
            zMax = math.log((1-2*lam + math.sqrt(1-4*lam))/(2*lam))
            if zMin <= z_i <= zMax:
                return (1 - 2*lam - math.sqrt(1 - 4*lam))/(1 - math.sqrt(1 - 4*lam)) + lam* (z_i - zMin)
            if z_i > zMax:
                return min(self.sigmoid(z_i), (1 - 2*lam - math.sqrt(1 - 4*lam))/(1 - math.sqrt(1 - 4*lam)) + lam* (z_i - zMin))
        else:
            R = x_norm / ((4*lam*p)**(1/(p-1)))
            
    
    

    def nominalValue(self,x):
        suma = 0
        N=0
        for model in self.modelSample:
            suma += model.predict_proba(x)[0, 1]
            N += 1
        return suma/N
    

    
    
