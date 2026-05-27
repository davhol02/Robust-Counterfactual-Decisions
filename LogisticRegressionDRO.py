from DroCounterfactual import  DROCounterfactual
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.optimize import minimize_scalar
import math
from scipy.special import expit


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


        def g(lam):
            vals = np.array([self.Phi(lam, modelo, x, p) for modelo in self.modelSample])

            avg_phi = np.mean(vals)
            return -(-lam*theta + avg_phi )
        
        res = minimize_scalar(g,method='bounded',bounds=(0,1/(theta**p)))
        return - res.fun


    
    def objective_function(self,lam,p,z_i,x_norm,z):
        return expit(z) + lam*((z_i - z)**p/(x_norm**(p-1)))

    def Phi(self,lam,modelo,x,p):
        if p<1:
            raise ValueError("p must be >= 1")
        z_i = modelo.decision_function([x])[0]
        if lam==0:
            return 0
        
        if p==1:
            if lam>= (1/4):
                return expit(z_i)
            zMin = math.log((1-2*lam - math.sqrt(1-4*lam))/(2*lam))
            if z_i < zMin:
                return expit(z_i)
            zMax = math.log((1-2*lam + math.sqrt(1-4*lam))/(2*lam))
            if zMin <= z_i <= zMax:
                return (1 - 2*lam - math.sqrt(1 - 4*lam))/(1 - math.sqrt(1 - 4*lam)) + lam* (z_i - zMin)
            if z_i > zMax:
                return min(expit(z_i), (1 - 2*lam - math.sqrt(1 - 4*lam))/(1 - math.sqrt(1 - 4*lam)) + lam* (z_i - zMin))
        else:
            q = p / (p-1)
            x_norm = (np.sum(np.abs(x)**q) +1)
            R = x_norm / ((4*lam*p)**(1/(p-1)))
            if z_i==R:
                return self.objective_function(lam,p,z_i,x_norm,0)
            else:
                solucion = minimize_scalar(lambda z: self.objective_function(lam,p,z_i,x_norm,z),method='bounded',bounds=(z_i-R,z_i))
                return solucion.fun
            
    
    

    def nominalValue(self,x):
        suma = 0
        N=0
        for model in self.modelSample:
            suma += model.predict_proba([x])[0, 1]
            N += 1
        return suma/N
    

    
    
