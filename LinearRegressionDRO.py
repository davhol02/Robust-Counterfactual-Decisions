from DroCounterfactual import  DROCounterfactual
from sklearn.linear_model import LinearRegression
import numpy as np



class LinearRegressionDRO(DROCounterfactual):

    def train(self,X_s,y_s,seed):
        model = LinearRegression()
        model.fit(X_s, y_s)
        return (model.coef_,model.intercept_)
    
    def v_D(self,x,p,theta):
        if p<1:
            raise ValueError("p must be >= 1")
        if theta<0:
            raise ValueError("theta cannot be negative")
        
       
        if theta==0:
            return self.nominalValue(x)
        
        if p==1:
            factor = max(np.linalg.norm(x, ord=np.inf),1)
        else:
            q = p / (p-1)
            factor = (np.sum(np.abs(x)**q) +1) ** (1/q)
        return - (theta * factor) + self.nominalValue(x)
    
    

    def nominalValue(self,x):
        suma = 0
        N=0
        for peso,ind in self.modelSample:
            suma += peso @ x + ind
            N += 1
        return suma/N
    

    
    
