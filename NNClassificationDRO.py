import torch
import torch.nn as nn
import torch.optim as optim
from DroCounterfactual import  DROCounterfactual
import numpy as np
import math
from scipy.optimize import minimize_scalar
import random
from scipy.special import expit
from scipy.optimize import minimize



class SimpleNN(nn.Module):
            def __init__(self, n_in, m):
                super().__init__()
                self.fc1 = nn.Linear(n_in, m)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(m, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

            

            def predict(self, x):
                x = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    logits = self.forward(x)
                return torch.sigmoid(logits)
                
            def get_weights_numpy(self):
                return {
                    "fc1_weight": self.fc1.weight.detach().cpu().numpy(),
                    "fc1_bias": self.fc1.bias.detach().cpu().numpy().reshape(-1),
                    "fc2_weight": self.fc2.weight.detach().cpu().numpy().reshape(-1),
                    "fc2_bias": self.fc2.bias.detach().cpu().numpy().reshape(-1).item(),
                }

class NNClassificationDRO(DROCounterfactual):

    def __init__(self,X,y,m=8,lr=1e-3,epochs=1000,weight_decay=1e-4, patience=50, val_split=0.2):
        super().__init__(X,y)
        self.m = m
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split
        
        

    def train(self,X_s,y_s,seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        X_s = torch.tensor(X_s, dtype=torch.float32)
        y_s = torch.tensor(y_s, dtype=torch.float32).view(-1, 1)

        n = len(X_s)
        g = torch.Generator()
        g.manual_seed(seed)

        idx = torch.randperm(n, generator=g)

        val_size = int(n * self.val_split)
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]

        X_train, y_train = X_s[train_idx], y_s[train_idx]
        X_val, y_val = X_s[val_idx], y_s[val_idx]

        n_in = X_s.shape[1]

        model = SimpleNN(n_in, self.m)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val = float("inf")
        best_state =  None
        patience_counter = 0

        for _ in range(self.epochs):
            model.train()

            pred = model(X_train)
            loss = loss_fn(pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = loss_fn(val_pred, y_val)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        model.load_state_dict(best_state)
        model.weights_cache = model.get_weights_numpy()
        return model
    

    def inner_objective_function(self,lam,p,z_i,x_norm,z):
        return expit(z) + lam*((z_i - z)**p/(x_norm**(p-1)))
    


    def innerInfimum(self, lam, modelo, x, p):
        if p < 1:
            raise ValueError("p must be >= 1")

        W2 = modelo.weights_cache["fc2_weight"]
        b2 = modelo.weights_cache["fc2_bias"]
        z_i = W2 @ x + b2

        if np.isclose(lam, 0):
            return 0

        if p == 1:
            norm_x = max(np.linalg.norm(x, ord=np.inf), 1.0)
            newLam = lam / norm_x

            if newLam <= 0:
                return expit(z_i)

            disc = 1 - 4 * newLam

            if disc <= 1e-12:
                return expit(z_i)

            sqrt_disc = math.sqrt(disc)

            denom = 2 * newLam
            if denom <= 0:
                return expit(z_i)
            num_min = 1 - 2 * newLam - sqrt_disc
            val_min = num_min / denom

            if val_min <= 0 or np.isnan(val_min):
                return expit(z_i)

            zMin = math.log(val_min)

            if z_i < zMin:
                return expit(z_i)

            num_max = 1 - 2 * newLam + sqrt_disc
            val_max = num_max / denom

            if val_max <= 0 or np.isnan(val_max):
                return expit(z_i)

            zMax = math.log(val_max)

            if zMin <= z_i <= zMax:
                denom2 = 1 - math.sqrt(disc)
                if np.isclose(denom2, 0):
                    return expit(z_i)

                return (
                    (1 - 2 * newLam - sqrt_disc) / denom2
                    + newLam * (z_i - zMin)
                )

            if z_i > zMax:
                lin = (
                    (1 - 2 * newLam - sqrt_disc) / (1 - math.sqrt(disc))
                    + newLam * (z_i - zMin)
                )
                return min(expit(z_i), lin)

            return expit(z_i)

        else:
            q = p / (p - 1)
            x_norm = (np.sum(np.abs(x) ** q) + 1)

            R = x_norm / ((4 * lam * p) ** (1 / (p - 1)))

            solucion = minimize_scalar(
                lambda z: self.inner_objective_function(lam, p, z_i, x_norm, z),
                method='bounded',
                bounds=(z_i - R, z_i)
            )

            return solucion.fun
            
    def constraint(self,z, z0, p, M):
        return M - np.sum(np.abs(z - z0)**p)
    
    def objective_function(self,lam,modelo,p,z,z0):
        x= np.maximum(0,z)
        infimo_z_2 = self.innerInfimum(lam,modelo,x,p)
        if p>1:
            q = p / (p-1)
            den = (np.sum(np.abs(x)**q) +1)**(p-1)
        else:
            den = max(np.linalg.norm(x, ord=np.inf),1)
        penalty = lam*(np.sum(np.abs(z-z0)**p) / (den))
        return infimo_z_2 + penalty

    

            


    def Phi(self,lam,modelo,x,p):
        if np.isclose(lam,0):
            return 0
        W1 = modelo.weights_cache["fc1_weight"]
        b1 = modelo.weights_cache["fc1_bias"]
        z0 = W1 @ x + b1
        if p>1:
            q = p / (p-1)
            num = (np.sum(np.abs(x)**q) +1)**(p-1)
        else:
            num = max(np.linalg.norm(x, ord=np.inf),1)
        M= num/lam
        cons = {'type': 'ineq','fun': lambda z: self.constraint(z,z0,p,M)}
        minim = minimize(lambda z: self.objective_function(lam,modelo,p,z,z0),z0,constraints=cons)
        return minim.fun
    

    def v_D(self,x,p,theta):
        if p<1:
            raise ValueError("p must be >= 1")
        if theta<0:
            raise ValueError("theta cannot be negative")
        
        if theta==0:
            return self.nominalValue(x)


        def g(lam):
            print(self.modelSample)
            vals = np.array([self.Phi(lam, modelo, x, p) for modelo in self.modelSample])

            avg_phi = np.mean(vals)
            return -(-lam*theta + avg_phi )
        print(p)
        
        res = minimize_scalar(g,method='bounded',bounds=(0,1/(theta**p)))
        return - res.fun
    
    def nominalValue(self,x):
        preds = torch.stack([m.predict(x) for m in self.modelSample])
        return preds.mean(dim=0).item()