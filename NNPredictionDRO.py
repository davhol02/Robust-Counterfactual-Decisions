import torch
import torch.nn as nn
import torch.optim as optim
from DroCounterfactual import  DROCounterfactual
import numpy as np
import math
from scipy.optimize import minimize_scalar
import random

class SimpleNN(nn.Module):
            def __init__(self, n_in, m):
                super().__init__()
                self.fc1 = nn.Linear(n_in, m)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(m, 1)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))
            
            def predict(self, x):
                x = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    return self.forward(x)
                
            def get_weights_numpy(self):
                return {
                    "fc1_weight": self.fc1.weight.detach().cpu().numpy(),
                    "fc1_bias": self.fc1.bias.detach().cpu().numpy().reshape(-1),
                    "fc2_weight": self.fc2.weight.detach().cpu().numpy().reshape(-1),
                    "fc2_bias": self.fc2.bias.detach().cpu().numpy().reshape(-1).item(),
                }

class NNPredictionDRO(DROCounterfactual):

    def __init__(self,X,y,m=16,lr=1e-3,epochs=1000,weight_decay=1e-4, patience=50, val_split=0.2):
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

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val = float("inf")
        best_state = None
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
    
    def v_D(self,x,p,theta):
        if p<2:
            raise ValueError("p must be >= 2")
        if theta<0:
            raise ValueError("theta cannot be negative")
        
        if theta==0:
            return self.nominalValue(x)
        q = 1 / (1 - (1/p))

        x_norm = (np.sum(np.abs(x)**q) +1)

        def g(lam):
            vals = np.array([self.Phi(lam, modelo, x, p) for modelo in self.modelSample])

            avg_phi = np.mean(vals)
            return -(-lam*theta + avg_phi )
        
        res = minimize_scalar(g,method='bounded',bounds=(math.sqrt(x_norm)/2,1e6))
        return - res.fun
        
        
    def Phi(self,lam,modelo,x,p):
        q = 1 / (1 - (1/p))
        W1 = modelo.weights_cache["fc1_weight"]
        b1 = modelo.weights_cache["fc1_bias"]
        W2 = modelo.weights_cache["fc2_weight"]
        b2 = modelo.weights_cache["fc2_bias"]
        res = -(1/q) * ((lam*p)**(1-q)) + b2
        x_norm = (np.sum(np.abs(x)**q) +1)
        
        infimo=0
        if p==2:
            for j in range(W1.shape[0]):
                W1_j = W1[j]
                b1_j = b1[j]
                W2_j = W2[j]
                z_j = W1_j @ x + b1_j
                
                if 4*(lam**2) < x_norm:
                    infimo += - np.inf
                    return infimo
                if (4*(lam**2) == x_norm) and (W2_j < (z_j/math.sqrt(x_norm))):
                    infimo += - np.inf
                    return infimo
                if (4*(lam**2) == x_norm) and (W2_j >= (z_j/math.sqrt(x_norm))):
                    infimo += lam *((max(0,z_j)**2)/(x_norm))
                if (4*(lam**2) > x_norm) and ((-W2_j*x_norm + 2*lam*z_j) < 0):
                    infimo += lam *((max(0,z_j)**2)/(x_norm))
                if (4*(lam**2) > x_norm) and ((-W2_j*x_norm + 2*lam*z_j) >= 0):
                    numerador = 4*lam*W2_j*z_j - (z_j**2) - x_norm*(W2_j**2)
                    alt = (lam*numerador)/(4*(lam**2) - x_norm)
                    infimo += min(alt,lam *((max(0,z_j)**2)/(x_norm)))
            return res + infimo
        else:
            for j in range(W1.shape[0]):
                W1_j = W1[j]
                b1_j = b1[j]
                W2_j = W2[j]
                z_j = W1_j @ x + b1_j

                if z_j<=0:
                    def g1(z):
                        return W2_j - ((lam*p)**(1-q))*(z**(q-1)) + (lam*p * ((z - z_j)**(p-1))) /(x_norm**(p-1))
                    solucionEstrella = minimize_scalar(g1,method='bounded',bounds=(1e-10,1e6))
                    valorEstrella = solucionEstrella.fun
                    if valorEstrella >=0:
                        infimo += 0 
                    else:
                        zEstrella = solucionEstrella.x
                        def g2(z):
                            return W2_j*z - (1/q)*((lam*p)**(1-q))*(z**q) + (lam*((z-z_j)**p)) / (x_norm**(p-1))
                        minimoIntervaloEstrella = minimize_scalar(g2,method='bounded',bounds=(zEstrella,1e6))
                        infimo+= min(minimoIntervaloEstrella.fun,0)
                else:
                    def g1(z):
                        return W2_j - ((lam*p)**(1-q))*(z**(q-1)) + (lam*p * ((z - z_j)**(p-1))) /(x_norm**(p-1))
                    solucionEstrella = minimize_scalar(g1,method='bounded',bounds=(z_j,1e6))
                    valorEstrella = solucionEstrella.fun
                    if valorEstrella >=0:
                        infimoProvisional =  lam *((z_j**p)/(x_norm**(p-1)))
                    else:
                        zEstrella = solucionEstrella.x
                        def g2(z):
                            return W2_j*z - (1/q)*((lam*p)**(1-q))*(z**q) + (lam*((z-z_j)**p)) / (x_norm**(p-1))
                        minimoIntervaloEstrella = minimize_scalar(g2,method='bounded',bounds=(zEstrella,1e6))
                        infimoProvisional= min(minimoIntervaloEstrella.fun,lam *((z_j**p)/(x_norm**(p-1))))
                    def g3(z):
                        return W2_j*z - (1/q)*((lam*p)**(1-q))*(z**q) + (lam*((z_j-z)**p)) / (x_norm**(p-1))
                    solucionIzquierda = minimize_scalar(g3,method='bounded',bounds=(1e-12, z_j - 1e-12))
                    infimo += min(infimoProvisional,solucionIzquierda.fun,g3(z_j)) 
            return res + infimo


        

        

    
    

    def nominalValue(self,x):
        preds = torch.stack([m.predict(x) for m in self.modelSample])
        return preds.mean(dim=0).item()
    