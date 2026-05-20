from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed


SEED = 1234

class DROCounterfactual(ABC):

    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.modelSample = np.array([])

    @abstractmethod
    def train(self, X_s, Y_s, seed):
        pass

    def fit(self, N=100):
        data_l = self.X.shape[0]

        # 1. Semilla maestra
        ss = np.random.SeedSequence(SEED)

        # 2. Generamos 2N semillas independientes
        #    (N para bootstrap, N para training)
        child_seeds = ss.spawn(2 * N)

        bootstrap_seeds = child_seeds[:N]
        model_seeds = child_seeds[N:]

        def one_fit(i):
            # -------------------------
            # Bootstrap RNG
            # -------------------------
            rng_boot = np.random.default_rng(bootstrap_seeds[i])

            idx = rng_boot.choice(data_l, size=data_l, replace=True)
            X_s = self.X[idx]
            y_s = self.y[idx]

            # -------------------------
            # Seed del modelo (independiente)
            # -------------------------
            model_seed = model_seeds[i].entropy

            return self.train(X_s, y_s, model_seed)

        # 3. Paralelización
        data_sample = Parallel(n_jobs=-1)(
            delayed(one_fit)(i) for i in range(N)
        )

        self.modelSample = np.array(data_sample, dtype=object)
    
    def v_D(self,x,p,theta):
        if p<1:
            raise ValueError("p must be >= 1")
        if theta<0:
            raise ValueError("theta cannot be negative")
        
        if theta==0:
            return self.nominalValue(x)
        

        def g(t):
            lam = np.exp(t)
            vals = Parallel(n_jobs=-1)(delayed(self.Phi)(lam, modelo, x, p) for modelo in self.modelSample)

            avg_phi = np.mean(vals)
            return -(-lam*theta + avg_phi )
        
        res = minimize_scalar(g,method='brent')
        return - res.fun
    
    def search(self,x0,p,eps,theta,value=False):
        #Se va a considerar la distancia entre x como ||.||_1 + ||.||_0
        d = lambda x1,x2 : np.linalg.norm(x1-x2, ord=1) + np.linalg.norm(x1-x2, ord=0)

        dist = np.array([d(x, x0) for x in self.X])
        # 2. Filtramos candidatos
        idxs = np.where(dist <= eps)[0]
        if len(idxs) == 0:
            raise ValueError("The value of epsilon must be increased in order to find any candidate")

        # 3. Loop solo sobre candidatos
        best_val = -np.inf
        best_x = None

        for i in idxs:
            x = self.X[i]
            val = self.v_D(x,p,theta)
            if val > best_val:
                best_val = val
                best_x = x
        if value:
            return best_x,best_val

        return best_x


    @abstractmethod
    def nominalValue(self,x):
        pass

    def percentile(self,perc):
 
        scores = np.array([self.nominalValue(x) for x in self.X])
        idx_sorted = np.argsort(scores)

        n = self.X.shape[0]
        pos = int(np.floor((perc / 100) * (n - 1)))

        return self.X[idx_sorted[pos]]

    def plotPareto(self,perc=[10,20,30,40,50],epsList=[90.0,92.0,94.0,96.0,98.0,100.0],lowTheta=0,highTheta=0.01,p=2):
        lista_perc = [self.percentile(p) for p in perc]
        lista = []
        for ind in lista_perc:
            listaUp = []
            listaLow = []
            for eps in epsList:
                listaUp.append(self.search(ind,p,eps,lowTheta,True)[1])
                listaLow.append(self.search(ind,p,eps,highTheta,True)[1])
            lista.append((listaUp,listaLow))
        
        plt.figure(figsize=(8, 5))

        n = len(perc)

        labels = [f"Percentile {i}" for i in perc]

        if n!=5:
            colors = plt.cm.viridis(np.linspace(0, 1, n))
        else:
            colors = ["red","yellow","green","blue","purple"]
        for i, (y_up, y_low) in enumerate(lista):

            # línea superior (sólida)
            plt.plot(
                epsList, y_up,
                color=colors[i],
                linestyle='-',
                label=labels[i]
            )

            # línea inferior (discontinua)
            plt.plot(
                epsList, y_low,
                color=colors[i],
                linestyle='--'
            )

            # puntos superiores (rellenos)
            plt.plot(
                epsList, y_up,
                'o',
                color=colors[i],
                markerfacecolor=colors[i]
            )

            # puntos inferiores (vacíos)
            plt.plot(
                epsList, y_low,
                'o',
                color=colors[i],
                markerfacecolor='none',
                markeredgecolor=colors[i]
            )

            # sombreado entre curvas
            plt.fill_between(
                epsList, y_low, y_up,
                color=colors[i],
                alpha=0.2
            )

        plt.xlabel("x")
        plt.ylabel("value")
        plt.grid(True)
        plt.legend()
        plt.show()



    
