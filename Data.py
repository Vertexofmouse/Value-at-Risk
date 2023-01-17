# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 12:06:41 2023

@author: Nagatorinne
"""

import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


class Brownian:
    def __init__(self, stop=50, K=1000):
        self.K = stop*(K//stop)
        self.stop = stop
        self.L = np.linspace(0, self.stop, self.K)
        self.discrete_L = np.arange(0,self.K, self.K//self.stop)
    
    def process(self): # Wiener process
        mean=0
        sd = np.sqrt(self.stop/self.K)
        M = np.random.normal(mean, sd, self.K)
        rvs = np.array(M.cumsum())
        B = np.array([self.L, rvs]).transpose()
        return B
    
    def discrete_process(self):
        b = self.process()
        return b[self.discrete_L]
    
    def transform(self, B, F):
        '''
        F: function mapping (t, B_t) to X_t
        B: Brownian.process()
        '''
        res = np.apply_along_axis(F, 1, B)
        X = np.array([self.L, res]).transpose()
        return X
    
    def discrete_transform(self, B, F):
        '''
        F: function mapping (t, B_t) to X_t
        B: Brownian.discrete_process()
        '''
        res = np.apply_along_axis(F, 1, B)
        X = np.array([self.L[self.discrete_L], res]).transpose()
        return X
    
    def discrete_transform_2(self, F):
        b = self.discrete_process()
        return self.discrete_transform(b, F)


class data:
    def __init__(self, window_size):
        path = 'all_stocks_5yr/spx.csv'
        self.df = pd.read_csv(path)
        self.df_test = None
        self.test_size = 60
        self.test_true_VaR = 0
        self.test_sigma = 0
        self.test_mu = 0
        self.window_size=window_size
        self.size = self.df.shape[0]
        self.drift=0
        self.sigma=0
        self.R_s = 0
        
    def make_test_df(self, mu=0.002, sigma=0.05):
        size = self.window_size +1
        B = Brownian(stop=size, K=1000)
        F = lambda x: 500*np.exp((mu-0.5*(sigma**2))*x[0] + sigma*x[1])
        X = B.discrete_transform_2(F)
        self.df_test = pd.DataFrame()
        self.test_size = size
        self.test_sigma = sigma
        self.test_mu = mu
        self.df_test['close'] = X[:,1]
        inverse_05_quant = norm.ppf(0.05)
        end_time =X[-1,0]
        self.test_true_VaR = np.exp(sigma*np.sqrt(end_time)*inverse_05_quant + np.log(500) + (mu - 0.5*(sigma**2))*end_time)
        #print(self.test_true_VaR)
        
    def running_mean(self):
        W = self.df['close'].rolling(self.window_size).mean()
        D1.df['running_mean'] = W
        D1.df.loc[np.isnan(W), 'running_mean'] = 0
    
    def plot_running_mean(self):
        self.running_mean()
        fig = plt.figure()
        Y = self.df.loc[self.df.index > np.amax(self.df.index)-1000]
        Xticks = [Y['date'][i] for i in Y.index if (i% 200) ==0]
        plt.plot(Y['date'],Y['close'])
        plt.plot(Y['date'],Y['running_mean'])
        plt.xticks(Xticks)
        plt.show()
        
    def param_estim(self, to_time, test=False):
        if test:
            data = self.df_test
            size = self.test_size
            time = self.test_size-1
        else:
            data = self.df
            size = self.size
            time = to_time
        slice_1 = data.loc[ data.index > time-self.window_size-1, 'close']
        slice_2 = slice_1.loc[ slice_1.index <= time]
        W = slice_2.rolling(2).apply(lambda x: np.log(x.tolist()[1]/x.tolist()[0]))
        W = W.loc[ W.index > time-60].values[1:]
        n = W.shape[0]
        self.R_s = W
        
        s = 0
        for k in range(5000): # Bootstrapping data
            if k==0:
                print('bootstrapping')
            sample = np.random.choice(W, size=n, replace=True)
            d = sample.mean() +0.5*(s**2)
            s = np.sqrt((1/(n-1))*(np.sum(sample-d)**2))
        drifts = np.zeros(10000)
        sigmas = np.zeros(10000)
        for k in range(5000): # Bootstrapping data
            if k==0:
                print('bootstrapping again')
            sample = np.random.choice(W, size=n, replace=True)
            drifts[k] = sample.mean()+0.5*(s**2)
            sigmas[k] = np.sqrt((1/(n-1))*(np.sum(sample-drifts[k])**2))
        self.drift = drifts.mean()
        self.sigma = sigmas.mean()
        
    def MC_ds_VaR(self, to_time, alpha=0.05, test=False):
        if test:
            data = self.df_test
            size = self.test_size
            time = self.test_size-1
        else:
            data = self.df
            size = self.window_size
            time = to_time
        print('size: ', size)
        #B = Brownian(size, 500)
        
        S_0 = data.iloc[time-60]['close']
        
        '''
        domain = np.linspace(0, 2*np.amax(data['close']), 1000)
        bin_width = 2*np.amax(data['close'])/1000
        emp_CDF = np.zeros(1000)
        for i in range(10000):
            if (i % 1000)==0:
                print('MC it: ', i, 'out of: ', 10000)
                
            N = np.random.normal(0, np.sqrt(time))
            X = S_0*np.exp((self.drift-0.5*(self.sigma**2))*time + self.sigma*N)
            '''
            #b = B.discrete_process()
            #F = lambda x: S_0*np.exp((self.drift-0.5*(self.sigma**2))*x[0] + self.sigma*x[1])
            #X = B.discrete_transform(b, F)
        '''
            
            
            
            for k in range(1000):
                if X <= domain[k]:
                    emp_CDF[k] +=1
        emp_CDF = emp_CDF/(1000.0*bin_width)
        for j in range(999):
            if emp_CDF[j] <= alpha and emp_CDF[j+1] >= alpha:
                VaR = domain[j]
        '''
        emp_CDF = np.zeros(10000)
        for i in range(10000):
            if (i % 1000)==0:
                print('MC it: ', i, 'out of: ', 10000)
            N = np.random.normal(0, np.sqrt(time))
            X = S_0*np.exp((self.drift-0.5*(self.sigma**2))*time + self.sigma*N)
            emp_CDF[i]=X
        G = np.sort(emp_CDF)
        VaR = G[int(10000*alpha)]
        return VaR
    
    def kernel_h_val(self, W, K, S_0, alpha, to_time): # W np.array shape (N,)
        P = np.linspace(0.1,5, 20)
        H_res = np.zeros(20)
        n = W.shape[0]
        
        s = 0
        for k in range(500): # Bootstrapping data
            if k==0:
                print('bootstrapping')
            sample = np.random.choice(W, size=n, replace=True)
            d = sample.mean() +0.5*(s**2)
            s = np.sqrt((1/(n-1))*(np.sum(sample-d)**2))
        drifts = np.zeros(10000)
        sigmas = np.zeros(10000)
        for k in range(5000): # Bootstrapping data
            if k==0:
                print('bootstrapping again')
            sample = np.random.choice(W, size=n, replace=True)
            drifts[k] = sample.mean()+0.5*(s**2)
            sigmas[k] = np.sqrt((1/(n-1))*(np.sum(sample-drifts[k])**2))
        mu = drifts.mean()
        sigma = sigmas.mean()
        
        for p in range(20):
            h = K**(-P[p])
            
            samples = list()
            boots = list()
            for k in range(K//10): # Bootstrapping data
                boot = np.random.choice(W, size=n, replace=True)
                boots.append(boot)
                
                boot = boot.reshape((n, 1))
                N = np.random.normal(np.matmul(boot, np.ones((1,n))), h) # shape: (n,size)
                U = np.random.choice(n, n)
                samples.append(N[U, np.arange(n)])
            samples = np.array(samples).reshape(n*(K//10))
            boots = np.concatenate(boots)
            emp_CDF = S_0*np.exp(np.sum(samples.reshape(K//10, n), axis=1))
            G = np.sort(emp_CDF)
            VaR = G[int(K*alpha)]
            
            @np.vectorize
            def test_density(x):
                return (1/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*(mu-x)**2/(sigma**2)))
        
            @np.vectorize
            def estim_density(x):
                Kernel = (1/(np.sqrt(2*np.pi))*np.exp(-0.5*(samples-x)**2/(h**2)))
                denom = (K*h)*n
                return (1/denom)*np.sum(Kernel)
                
            X = np.linspace(-0.5, 0.5, 500)
            Res_estim = estim_density(X)
            Res_test = test_density(X)
            H_res[p] = (1/500)*np.sum((Res_estim - Res_test)**2)
        #print('H_res: ', H_res)
        p_0 = np.argmin(H_res)
        return K**(-P[p_0])
        
    
    def MC_kernel_VaR(self, to_time, h_start=0, alpha=0.05, K=1000, test=False, plot=False, prints=False):
        
        if test:
            data = self.df_test
            size = self.test_size
            time = self.test_size-1
        else:
            data = self.df
            size = self.size
            time = to_time
        slice_1 = data.loc[ data.index > time-self.window_size-1, 'close']
        slice_2 = slice_1.loc[ slice_1.index <= time]
        W = slice_2.rolling(2).apply(lambda x: np.log(x.tolist()[1]/x.tolist()[0]))
        W = W.loc[ W.index > time-self.window_size]
        n = W.shape[0]
        #index_val = np.random.choice(n, n//5)
        #index_train = np.delete(np.arange(n), index_val)
        #W_val = W.iloc[index_val]
        #W = W.iloc[index_train]
        S_0 = data.iloc[time-60]['close']
        if h_start==0:
            h = self.kernel_h_val(W_val, K, S_0, alpha, time)
            print('validated h: ', h, 'heuristic h: ', K**(-0.6))
        else:
            h = h_start
        
        samples = list()
        boots = list()
        for k in range(K): # Bootstrapping data
            if k==0 and prints:
                print('bootstrapping')
            boot = np.random.choice(W, size=n, replace=True)
            boots.append(boot)
            
            boot = boot.reshape((n, 1))
            N = np.random.normal(np.matmul(boot, np.ones((1,n))), h) # shape: (n,size)
            U = np.random.choice(n, n)
            samples.append(N[U, np.arange(n)])
        samples = np.array(samples).reshape(n*K)
        boots = np.concatenate(boots)
        
        if prints:
            print('samples size, boots size: ', samples.shape, boots.shape)
        
        if plot:
            @np.vectorize
            def estim_density(x):
                Kernel = (1/(np.sqrt(2*np.pi))*np.exp(-0.5*(samples-x)**2/(h**2)))
                denom = (K*h)*n
                return (1/denom)*np.sum(Kernel)
            
            @np.vectorize
            def test_density(x):
                return (1/(np.sqrt(2*np.pi)*self.test_sigma)*np.exp(-0.5*(self.test_mu-x)**2/(self.test_sigma**2)))
            
            X = np.linspace(-0.5, 0.5, 500)
            plt.plot(X, estim_density(X), label='estimated density')
            plt.plot(X, test_density(X), label='test density')
            plt.legend()
        
        emp_CDF = S_0*np.exp(np.sum(samples.reshape(K, n), axis=1))
        G = np.sort(emp_CDF)
        VaR = G[int(K*alpha)]
        return VaR
    
    
def test_of_test_set_calc(mu=0.002, sigma=0.05):
    N=60
    K=5000
    res =0
    for k in range(K):
        print('it: ', k)
        D1 = data(N)   
        D1.make_test_df(mu, sigma)
        if k==K-1:
            print('true VaR: ', D1.test_true_VaR)
        if (k % 100) ==0:
            plt.plot(list(range(N+1)), D1.df_test['close'])
        if D1.df_test.iloc[N-1]['close'] <= D1.test_true_VaR:
            res +=1
    print('fraction: ', res/K)


def run_ds_MC(test=True, mu=0.002, sigma=0.05, window=60):
    if test:
        D1 = data(window)
        D1.make_test_df(mu, sigma)
        D1.param_estim(5060, test=True)
        V = D1.MC_ds_VaR(7060, 0.05, test=True)  
        print('running on test set \n True VaR: ', D1.test_true_VaR)
        print('estimated: \n', 'drift: ', D1.drift, '\n', 'sigma: ', D1.sigma)
        print('estimated VaR: ', V)
        plt.plot(D1.df_test.index, D1.df_test['close'])
        plt.plot(D1.df_test.index, [D1.test_true_VaR]*D1.df_test.shape[0])
        plt.plot(D1.df_test.index, [V]*D1.df_test.shape[0])
    else:
        D1 = data(window)
        D1.param_estim(7060)
        V = D1.MC_ds_VaR(7060, 0.05)
        print('drift: ', D1.drift, '\n', 'sigma: ', D1.sigma)
        plt.plot(D1.df.index, D1.df['close'])
        plt.plot(D1.df.index, [V]*D1.df.shape[0])
        
def run_kernel_MC(test=True, mu=0.002, sigma=0.05, window=60, plot=True, it=1):
    res = np.zeros(it)
    if test:
        true_VaR = 0
        for k in range(it):
            print('kernel MC test set iteration: ', k)
            D1 = data(window)
            D1.make_test_df(mu, sigma)
            true_VaR = D1.test_true_VaR
            K_0=1000
            h_0 = K_0**(-0.6)
            V = D1.MC_kernel_VaR(to_time=7060, h_start=h_0, alpha=0.05, K=K_0, test=True, plot=False)
            res[k]=V
            if plot:
                print('running on test set \n True VaR: ', true_VaR)
                print('estimated VaR: ', V)
                plt.plot(D1.df_test.index, D1.df_test['close'])
                plt.plot(D1.df_test.index, [D1.test_true_VaR]*D1.df_test.shape[0])
                plt.plot(D1.df_test.index, [V]*D1.df_test.shape[0])
        return true_VaR, res
    else:
        for k in range(it):
            print('kernel MC real set iteration: ', k)
            D1 = data(window)
            D1.param_estim(7060)
            V = D1.MC_kernel_VaR(7060, 0.05)
            res[k]=V
            if plot:
                print('estimated VaR: ', V)
                plt.plot(D1.df.index, D1.df['close'])
                plt.plot(D1.df.index, [V]*D1.df.shape[0])
        return res
        
#test_of_test_set_calc()
#run_ds_MC(window=60)
true, V_its = run_kernel_MC(it=20, plot=False)
fig = plt.axes()
H = plt.hist(V_its, bins=20)
plt.vlines(true, 0, np.amax(H[0]))
print(V_its)
print('true VaR: ', true, 'mean estimated VaR: ', V_its.mean())

'''
D1 = data(60)
D1.param_estim(5060)
print('drift: ', D1.drift, '\n', 'sigma: ', D1.sigma)
V = D1.MC_VaR(7060, 0.05)
'''



