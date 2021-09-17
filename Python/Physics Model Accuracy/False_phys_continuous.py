#!/usr/bin/env python
# coding: utf-8


## Import packages
import numpy as np
import pandas as pd
import random
from datetime import datetime
import seaborn as sn
import matplotlib.pyplot as plt
from modAL.models import ActiveLearner
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, RationalQuadratic, Matern
from modAL.models import ActiveLearner
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.stats import norm
from scipy.special import ndtr
import warnings
from time import perf_counter
plt.switch_backend('agg')


## Objective function class
class obj_fun():
    def __init__(self):
        self.descr = 'This is an objective function'
        self.choices = ['rosen', 'rosen_3', 'branin', 'camel_3', 'sphere', 'sphere_3', 'sphere_4', 'mccor', 'hart_3', 'hart_4', 'hart_6']
        
    def get_obj_fun(self, choice, x, params, *args):
        if choice == 'rosen':
            return self.rosen(x, params)
        elif choice == 'branin':
            return self.branin(x,params)
        elif choice == 'camel_3':
            return self.camel_3(x,params)
        elif choice == 'sphere':
            return self.sphere(x,params)
        elif choice == 'mccor':
            return self.mccor(x,params)
        elif choice == 'hart_3':
            return self.hart_3(x,params)
        elif choice == 'hart_4':
            return self.hart_4(x,params)
        elif choice == 'hart_6':
            return self.hart_6(x,params)
        elif choice == 'rosen_3':
            return self.rosen_3(x,params)
        elif choice == 'sphere_3':
            return self.sphere_3(x,params)
        elif choice == 'sphere_4':
            return self.sphere_4(x,params)
        elif choice == 'sphere_6':
            return self.sphere_6(x,params)
        elif choice == 'rosen_4':
            return self.rosen_4(x,params)
        elif choice == 'rosen_6':
            return self.rosen_6(x,params)
        elif len(choice) == 2:
            ratio = args[0]
            return self.comb(x,params,choice, ratio)
        else:
            return 'Invalid input'
        
    def comb(self, x, params, choices, ratio):
        choice_1 = choices[0]
        choice_2 = choices[1]
        num_params_1 = get_obj_info(choice_1, 'num params')
        params_1 = params[:num_params_1]
        params_2 = params[num_params_1:]
        obj_1 = self.get_obj_fun(choice_1, x, params_1)
        obj_2 = self.get_obj_fun(choice_2, x, params_2)
        return ratio*obj_1 + (1-ratio)*obj_2
    
     ## Rosenbrock
    def rosen(self, x, params):
        return -1*(params[0]*((x[1]+params[2]) - (x[0]+params[3])**2)**2 + (params[1] - (x[0]+params[3]))**2)
    
    def rosen_3(self, x, params):
        val = 0
        for j in range(len(x)-1):
            val = val + -1*(params[0]*((x[j+1]+params[2]) - (x[j]+params[3])**2)**2 + (params[1] - (x[j]+params[3]))**2)
        return val
    
    def rosen_4(self, x, params):
        val = 0
        for j in range(len(x)-1):
            val = val + -1*(params[0]*((x[j+1]+params[2]) - (x[j]+params[3])**2)**2 + (params[1] - (x[j]+params[3]))**2)
        return val
    
    def rosen_6(self, x, params):
        val = 0
        for j in range(len(x)-1):
            val = val + -1*(params[0]*((x[j+1]+params[2]) - (x[j]+params[3])**2)**2 + (params[1] - (x[j]+params[3]))**2)
        return val
    
    ## Branin
    def branin(self, x, params):
        PI = 3.14159265359
        a = params[0] # 1
        b = params[1]/(4*pow(PI, 2)) # 5.1
        c = params[2]/PI #5
        r = params[3] #6
        s = params[4] #10
        t = params[5]/(8*PI) # 1
        return -1*(a*(x[1] - b*x[0]**2 + c*x[0] -r)**2 + s*(1-t)*np.cos(x[0]) + s)
    
    ## 3-hump camel
    def camel_3(self, x, params):
        return -1*((params[0])*x[0]**2 - (params[1])*x[0]**4 + params[2]*(x[0]**6)/6 + params[3]*x[0]*x[1] + params[4]*x[1]**2)
    
    ## Sphere
    def sphere(self, x, params):
        dim = len(x)
        f = 0
        ind = 0
        for val in x:
            f = f + (params[ind]*(val + params[ind+dim]))**2
            ind = ind + 1
        return -f
    
    def sphere_3(self, x, params):
        dim = len(x)
        f = 0
        ind = 0
        for val in x:
            f = f + (params[ind]*(val + params[ind+dim]))**2
            ind = ind + 1
        return -f
    
    def sphere_4(self, x, params):
        dim = len(x)
        f = 0
        ind = 0
        for val in x:
            f = f + (params[ind]*(val + params[ind+dim]))**2
            ind = ind + 1
        return -f
    
    def sphere_6(self, x, params):
        dim = len(x)
        f = 0
        ind = 0
        for val in x:
            f = f + (params[ind]*(val + params[ind+dim]))**2
            ind = ind + 1
        return -f
    
    ## Mccormick
    def mccor(self, x, params):
        return -1*(np.sin(x[0]+x[1]) + (x[0]-x[1])**2 - params[0]*x[0] + params[1]*x[1] + params[2]) # params = 1.5, 2.5, 1
    
    ## 3-D Hartmann
    def hart_3(self, x, params):
        alpha = np.array([1, 1.2, 3, 3.2]) # 1, 1.2, 3, 3.2
        alpha = np.reshape(alpha, (-1,1))
        A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        A = np.reshape(A, (-1,3))
        P = np.array([[.3689, .1170, .2673], [.4699, .4387, .7470], [.1091, .8732, .5547], [.0381, .5743, .8828]])
        P = np.reshape(P, (-1,3))

        f = 0

        for i in range(4):
            second = 0
            for j in range(3):
                second = second - params[0]*A[i,j]*(params[1]*x[j]-params[2]*P[i,j])**2
            f = f - alpha[i]*np.exp(second)

        return -f
    
    ## 4-D Hartmann
    def hart_4(self, x, params):
        alpha = np.array([1, 1.2, 3, 3.2])
        alpha = np.reshape(alpha, (-1,1))
        A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
        A = np.reshape(A, (-1,6))
        P = np.array([[.1312, .1696, .5569, .0124, .8283, .5886], [.2329, .4135, .8307, .3736, .1004, .9991], [.2348, .1451, .3522, .2883, .3047, .6650], [.4047, .8828, .8732, .5743, .1091, .0381]])
        P = np.reshape(P, (-1,6))

        f = 0
        first = 0

        for i in range(4):
            second = 0
            for j in range(4):
                second = second - params[0]*A[i,j]*(params[1]*x[j]-params[2]*P[i,j])**2
            first = first - alpha[i]*np.exp(second)

        f = (1/0.839)*(1.1-first)

        return -f
    
    ## 6-D Hartmann
    def hart_6(self, x, params):
        alpha = np.array([1, 1.2, 3, 3.2])
        alpha = np.reshape(alpha, (-1,1))
        A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
        A = np.reshape(A, (-1,6))
        P = np.array([[.1312, .1696, .5569, .0124, .8283, .5886], [.2329, .4135, .8307, .3736, .1004, .9991], [.2348, .1451, .3522, .2883, .3047, .6650], [.4047, .8828, .8732, .5743, .1091, .0381]])
        P = np.reshape(P, (-1,6))

        f = 0

        for i in range(4):
            second = 0
            for j in range(6):
                second = second - params[0]*A[i,j]*(params[1]*x[j]-params[2]*P[i,j])**2
            f = f - alpha[i]*np.exp(second)
    
        return -f


## acquisition function to minimize - Modified ranked-batch (MRB)
def max_score_ours(X_0, X_known, y_known, X_grid, regressor, params, LB, UB):
    beta = params
    
    X_0 = np.reshape(X_0, (1,-1))
    UB_arr = np.nonzero(X_0 > UB)
    UB_arr = np.asarray(UB_arr)
    LB_arr = np.nonzero(X_0 < LB)
    LB_arr = np.asarray(LB_arr)
    if UB_arr.size == 0 and LB_arr.size == 0:
        X_0
    elif UB_arr.size != 0:
        X_0[0][UB_arr[1]] = (UB+LB)/2
        if LB_arr.size != 0:
            X_0[0][LB_arr[1]] = (UB+LB)/2
    elif LB_arr.size != 0:
        X_0[0][LB_arr[1]] = (UB+LB)/2
        if UB_arr.size != 0:
            X_0[0][UB_arr[1]] = (UB+LB)/2
    
    check_nan = np.isnan(X_0)
    if sum(check_nan[0]) != 0:
        X_0[check_nan] = (UB+LB)/2
        
    test_pred, std = regressor.predict(X_0, return_std=True)
    test_pred = np.reshape(test_pred, (-1,1))
    
    grid_pred, grid_std = regressor.predict(X_grid, return_std=True)
    grid_pred = np.reshape(grid_pred, (-1,1))
    
    distance_scores = pairwise_distances(X_0, X_known, metric='euclidean')
    distance_scores = np.reshape(distance_scores, (1, len(X_known)))
    c = np.amin(distance_scores)
    
    max_grid_val = np.amax(X_grid)
    min_grid_val = np.amin(X_grid)
    dist_max = 0.5*(max_grid_val - min_grid_val)
    
    FE = 0
    STD = 0
    DIST = 0
    fe_max = np.amax(grid_pred)
    fe_min = np.amin(grid_pred)
    if fe_max == fe_min:
        fe_min = np.amin(y_known)
        fe_max = np.amax(y_known)
    FE = (test_pred - fe_min)/(fe_max-fe_min)
    STD = std
    DIST = (c)/(dist_max)
    if DIST < 0:
        DIST = 0
    similarity_score = 1 / (1 + DIST)
    score = beta*(1 - similarity_score) + beta*(STD) + (FE)
    score = np.ravel(score)
    return -score


## Acquistion function to minimize - Probability of improvement (PI)
def max_score_PI(X_0, X_known, y_known, X_grid, regressor, params, LB, UB):
    tradeoff = params
    
    X_0 = np.reshape(X_0, (1,-1))
    UB_arr = np.nonzero(X_0 > UB)
    UB_arr = np.asarray(UB_arr)
    LB_arr = np.nonzero(X_0 < LB)
    LB_arr = np.asarray(LB_arr)
    if UB_arr.size == 0 and LB_arr.size == 0:
        X_0
    elif UB_arr.size != 0:
        X_0[0][UB_arr[1]] = (UB+LB)/2
        if LB_arr.size != 0:
            X_0[0][LB_arr[1]] = (UB+LB)/2
    elif LB_arr.size != 0:
        X_0[0][LB_arr[1]] = (UB+LB)/2
        if UB_arr.size != 0:
            X_0[0][UB_arr[1]] = (UB+LB)/2
    
    check_nan = np.isnan(X_0)
    if sum(check_nan[0]) != 0:
        X_0[check_nan] = (UB+LB)/2
        
    test_pred, std = regressor.predict(X_0, return_std=True)
    test_pred = np.reshape(test_pred, (-1,1))
    
    grid_pred, grid_std = regressor.predict(X_grid, return_std=True)
    grid_pred = np.reshape(grid_pred, (-1,1))
    grid_std = np.reshape(std, (-1,1))
    
    grid_max = np.amax(grid_pred)
    grid_min = np.amin(grid_pred)
    test_pred = (test_pred-grid_min)/(grid_max-grid_min)
    y_known_max = np.amax(y_known)
    y_known_max = (y_known_max-grid_min)/(grid_max-grid_min)
    
    score = norm.cdf((test_pred - y_known_max - tradeoff)/std)
    score = np.ravel(score)
    return -score


## Acquisition function to minimize - Upper confidence bound (UCB)
def max_score_UCB(X_0, X_known, y_known, X_grid, regressor, params, LB, UB):
    beta = params
    
    X_0 = np.reshape(X_0, (1,-1))
    UB_arr = np.nonzero(X_0 > UB)
    UB_arr = np.asarray(UB_arr)
    LB_arr = np.nonzero(X_0 < LB)
    LB_arr = np.asarray(LB_arr)
    if UB_arr.size == 0 and LB_arr.size == 0:
        X_0
    elif UB_arr.size != 0:
        X_0[0][UB_arr[1]] = (UB+LB)/2
        if LB_arr.size != 0:
            X_0[0][LB_arr[1]] = (UB+LB)/2
    elif LB_arr.size != 0:
        X_0[0][LB_arr[1]] = (UB+LB)/2
        if UB_arr.size != 0:
            X_0[0][UB_arr[1]] = (UB+LB)/2
    
    check_nan = np.isnan(X_0)
    if sum(check_nan[0]) != 0:
        X_0[check_nan] = (UB+LB)/2
        
    test_pred, std = regressor.predict(X_0, return_std=True)
    test_pred = np.reshape(test_pred, (-1,1))
    
    grid_pred, grid_std = regressor.predict(X_grid, return_std=True)
    grid_pred = np.reshape(grid_pred, (-1,1))
    grid_std = np.reshape(std, (-1,1))
    
    score = test_pred + beta*std
    score = np.ravel(score)
    return -score


## Acquisition function to minimize - Expected Improvement (EI)
def max_score_EI(X_0, X_known, y_known, X_grid, regressor, params, LB, UB):
    tradeoff = params
    
    X_0 = np.reshape(X_0, (1,-1))
    UB_arr = np.nonzero(X_0 > UB)
    UB_arr = np.asarray(UB_arr)
    LB_arr = np.nonzero(X_0 < LB)
    LB_arr = np.asarray(LB_arr)
    if UB_arr.size == 0 and LB_arr.size == 0:
        X_0
    elif UB_arr.size != 0:
        X_0[0][UB_arr[1]] = (UB+LB)/2
        if LB_arr.size != 0:
            X_0[0][LB_arr[1]] = (UB+LB)/2
    elif LB_arr.size != 0:
        X_0[0][LB_arr[1]] = (UB+LB)/2
        if UB_arr.size != 0:
            X_0[0][UB_arr[1]] = (UB+LB)/2
    
    check_nan = np.isnan(X_0)
    if sum(check_nan[0]) != 0:
        X_0[check_nan] = (UB+LB)/2
    
    test_pred, std = regressor.predict(X_0, return_std=True)
    test_pred = np.reshape(test_pred, (-1,1))
    
    grid_pred, grid_std = regressor.predict(X_grid, return_std=True)
    grid_pred = np.reshape(grid_pred, (-1,1))
    grid_std = np.reshape(std, (-1,1))
    
    grid_max = np.amax(grid_pred)
    grid_min = np.amin(grid_pred)
    test_pred = (test_pred-grid_min)/(grid_max-grid_min)
    y_known_max = np.amax(y_known)
    y_known_max = (y_known_max-grid_min)/(grid_max-grid_min)
    
    z = (test_pred - y_known_max - tradeoff)/std
    score = (test_pred - y_known_max - tradeoff)*norm.cdf(z) + std*norm.pdf(z)
    score = np.ravel(score)
    return -score


## acquisition function class
class acquisition():
    def __init__(self):
        self.descr = 'This is an acquisition function'
        
    ## get_acq_fun calls acquisition function
    def get_acq_fun(self, regressor,choice, X_grid, X_known, y_known, n_instances, params, LB, UB):
        if choice == 'ours':
            return self.ours(regressor, X_grid, X_known, y_known, n_instances, params, LB, UB)
        elif choice == 'PI':
            return self.PI(regressor, X_grid, X_known, y_known, n_instances, params, LB, UB)
        elif choice == 'EI':
            return self.EI(regressor, X_grid, X_known, y_known, n_instances, params, LB, UB)
        elif choice == 'UCB':
            return self.UCB(regressor, X_grid, X_known, y_known, n_instances, params, LB, UB)
        else:
            return 'Invalid input'
    
    ## ours = MRB
    def ours(self, regressor, X_grid, X_known, y_known, n_instances, params, LB, UB):
        
        # Create an array where we're going to put the chosen X, X_new
        dims = X_known.shape[1]
        beta = params
        X_new = np.empty((0,dims))
        bnds = Bounds(LB[0], UB[0], keep_feasible=True)

        for j in range(0, n_instances):
            
            best_options = np.zeros((25,dims))
            best_scores = np.zeros((25,1))
            for k in range(0,25):
                x0 = np.zeros((dims))
                for col in range(dims):
                    x0[col] = np.random.uniform(LB[col]+0.001, UB[col]-0.001)
                res = minimize(max_score_ours, x0, args= (X_known, y_known, X_grid, regressor, params, LB[0], UB[0]), method='L-BFGS-B',bounds = bnds)
                best_options[k] = res.x
                best_scores[k] = res.fun
            
            if j == 0:
                best_ind = np.argmin(best_scores)
                X_new = best_options[best_ind,:]
                X_new = np.reshape(X_new, (-1,dims))
            else:
                best_ind = np.argmin(best_scores)
                X_new_ = best_options[best_ind,:]
                X_new = np.append(X_new, X_new_)
                X_new = np.reshape(X_new, (-1,dims))
            
            X_known = np.append(X_known, X_new[j])
            X_known = np.reshape(X_known, (-1,dims))
            X_new_now = np.reshape(X_new[j], (1,-1))
            y_fake_new = regressor.predict(X_new_now)
            y_fake_new = np.reshape(y_fake_new, (-1,1))
            y_known = np.append(y_known, y_fake_new)
            y_known = np.reshape(y_known, (-1,1))
            regressor.fit(X_known, y_known)

        return X_new, dims

    def PI(self, regressor, X_grid, X_known, y_known, n_instances, params, LB, UB):
        dims = X_known.shape[1]
        tradeoff = params
        X_new = np.empty((0,dims))
        bnds = Bounds(LB[0], UB[0], keep_feasible=True)
        
        for j in range(0, n_instances):
            
            best_options = np.zeros((25,dims))
            best_scores = np.zeros((25,1))
            for k in range(0,25):
                x0 = np.zeros((dims))
                for col in range(dims):
                    x0[col] = np.random.uniform(LB[col]+0.001, UB[col]-0.001)
                res = minimize(max_score_PI, x0, args= (X_known, y_known, X_grid, regressor, params, LB[0], UB[0]), method='L-BFGS-B',bounds = bnds)
                best_options[k] = res.x
                best_scores[k] = res.fun
            
            if j == 0:
                best_ind = np.argmin(best_scores)
                X_new = best_options[best_ind,:]
                X_new = np.reshape(X_new, (-1,dims))
            else:
                best_ind = np.argmin(best_scores)
                X_new_ = best_options[best_ind,:]
                X_new = np.append(X_new, X_new_)
                X_new = np.reshape(X_new, (-1,dims))
            
            X_known = np.append(X_known, X_new[j])
            X_known = np.reshape(X_known, (-1,dims))
            X_new_now = np.reshape(X_new[j], (1,-1))
            y_fake_new = regressor.predict(X_new_now)
            y_fake_new = np.reshape(y_fake_new, (-1,1))
            y_known = np.append(y_known, y_fake_new)
            y_known = np.reshape(y_known, (-1,1))
            regressor.fit(X_known, y_known)

        return X_new, dims
    
    def EI(self, regressor, X_grid, X_known, y_known, n_instances, params, LB, UB):
        dims = X_known.shape[1]
        tradeoff = params
        X_new = np.empty((0,dims))
        bnds = Bounds(LB[0], UB[0], keep_feasible=True)
        
        for j in range(0, n_instances):
            
            best_options = np.zeros((25,dims))
            best_scores = np.zeros((25,1))
            for k in range(0,25):
                x0 = np.zeros((dims))
                for col in range(dims):
                    x0[col] = np.random.uniform(LB[col]+0.001, UB[col]-0.001)
                res = minimize(max_score_EI, x0, args= (X_known, y_known, X_grid, regressor, params, LB[0], UB[0]), method='L-BFGS-B',bounds = bnds)
                best_options[k] = res.x
                best_scores[k] = res.fun
            
            if j == 0:
                best_ind = np.argmin(best_scores)
                X_new = best_options[best_ind,:]
                X_new = np.reshape(X_new, (-1,dims))
            else:
                best_ind = np.argmin(best_scores)
                X_new_ = best_options[best_ind,:]
                X_new = np.append(X_new, X_new_)
                X_new = np.reshape(X_new, (-1,dims))
            
            X_known = np.append(X_known, X_new[j])
            X_known = np.reshape(X_known, (-1,dims))
            X_new_now = np.reshape(X_new[j], (1,-1))
            y_fake_new = regressor.predict(X_new_now)
            y_fake_new = np.reshape(y_fake_new, (-1,1))
            y_known = np.append(y_known, y_fake_new)
            y_known = np.reshape(y_known, (-1,1))
            regressor.fit(X_known, y_known)

        return X_new, dims
    
    def UCB(self, regressor, X_grid, X_known, y_known, n_instances, params, LB, UB):
        dims = X_known.shape[1]
        tradeoff = params
        X_new = np.empty((0,dims))
        bnds = Bounds(LB[0], UB[0], keep_feasible=True)
        
        for j in range(0, n_instances):
            
            best_options = np.zeros((25,dims))
            best_scores = np.zeros((25,1))
            for k in range(0,25):
                x0 = np.zeros((dims))
                for col in range(dims):
                    x0[col] = np.random.uniform(LB[col]+0.001, UB[col]-0.001)
                res = minimize(max_score_UCB, x0, args= (X_known, y_known, X_grid, regressor, params, LB[0], UB[0]), method='L-BFGS-B',bounds = bnds)
                best_options[k] = res.x
                best_scores[k] = res.fun
            
            if j == 0:
                best_ind = np.argmin(best_scores)
                X_new = best_options[best_ind,:]
                X_new = np.reshape(X_new, (-1,dims))
            else:
                best_ind = np.argmin(best_scores)
                X_new_ = best_options[best_ind,:]
                X_new = np.append(X_new, X_new_)
                X_new = np.reshape(X_new, (-1,dims))
            
            X_known = np.append(X_known, X_new[j])
            X_known = np.reshape(X_known, (-1,dims))
            X_new_now = np.reshape(X_new[j], (1,-1))
            y_fake_new = regressor.predict(X_new_now)
            y_fake_new = np.reshape(y_fake_new, (-1,1))
            y_known = np.append(y_known, y_fake_new)
            y_known = np.reshape(y_known, (-1,1))
            regressor.fit(X_known, y_known)

        return X_new, dims


## Function for least-squares error regression
def fun(x, X_known, y_known, obj_name):
    obj = obj_fun()
    data = np.zeros((len(X_known), 1))
    x = np.append(x,ratio)
    ind = 0
    for X in X_known:
        data[ind] = obj.get_obj_fun(obj_name, X, x)
        ind = ind + 1
    f_diff = np.ravel(data) - np.ravel(y_known)
    return f_diff


## function that returns new physics-model parameters
def non_lin(fun, x0, X_known, y_known, obj_name):
    # fun, x0, X_known, y_known, obj_fun_name
    orig_params =get_obj_info(obj_name, 'params')
    variance = get_obj_info(obj_name, 'param var')
    poss_params = np.zeros((10, len(orig_params)))
    poss_x = np.zeros((10, len(orig_params)))
    cost = np.zeros((10,1))
    for k in range(len(orig_params)):
        poss_params[:,k] = np.random.normal(orig_params[k], variance[k],10)
    dims = X_known.shape[1]
    for j in range(10):
        lsq = least_squares(fun, x0, args = (X_known,y_known,obj_name))
        poss_x[j,:] = lsq.x
        cost[j] = lsq.cost
    lowest = np.argmin(cost)
    new_phys_tot = np.zeros((len(X_known),1))
    return new_phys_tot, poss_x[lowest]


## Select initial points and calculate objective values
def get_init_known(ob_name, init, dim, seed, params, ratio):

    choices = np.zeros((int(init),dim))
    y_choices = np.zeros((int(init),1))
    np.random.seed(seed)
    
    LB = get_obj_info(ob_name[0], 'LB')
    UB = get_obj_info(ob_name[0], 'UB')
    obj = obj_fun()
    for col in range(dim):
        choices[:,col] = np.random.uniform(LB[col], UB[col], (int(init)))
        
    for j in np.arange(int(init)):
        y_choices[j] = obj.get_obj_fun(ob_name, choices[j,:], params, ratio)

    X_known = choices
    y_known = y_choices
    
    return X_known, y_known


## Select physics-model data and add it to X_known and y_known
def get_used_data(X_known, y_known, ob_name, num, params):
    dims = X_known.shape[1]
    num_added = num - len(X_known)
    choices = np.zeros((num_added,dims))
    y_choices = np.zeros((num_added,1))
    LB = get_obj_info(ob_name, 'LB')
    UB = get_obj_info(ob_name, 'UB')
    np.random.seed(5)
    obj = obj_fun()
    for col in range(dims):
        choices[:,col] = np.random.uniform(LB[col], UB[col], (num_added))
        
    for j in np.arange(num_added):
        y_choices[j] = obj.get_obj_fun(ob_name, choices[j,:], params)
    
    X_used = np.append(X_known, choices)
    X_used = np.reshape(X_used, (-1,dims))
    
    y_used = np.append(y_known, y_choices)
    y_used = np.reshape(y_used, (-1,1))
    return X_used, y_used


## Initialize and run the optimizer. Outputs new experiments to run in X_new
def run_learner(X_grid, X_known, y_known, query_number, params, kern, acq, ob_name):
    # orig_data, X_tot, X_used, y_used, batch, a, b, d, new_eps, kern, acq
    
    LB = get_obj_info(ob_name[0], 'LB')
    UB = get_obj_info(ob_name[0], 'UB')
    diff = UB[0] - LB[0]
    length_scale = 0.3*diff
    alpha = 1e-3
    if kern == 'rbf':
        kernel = RBF(length_scale=length_scale, length_scale_bounds=(1e-3, diff)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1))
    elif kern == 'matern':
        kernel = Matern(length_scale=length_scale, length_scale_bounds=(1e-3, diff)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1))
    elif kern == 'rq':
        kernel = RationalQuadratic(length_scale=length_scale, length_scale_bounds=(1e-3, diff)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1))
        
    acq_obj = acquisition()
    
    learner = ActiveLearner(
        estimator=GaussianProcessRegressor(normalize_y=True, kernel=kernel, alpha=alpha, n_restarts_optimizer=100),
        query_strategy=acq_obj.get_acq_fun,
        X_training=X_known, y_training=np.ravel(y_known)) 
    dims = X_grid.shape[1]
    tot_preds, std = learner.predict(X_grid, return_std=True) 
    tot_preds = np.ravel(tot_preds)

    X_new, d = learner.query(acq, X_grid, X_known, y_known, query_number, params, LB, UB)
    X_new = np.reshape(X_new, (-1,dims))

    return X_new, tot_preds


## Testing on 20 different alternate models
def alg_test(alg_info, all_data, obj_fun_name, data_size, alt_params, ratio):
    # Set lists of initial point size and batch number size
    dims = all_data.shape[1]-1
    init_points_num = 10
    batch_size = 3
    tot_points = 50
    
    phys_dec = alg_info[0]
    acq = alg_info[1]
    kern = alg_info[2]
    
    date = alg_info[6]
    
    for alt in range(len(alt_params)):
        print(alt)
        orig_data = all_data[0:data_size, :]
        X_tot = orig_data[:, :dims]
        obj = obj_fun()
        y_grid = np.zeros((len(X_tot),1))
        for row in range(len(X_tot)):
            y_grid[row] = obj.get_obj_fun(obj_fun_name, X_tot[row,:], alt_params[alt,:], ratio)
        
        top_data = np.append(X_tot[np.argmax(y_grid),:], y_grid[np.argmax(y_grid)])
        min_data = np.append(X_tot[np.argmin(y_grid),:], y_grid[np.argmin(y_grid)])
        np.savetxt('Top_val_%s_%s_%s_%s_%s_%s.csv' % (phys_dec, acq, kern, 'comb', alt, date), top_data, delimiter=',')
        np.savetxt('Min_val_%s_%s_%s_%s_%s_%s.csv' % (phys_dec, acq, kern, 'comb', alt, date), min_data, delimiter=',')

        tot_batches = np.floor((tot_points - init_points_num)/batch_size).astype(int)
        rnd_seed = alt

        if acq == 'ours':
            params = np.linspace(1,0,tot_batches)
        elif acq == 'UCB':
            params = np.linspace(4,0,tot_batches)
        else:
            params = np.logspace(-1.3,-7,tot_batches)

        X_known, y_known = get_init_known(obj_fun_name, init_points_num, dims, rnd_seed, alt_params[alt,:], ratio)
        X_known = np.reshape(X_known, (-1,dims))
        y_known = np.reshape(y_known, (-1,1))                                                        

        test_tot_preds = np.zeros((tot_batches, data_size))
        x_vals = np.zeros((tot_batches, len(get_obj_info(obj_fun_name[0], 'params'))))

        phys_params = get_obj_info(obj_fun_name[0], 'params')
        for j in range(0, tot_batches):
            if phys_dec == 0:
                pass
            elif phys_dec == 1:
                # X_known, y_known, obj_fun_name, tot_points, phys_params
                X_used, y_used = get_used_data(X_known, y_known, obj_fun_name[0], tot_points, phys_params)
                X_used = np.reshape(X_used, (-1,dims))
                y_used = np.reshape(y_used, (-1,1))

            par = params[j]

            if phys_dec == 0:
                X_new, tot_preds = run_learner(X_tot, X_known, y_known, batch_size, par, kern, acq, obj_fun_name)
            elif phys_dec == 1:
                X_new, tot_preds = run_learner(X_tot, X_used, y_used, batch_size, par, kern, acq, obj_fun_name)                                        
            X_known = np.append(X_known, X_new)
            X_known = np.reshape(X_known, (-1,dims))
            obj = obj_fun()
            y_known_new = np.zeros((X_new.shape[0],1))
            for row in range(X_new.shape[0]):
                y_known_new[row] = obj.get_obj_fun(obj_fun_name, X_new[row,:], alt_params[alt,:], ratio)
            y_known = np.append(y_known, y_known_new)
            y_known = np.reshape(y_known, (-1,1))

            if phys_dec == 0:
                pass
            elif phys_dec == 1:
                x0 = phys_params
                temp, phys_params = non_lin(fun, x0, X_known, y_known, obj_fun_name[0]) 
                x_vals[j,:] = phys_params

                if j == tot_batches -1:
                    np.savetxt('Obj_params_%s_%s_%s_%s_%s_%s.csv' % (phys_dec, acq, kern, 'comb', alt, date), x_vals, delimiter=',')

            if j == tot_batches - 1:
                np.savetxt('Known_vals_%s_%s_%s_%s_%s_%s.csv' % (phys_dec, acq, kern, 'comb', alt, date), np.concatenate((X_known,y_known),axis=1), delimiter=',')
            

## Calculation of the average and standard deviation of the difference between the optimal location and the most optimal experimental location so far
def dist_calc(phys_dec, acq, kern, ob_name, date):
    num_exp = 50
    ob_name = 'comb'
    combined = np.zeros((20, num_exp))
    
    for j in range(20):
        known = pd.read_csv('Known_vals_%s_%s_%s_%s_%s_%s.csv' % (phys_dec, acq, kern, ob_name, j, date), sep=',', header=None)
        known = known.values
        top = pd.read_csv('Top_val_%s_%s_%s_%s_%s_%s.csv' % (phys_dec, acq, kern, ob_name, j, date), sep=',', header=None)
        top = top.values
        top_loc = top[:-1]
        dims = top_loc.shape[0]
        top_loc = np.reshape(top_loc, (-1,dims))

        ind = 0
        for val in known[:,:-1]:
            dist_temp = np.linalg.norm(val-top_loc)
            if ind == 0:
                combined[j, ind] = dist_temp
            else:
                if dist_temp < combined[j, ind-1]:
                    combined[j,ind] = dist_temp
                else:
                    combined[j,ind] = combined[j, ind-1]
            ind = ind + 1
    
    combined_avg = np.mean(combined, axis=0)
    combined_std = np.std(combined, axis=0)
    return combined_avg, combined_std


## Calculation of the average and standard deviation of the difference between the optimal value and the most optimal value so far
def error_calc(phys_dec, acq, kern, ob_name, date):
    num_exp = 50
    
    ob_name = 'comb'
    combined = np.zeros((20, num_exp))
    
    for j in range(20):
        known = pd.read_csv('Known_vals_%s_%s_%s_%s_%s_%s.csv' % (phys_dec, acq, kern, ob_name, j, date), sep=',', header=None)
        known = known.values
        top = pd.read_csv('Top_val_%s_%s_%s_%s_%s_%s.csv' % (phys_dec, acq, kern, ob_name, j, date), sep=',', header=None)
        top = top.values
        top_val = top[-1][0]
        min_data = pd.read_csv('Min_val_%s_%s_%s_%s_%s_%s.csv' % (phys_dec, acq, kern, ob_name, j, date), sep=',', header=None)
        min_data = min_data.values
        min_val = min_data[-1][0]
        dims = known.shape[1]-1
        
        top_val_maybe = np.amax(known[:,dims])
        if top_val_maybe > top_val:
            top_val = top_val_maybe
        
        ind = 0
        for val in known[:,dims]:
            
            err_temp = (top_val - val)/(top_val-min_val)
            if ind == 0:
                combined[j, ind] = err_temp
            else:
                if err_temp < combined[j, ind-1]:
                    combined[j,ind] = err_temp
                else:
                    combined[j,ind] = combined[j, ind-1]
            ind = ind + 1
    
    combined_avg = np.mean(combined, axis=0)
    combined_std = np.std(combined, axis=0)
    return combined_avg, combined_std


## Used to return information about the objective function from the file 'obj_test.csv'
def get_obj_info(obj_name, info_name):
    obj_fun_info = pd.read_csv('obj_test.csv', sep=',', header=0)
    obj_fun_info = obj_fun_info.values
    
    obj_row = obj_fun_info[obj_fun_info[:,0] == obj_name,:]
    
    if info_name == 'dims':
        return int(obj_row[0,1])
    elif info_name == 'params':
        params = obj_row[0,2].split()
        ind = 0
        for par in params:
            params[ind] = float(par)
            ind += 1
        return params
    elif info_name == 'param var':
        variance = obj_row[0,3].split()
        ind = 0
        for var in variance:
            variance[ind] = float(var)
            ind += 1
        return variance
    elif info_name == 'num params':
        return int(obj_row[0,4])
    elif info_name == 'LB':
        LB = obj_row[0,5].split()
        ind=0
        for L in LB:
            LB[ind] = float(L)
            ind += 1
        return LB
    elif info_name == 'UB':
        UB = obj_row[0,6].split()
        ind=0
        for U in UB:
            UB[ind] = float(U)
            ind += 1
        return UB


## Generates a grid of points between the bound
def create_grid(LB, UB, num, dim, num_diff):
    if num_diff == 1:
        if dim == 2:
            x = np.linspace(LB[0], UB[0], num)
            y = np.linspace(LB[0], UB[0], num)
        elif dim == 3:
            x = np.linspace(LB[0], UB[0], num)
            y = np.linspace(LB[0], UB[0], num)
            w = np.linspace(LB[0], UB[0], num)
        else:
            x = np.linspace(LB[0], UB[0], num)
            y = np.linspace(LB[0], UB[0], num)
            w = np.linspace(LB[0], UB[0], num)
            v = np.linspace(LB[0], UB[0], num)
    elif num_diff == 2:
        if dim == 2:
            x = np.linspace(LB[0], UB[0], num)
            y = np.linspace(LB[1], UB[1], num)
        elif dim == 3:
            x = np.linspace(LB[0], UB[0], num)
            y = np.linspace(LB[0], UB[0], num)
            w = np.linspace(LB[1], UB[1], num)
        else:
            x = np.linspace(LB[0], UB[0], num)
            y = np.linspace(LB[0], UB[0], num)
            w = np.linspace(LB[1], UB[1], num)
            v = np.linspace(LB[1], UB[1], num)
    else:
        if dim == 3:
            x = np.linspace(LB[0], UB[0], num)
            y = np.linspace(LB[1], UB[1], num)
            w = np.linspace(LB[2], UB[2], num)
        else:
            x = np.linspace(LB[0], UB[0], num)
            y = np.linspace(LB[0], UB[0], num)
            w = np.linspace(LB[1], UB[1], num)
            v = np.linspace(LB[2], UB[2], num)
    if dim == 2:
        z = np.meshgrid(x, y, indexing='ij')
        z_ = np.zeros((num**dim,dim))
        for row in range(num**dim):
            z_[row] = [np.ravel(z[0])[row], np.ravel(z[1])[row]]
    elif dim == 3:
        z = np.meshgrid(x, y, w, indexing='ij')
        z_ = np.zeros((num**dim,dim))
        for row in range(num**dim):
            z_[row] = [np.ravel(z[0])[row], np.ravel(z[1])[row], np.ravel(z[2])[row]]
    elif dim == 4:
        z = np.meshgrid(x, y, w, v, indexing='ij')
        z_ = np.zeros((num**dim,dim))
        for row in range(num**dim):
            z_[row] = [np.ravel(z[0])[row], np.ravel(z[1])[row], np.ravel(z[2])[row], np.ravel(z[3])[row]]
    elif dim == 6:
        z = np.meshgrid(x, x, x, x, x, x, indexing='ij')
        z_ = np.zeros((num**dim,dim))
        for row in range(num**dim):
            z_[row] = [np.ravel(z[0])[row], np.ravel(z[1])[row], np.ravel(z[2])[row], np.ravel(z[3])[row], np.ravel(z[4])[row], np.ravel(z[5])[row]]

    return z_


## Running of file begins here
t1_start = perf_counter()
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
# read in array of information about different algorithms
alg_info = pd.read_csv('Alg_test.csv', sep=',', header=0)
alg_info = alg_info.values
date = alg_info[0,6]
# Objective function info: [name, dims, params, param var, num params, LB, UB] - 8 items
# Options = 'rosen', 'rosen_3', 'branin', 'camel_3', 'sphere', 'sphere_3', 'sphere_4', 'mccor', 'hart_3', 'hart_4', 'hart_6'
ob_name_1 = alg_info[0,3]
ob_name_2 = alg_info[0,4]
choices = [ob_name_1, ob_name_2]
ratio = alg_info[0,7]
# Generate data with physics model
ob_fun = obj_fun()
LB = get_obj_info(ob_name_1, 'LB')
UB = get_obj_info(ob_name_1, 'UB')
num = alg_info[0,5]
dim = get_obj_info(ob_name_1, 'dims')
total_size = num**dim

params_1 = get_obj_info(ob_name_1, 'params')
params_2 = get_obj_info(ob_name_2, 'params')
params = np.concatenate((params_1, params_2), axis=0)


variance_1 = get_obj_info(ob_name_1, 'param var')
variance_2 = get_obj_info(ob_name_2, 'param var')
variance = np.concatenate((variance_1, variance_2), axis=0)


# Original data
X = create_grid(LB, UB, num, dim, 1)
Z = np.zeros((len(X), 1))
ind =0
for x in X:
    Z[ind] = ob_fun.get_obj_fun(ob_name_1, x, params_1)
    ind = ind + 1


# Generate alternate models
num_alts = 20
alt_params = np.zeros((num_alts, len(params)))
for j in range(len(params)):
    alt_params[:,j] = np.random.normal(params[j], variance[j], num_alts)

np.savetxt('alt_params_%s_%s_%s.csv' % (choices[0], choices[1], date), alt_params, delimiter=',')

all_data = np.zeros((total_size*(num_alts+1), dim+1))
all_data[:total_size,:] = np.concatenate((X,Z), axis=1)

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

so_far = 0
## go through each algorithm and save error
for row in alg_info:
    phys_dec = row[0]
    acq_fun = row[1]
    kern = row[2]
    print(so_far)
    so_far += 1

    alg_test(row, all_data, choices, total_size, alt_params, ratio)

    mean_error, std_error = error_calc(phys_dec, acq_fun, kern, choices, date)
    mean_dist, std_dist = dist_calc(phys_dec, acq_fun, kern, choices, date)
    np.savetxt('Error_vals_%s_%s_%s_%s_%s.csv' % (phys_dec, acq_fun, kern, 'comb', date), mean_error, delimiter=',')
    np.savetxt('Error_std_%s_%s_%s_%s_%s.csv' % (phys_dec, acq_fun, kern, 'comb', date), std_error, delimiter=',')
    np.savetxt('Dist_vals_%s_%s_%s_%s_%s.csv' % (phys_dec, acq_fun, kern, 'comb', date), mean_dist, delimiter=',')
    np.savetxt('Dist_std_%s_%s_%s_%s_%s.csv' % (phys_dec, acq_fun, kern, 'comb', date), std_dist, delimiter=',')
    

t1_stop = perf_counter()  
print(t1_stop-t1_start)


