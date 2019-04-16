#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:56:43 2019

@author: charlotteperlant
"""

import numpy as np
from numpy.linalg import lstsq
from numpy.linalg import qr

#Question 1 : extraction et chargement des données.
data = np.loadtxt("data.txt")
print(data)

#Question 2 : Extraction des matrices de points 3D et de points 2D
pts_3D_ini = data[:,0:3]
pts_2D_ini = data[:,3:5]

length = pts_3D_ini.shape[0]

B = np.ones((length, 1))
print(B)

#Ajout aux points 3D et 2D d'une dernière coordonnée valant 1
pts_3D = np.c_[pts_3D_ini, B]
pts_2D = np.c_[pts_2D_ini, B]

print(pts_3D)
print(pts_2D)

#Question 3 : Tracé des points 3D de la mire de calibration

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(pts_3D[:,0], pts_3D[:,1], pts_3D[:,2])
plt.show()

#question 4 : écriture de la matrice A
def matrix_A(X, x):
    x1, x2, x3 = x[0], x[1], x[2]
    X1, X2, X3, X4  = X[0], X[1], X[2], X[3]
    A = np.zeros((2, 12))
    A[0] = [0, -x3*X1, x2*X1, 0, -x3*X2, x2*X2, 0, -x3*X3, x2*X3, 0, -x3*X4, x2*X4]
    A[1] = [x3*X1, 0, -x1*X1, x3*X2, 0, -x1*X2, x3*X3, 0, -x1*X3, x3*X4, 0, -x1*X4]
    return A 

A = np.zeros((0, 12))
for k in range(length):
    A = np.concatenate((A, matrix_A(pts_3D[k, :], pts_2D[k, :])), axis=0)
  

#question 5 : résolution du système
A_Final = A[:, 0:11]
print(A_Final)
b = -A[:, 11]
print(b)

Pint = np.linalg.lstsq(A_Final, b, rcond = None)[0]
Pint = np.append(Pint, 1)
print(Pint)

#question 6 : calcul de la matrice K


def matrix_P(S):
    Q = np.zeros((3, 4))
    Q[0] = [S[0], S[3], S[6], S[9]]
    Q[1] = [S[1], S[4], S[7], S[10]]
    Q[2] = [S[2], S[5], S[8], S[11]]
    return Q

P = matrix_P(Pint)
print(P)

def matrix_RQ(M):
    n, m = np.shape(M)
    J = np.fliplr(np.eye(n))
    Q, R = np.linalg.qr(np.dot(J, np.dot(M.T, J)))
    Rfin = np.dot(J, np.dot(R.T, J))
    Qfin = np.dot(J, np.dot(Q.T, J))
    return Rfin, Qfin 
   
Pint2 = P[:, 0:3]
R, Q = matrix_RQ(Pint2)
#Normalisation de la matrice K de telle sorte que le coefficient K[3][3] soit égal à 1.
R = R/R[2][2]
print(R)
print(Q)
#Ici la matrice R est exactement la matrice K recherchée

#question 7 : projection des points 3D sur le plan de la camera 2D

pts_2D_appro = np.zeros((length, 3))
for k in range(length):
    pts_2D_appro[k] = np.dot(P, pts_3D[k])/np.dot(P, pts_3D[k])[2]
    
plt.scatter(pts_2D[:, 0], pts_2D[:, 1], color='r')
plt.scatter(pts_2D_appro[:, 0], pts_2D_appro[:, 1], color='g')
plt.show()
    
    
        
    

