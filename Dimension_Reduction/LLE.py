"""
    LLE ALGORITHM (using k nearest neighbors)
    X: data as D x N matrix (D = dimensionality, N = #points)
    k: number of neighbors
    dmax = max embedding dimensionality
    Y: lle(X,k,dmax) -> embedding as dmax x N matrix
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs


def LLE(X, k, d, sparse=True):
    D, N = X.shape
    print('LLE running on {} points in {} dimensions\n'.format(N, D))

    # Step1: compute pairwise distances & find neighbors
    print('-->Finding {} nearest neighbours.\n'.format(k))

    X2 = np.sum(X ** 2, axis=0).reshape(1, -1)  # 1xN
    distance = np.tile(X2, (N, 1)) + np.tile(X2.T, (1, N)) - 2 * np.dot(X.T, X)  # NxN

    index = np.argsort(distance, axis=0)
    neighborhood = index[1:1 + k, :]  # kxN filter itself

    # Step2: solve for reconstruction weights
    print('-->Solving for reconstruction weights.\n')

    if k > D:
        print(' [note: k>D; regularization will be used]\n')
        tol = 1e-3  # regularlizer in case constrained fits are ill conditioned
    else:
        tol = 0

    w = np.zeros((k, N))
    for ii in range(N):
        xn = X[:, neighborhood[:, ii]] - np.tile(X[:, ii].reshape(-1, 1), (1, k))  # shift ith pt to origin
        S = np.dot(xn.T, xn)  # local covariance,xn = Xi-Ni
        S = S + np.eye(k, k) * tol * np.trace(S)  # regularlization (k>D)
        Sinv = np.linalg.inv(S)  # inv(S)
        w[:, ii] = np.dot(Sinv, np.ones((k, 1))).reshape(-1, )  # solve Cw=1
        w[:, ii] = w[:, ii] / sum(w[:, ii])  # enforce sum(wi)=1

    # Step 3: compute embedding from eigenvectors of cost matrix M = (I-W)'(I-W)
    print('-->Computing embedding to get eigenvectors .\n')

    if sparse:  # parse solution
        M = lil_matrix(np.eye(N, N))  # use a sparse matrix lil_matrix((N,N))
        for ii in range(N):
            wi = w[:, ii].reshape(-1, 1)  # kx1, i point neighborhood (wji)
            jj = neighborhood[:, ii].tolist()  # k,
            M[ii, jj] = M[ii, jj] - wi.T
            M[jj, ii] = M[jj, ii] - wi
            M_temp = M[jj, :][:, jj].toarray() + np.dot(wi, wi.T)
            for ir, row in enumerate(jj):  ### TO DO
                for ic, col in enumerate(jj):
                    M[row, col] = M_temp[ir, ic]
    else:  # dense solution
        M = np.eye(N, N)  # use a dense eye matrix
        for ii in range(N):
            wi = w[:, ii].reshape(-1, 1)  # kx1
            jj = neighborhood[:, ii].tolist()  # k,
            M[ii, jj] = M[ii, jj] - wi.reshape(-1, )
            M[jj, ii] = M[jj, ii] - wi.reshape(-1, )
            M_temp = M[jj, :][:, jj] + np.dot(wi, wi.T)  # kxk
            for ir, row in enumerate(jj):  ### TO DO
                for ic, col in enumerate(jj):
                    M[row, col] = M_temp[ir, ic]
        M = lil_matrix(M)
        # Calculation of embedding
    # note: eigenvalue(M) >=0
    eigenvals, Y = eigs(M, k=d + 1, sigma=0)  # Y-> Nx(d+1)
    Y = np.real(Y)[:, :d + 1]  # get d+1 eigenvalue -> eigenvectors
    Y = Y[:, 1:d + 1].T * np.sqrt(N)  # bottom evect is [1,1,1,1...] with eval 0
    print('Done.\n')

    # other possible regularizers for k>D
    #   S = S + tol*np.diag(np.diag(S))           # regularlization
    #   S = S + np.eye(k,k)*tol*np.trace(S)*k     # regularlization
    return Y


import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# Swill Roll
def plot_swill_roll(N=2000, k=12, d=2, sparse=True):
    # Plot true manifold
    tt0 = (3 * math.pi / 2) * (1 + 2 * np.linspace(0, 1, 51)).reshape(1, -1)
    hh = 30 * np.linspace(0, 1, 9).reshape(1, -1)
    xx = np.dot((tt0 * np.cos(tt0)).T, np.ones(np.shape(hh)))
    yy = np.dot(np.ones(np.shape(tt0)).T, hh)
    zz = np.dot((tt0 * np.sin(tt0)).T, np.ones(np.shape(hh)))
    cc = np.dot(tt0.T, np.ones(np.shape(hh)))

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(131, projection='3d')
    # ax = Axes3D(fig) # change to 3D figure
    cc = cc / cc.max()  # normalize 0 to 1
    # ax.plot_surface(xx, yy, zz, rstride = 1, cstride = 1, cmap = cm.coolwarm,facecolors=cm.rainbow(cc))
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=cm.rainbow(cc))
    ax.grid(True)

    lnx = -5 * np.array([[3, 3, 3], [3, -4, 3]]).T
    lny = np.array([[0, 0, 0], [32, 0, 0]]).T
    lnz = -5 * np.array([[3, 3, 3], [3, 3, -3]]).T
    ax.plot3D(lnx[0], lny[0], lnz[0], color='red', linewidth='2', linestyle='-')
    ax.plot3D(lnx[1], lny[1], lnz[1], color='red', linewidth='2', linestyle='-')
    ax.plot3D(lnx[2], lny[2], lnz[2], color=[1, 0, 0], linewidth='2', linestyle='-')
    ax.set_xlim(-15, 20)
    ax.set_ylim(0, 32)
    ax.set_zlim(-15, 15)

    # Generate sampled data
    tt = (3 * math.pi / 2) * (1 + 2 * np.random.rand(1, N))
    tts = tt.reshape(-1, )
    height = 21 * np.random.rand(1, N)
    X = np.concatenate([tt * np.cos(tt), height, tt * np.sin(tt)], axis=0)

    # Scatterplot of sampled data
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(X[0, :], X[1, :], X[2, :], marker='+', s=12, c=tts)
    ax.plot3D(lnx[0], lny[0], lnz[0], color='red', linewidth='2', linestyle='-')
    ax.plot3D(lnx[1], lny[1], lnz[1], color='red', linewidth='2', linestyle='-')
    ax.plot3D(lnx[2], lny[2], lnz[2], color=[1, 0, 0], linewidth='2', linestyle='-')
    # plt.axis([-15,20,0,32,-15,15])
    ax.set_xlim(-15, 20)
    ax.set_ylim(0, 32)
    ax.set_zlim(-15, 15)

    # Run LLE  algorithm
    Y = LLE(X, k, d, sparse=sparse)
    # Scatterplot of embedding
    ax = fig.add_subplot(133)
    ax.scatter(Y[0, :], Y[1, :], marker='+', s=12, c=tts)
    ax.grid(True)
    plt.show()


# S-Curve
def plot_s_curve(N=2000, k=12, d=2, sparse=False):
    # Plot true manifold
    tt = math.pi * np.linspace(-1, 0.5, 16)
    uu = tt[::-1].reshape(1, -1)
    tt = tt.reshape(1, -1)
    hh = 5 * np.linspace(0, 1, 11).reshape(1, -1)
    xx = np.dot(np.concatenate([np.cos(tt), -np.cos(uu)], axis=1).T, np.ones(np.shape(hh)))
    yy = np.dot(np.ones(np.shape(np.concatenate([tt, uu], axis=1))).T, hh)
    zz = np.dot(np.concatenate([np.sin(tt), 2 - np.sin(uu)], axis=1).T, np.ones(np.shape(hh)))
    cc = np.dot(np.concatenate([tt, uu], axis=1).T, np.ones(np.shape(hh)))

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(131, projection='3d')
    # ax = Axes3D(fig) # change to 3D figure
    # cc = cc / cc.max()  # normalize 0 to 1
    # ax.plot_surface(xx, yy, zz, rstride = 1, cstride = 1, cmap = cm.coolwarm)
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=cm.jet(cc))
    ax.grid(True)

    lnx = -1 * np.array([[1, 1, 1], [1, -1, 1]]).T
    lny = np.array([[0, 0, 0], [5, 0, 0]]).T
    lnz = -1 * np.array([[1, 1, 1], [1, 1, -3]]).T
    ax.plot3D(lnx[0], lny[0], lnz[0], color='red', linewidth='2', linestyle='-')
    ax.plot3D(lnx[1], lny[1], lnz[1], color='red', linewidth='2', linestyle='-')
    ax.plot3D(lnx[2], lny[2], lnz[2], color=[1, 0, 0], linewidth='2', linestyle='-')
    # plt.axis([-1,1,0,5,-1,3])
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 5)
    ax.set_zlim(-1, 3)

    # Generate sampled data
    angle = math.pi * (1.5 * np.random.rand(1, int(N / 2)) - 1)
    angle2 = np.concatenate([angle, angle], axis=1).reshape(-1, )
    angle2 = angle2 / angle2.max()
    height = 5 * np.random.rand(1, N)
    X = np.concatenate([np.concatenate([np.cos(angle), -np.cos(angle)], axis=1),
                        height, np.concatenate([np.sin(angle), 2 - np.sin(angle)], axis=1)])

    # Scatterplot of sampled data
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(X[0, :], X[1, :], X[2, :], marker='+', s=12, c=angle2)  #
    ax.plot3D(lnx[0], lny[0], lnz[0], color='red', linewidth='2', linestyle='-')
    ax.plot3D(lnx[1], lny[1], lnz[1], color='red', linewidth='2', linestyle='-')
    ax.plot3D(lnx[2], lny[2], lnz[2], color=[1, 0, 0], linewidth='2', linestyle='-')
    # plt.axis([-1,1,0,5,-1,3])
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 5)
    ax.set_zlim(-1, 3)

    # Run LLE  algorithm
    Y = LLE(X, k, d, sparse=sparse)
    # Scatterplot of embedding
    ax = fig.add_subplot(133)
    ax.scatter(Y[0, :], Y[1, :], marker='+', s=12, c=angle2)
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_swill_roll(N=2000, k=12, d=2)
    plot_s_curve(N=2000, k=12, d=2)
