import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def HG_theta(g, theta):
    ng = len(g)
    ntheta = len(theta)
    p = np.zeros((ng, ntheta))
    for i in range(ng):
        p[i,:] = 0.5*(1-g[i]**2.0)/((1+g[i]**2.0-2*g[i]*np.cos(theta))**1.5) # 2*pi*p(theta)
        p[i,:] *= np.sin(theta)

    return p

def HG_u(g, u):
    ng = len(g)
    nu = len(u)
    p = np.zeros((ng, nu))
    for i in range(ng):
        p[i,:] = 0.5*(1-g[i]**2.0)/((1+g[i]**2.0-2*g[i]*u)**1.5) 

    return p

# ========================================================

if __name__=='__main__':
    g_train = {'g':[0.65, 0.75, 0.85, 0.95], 'Events':['Train', 'Train', 'Train', 'Train']}
    g_val   = {'g':[0.6, 0.7, 0.8, 0.9], 'Events':['Val', 'Val', 'Val', 'Val']}

    theta = np.arange(0, np.pi, 0.01)

    g = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
    pHG = HG_theta(g, theta)

    fig, ax = plt.subplots(figsize=(4,3), dpi=300)
    for i in range(0,8):
        sns.lineplot(ax=ax, x=theta, y=pHG[i,:], label=f'g={g[i]:>0.2f}')
    plt.xlabel(''r'$\theta$ [radian]')
    plt.ylabel(''r'$p\'(\theta)$')
    plt.legend()
    figFile = os.path.join('testing_results', f'Figure_HG.png')
    plt.savefig(figFile, bbox_inches='tight')
    plt.show()

    u = np.arange(-1, 1, 0.01)
    pHG_u = HG_u(g, u)

    fig, ax = plt.subplots(figsize=(4,3), dpi=300)
    for i in range(pHG_u.shape[0]):
        sns.lineplot(ax=ax, x=u, y=pHG_u[i,:], label=f'g={g[i]:>0.2f}')
    ax.set_xlabel(''r'$\mu$')
    ax.set_ylabel(''r'$p(\mu)$')
    ax.legend()
    figFile = os.path.join('testing_results', f'Figure_HG_mu.png')
    plt.savefig(figFile, bbox_inches='tight')
    plt.show()


    print('Done')