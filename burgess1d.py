import numpy as np
import sympy as sp
from matplotlib import pyplot as p
from sympy import init_printing
from sympy.utilities.lambdify import lambdify

init_printing(use_latex=True)

def sawtoothSolution():
    x, nu, t = sp.symbols('x nu t')
    phi1 = sp.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1)))
    phi2 = sp.exp(-(x - 4 * t - 2 * sp.pi)**2 / (4 * nu * (t + 1)))
    phi = phi1 + phi2

    dphi_dx = phi.diff(x)
    u = -2 * nu * (dphi_dx/phi) + 4

    return lambdify((t, x, nu), u)

def setupGrid(nx,nu):
    x = np.linspace(0, 2*np.pi, nx)
    ufunc = sawtoothSolution()
    u = np.asarray([ufunc(0, xi, nu) for xi in x])
    return (x, u)
    

def burgess1d(x,u,t,nu,cfl):
    dx = x[1]-x[0]
    ti = 0
    dt = cfl*dx
    while(ti < t):
        un = u.copy()
        ti += dt
        u[1:-1] = un[1:-1] - un[1:-1]*(dt/dx)*(un[1:-1]-un[:-2]) + (nu*dt/dx**2)*(un[2:] - 2*un[1:-1] + un[:-2])
        u[0] = un[0] - un[0]*(dt/dx)*(un[0]-un[-2]) + (nu*dt/dx**2)*(un[1]-2*un[0]+un[-2])
        u[-1] = u[0]
    return u


if __name__ == "__main__":

    nu = 0.07
    nx = 1000
    t = 0.75

    (x, u) = setupGrid(nx, nu)
    u = burgess1d(x,u,t,nu,0.01)

    p.figure(figsize=(11, 7), dpi=100)
    p.plot(x, u, marker='o', lw=2)
    p.plot(x, np.asarray([(sawtoothSolution())(t, xi, nu) for xi in x]))
    p.xlim([0, np.max(x)])
    p.ylim([0, 10])
    p.grid()
    p.xlabel('x')
    p.ylabel('u')
    p.legend(['numerical','analytical'])
    p.show()


