import numpy as np
from matplotlib import pyplot as p

def setupGrid(nx,l):

    # Setup grid
    x = np.linspace(-l/2, l/2, nx)

    # Initial conditions
    u = np.ones(nx)
    u[int(nx*0.333):int(nx*0.6666)] = 2

    return (x,u)

def diff1d(x,u,nu,t):

    nx = len(x)
    dx = x[1]-x[0]

    ti = 0
    while(ti < t):
        un = u.copy()
        dt = np.min((0.2*dx**2)/nu)
        ti += dt
        for i in range(1,nx-1):
            u[i] = un[i] + (nu*dt/dx**2)*(un[i+1]-2*un[i]+un[i-1])

    return u


if __name__ == "__main__":
    
    (x, u0) = setupGrid(80, 3)
    u = diff1d(x, u0.copy(), 0.3, 0.1)

    p.plot(x, u0)
    p.plot(x, u)
    p.show()