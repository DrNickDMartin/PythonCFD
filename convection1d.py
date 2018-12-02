import numpy as np
from matplotlib import pyplot as p

def conv1d(nx,t,l,c=0):

    # Setup grid
    x = np.linspace(0, l, nx)
    dx = x[1]-x[0]

    # Initial conditions
    u = np.ones(nx)
    u[int(0.5/dx):int(1/dx + 1)] = 2

    # Loop through time and space
    ti = 0
    while (ti <= t):
        un = u.copy()
        dt = np.min((dx/un)*0.5)
        ti += dt
        for i in range(1, nx):
            if c>0:
                u[i] = un[i] - c*(dt/dx)*(un[i]-un[i-1])
            else:
                u[i] = un[i] - un[i]*(dt/dx)*(un[i]-un[i-1])
        
    return (x,u)


if __name__ == "__main__":

    (x1,u1) = conv1d(25,0.5,2,1)
    (x2,u2) = conv1d(50,0.5,2,1)
    (x3,u3) = conv1d(500,0.5,2,1)

    p.figure()
    p.plot(x1,u1)
    p.plot(x2,u2)
    p.plot(x3,u3)
    p.grid()
    p.title("Linear Convection")
    p.legend(["25","50","500"])
    p.show()

    (x1,u1) = conv1d(25,0.5,2)
    (x2,u2) = conv1d(50,0.5,2)
    (x3,u3) = conv1d(500,0.5,2)

    p.figure()
    p.plot(x1,u1)
    p.plot(x2,u2)
    p.plot(x3,u3)
    p.grid()
    p.title("Non-linear Convection")
    p.legend(["25","50","500"])
    p.show()
        

