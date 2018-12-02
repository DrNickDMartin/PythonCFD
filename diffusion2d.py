import numpy as np
from matplotlib import pyplot as p
from matplotlib import cm


def setupGrid(lx, ly, nx, ny):
    x = np.linspace(0,lx,nx)
    y = np.linspace(0,ly,ny)

    xmat, ymat = np.meshgrid(x,y,indexing='ij')

    xind = np.logical_and(xmat>0.8,xmat<1.2)
    yind = np.logical_and(ymat>0.8,ymat<1.2)
    ind = np.logical_and(xind,yind)

    u = np.ones((nx,ny))
    v = np.ones((nx,ny))*-0.25
    u[ind] = 2
    v[ind] = -0.5

    return (x, y, xmat, ymat, u, v)

def diff2d(x, y, u, v, t, nu, cfl):

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    ti = 0
    dt = (cfl * dx * dy) / nu
    print(dt)
    while (ti < t):
        un = u.copy()
        vn = v.copy()
        ti += dt
        d2u_dx2 = (dt/dx**2)*(un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1])
        d2u_dy2 = (dt/dy**2)*(un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])
        d2v_dx2 = (dt/dx**2)*(vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1])
        d2v_dy2 = (dt/dy**2)*(vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2])
        u[1:-1,1:-1] = un[1:-1,1:-1] + nu*(d2u_dx2 + d2u_dy2)
        v[1:-1,1:-1] = vn[1:-1,1:-1] + nu*(d2v_dx2 + d2v_dy2)
        u[:, [0,-1]] = 1
        u[[0,-1], :] = 1
        v[:, [0,-1]] = 1
        v[[0,-1], :] = 1
        
    return (u, v)

def main():

    (x, y, xmat, ymat, u, v) = setupGrid(2, 2, 31, 31)
    (u, v) = diff2d(x, y, u, v, 1, 0.05, 0.2)

    p.figure()
    p.contourf(xmat, ymat, np.sqrt(u**2+v**2), cmap='jet')
    p.colorbar()
    p.streamplot(x, y, u, v)
    p.show(block=False)
    p.xlabel('x')
    p.ylabel('y')
    p.show()


if __name__ == "__main__":
    main()

