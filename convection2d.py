import numpy as np
from matplotlib import pyplot as p
from matplotlib import cm


def setupGrid(lx, ly, nx, ny):
    x = np.linspace(0,lx,nx)
    y = np.linspace(0,ly,ny)

    xmat, ymat = np.meshgrid(x,y,indexing='ij')

    xind = np.logical_and(xmat>0.8,xmat<1.2)
    yind = np.logical_and(ymat>1,ymat<1.3)
    ind = np.logical_and(xind,yind)

    u = np.ones((nx,ny))
    v = np.ones((nx,ny))
    u[ind] = 2
    v[ind] = 2

    return (x, y, xmat, ymat, u, v)

def conv2d(x, y, u, v, t, c, cfl):

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    ti = 0
    dt = np.min([np.min((dx/u)*cfl),np.min((dy/v)*cfl)])
    while (ti <= t):
        un = u.copy()
        vn = v.copy()
        ti += dt
        du_dx = (dt/dx)*(un[1:,1:] - un[:-1,1:])
        du_dy = (dt/dy)*(un[1:,1:] - un[1:,:-1])
        dv_dx = (dt/dx)*(vn[1:,1:] - vn[:-1,1:])
        dv_dy = (dt/dy)*(vn[1:,1:] - vn[1:,:-1])
        u[1:,1:] = un[1:,1:] - un[1:,1:]*du_dx - vn[1:,1:]*du_dy
        v[1:,1:] = vn[1:,1:] - un[1:,1:]*dv_dx - vn[1:,1:]*dv_dy
        u[[0,-1], :] = 1
        u[:, [0,-1]] = 1
        v[:, [0,-1]] = 1
        v[[0,-1], :] = 1
        
    return (u, v)

def main():

    (x, y, xmat, ymat, u, v) = setupGrid(2, 2.5, 500, 500)

    p.figure()
    p.contourf(xmat, ymat,  np.sqrt(u**2 + v**2), cmap='jet')
    p.streamplot(x, y, u, v)
    p.xlabel('x')
    p.ylabel('y')
    p.show(block=False)

    (u, v) = conv2d(x, y, u, v, 0.5, 1, 0.5)

    p.figure()
    p.contourf(xmat, ymat, np.sqrt(u**2 + v**2), cmap='jet')
    p.streamplot(x, y, u, v)
    p.show(block=False)
    p.xlabel('x')
    p.ylabel('y')
    p.show()


if __name__ == "__main__":
    main()

