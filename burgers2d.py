import numpy as np
from matplotlib import pyplot as p
from matplotlib import cm
import matplotlib.animation as animation


def setupGrid(lx, ly, nx, ny):
    x = np.linspace(0,lx,nx)
    y = np.linspace(0,ly,ny)

    xmat, ymat = np.meshgrid(x,y,indexing='ij')

    xind = np.logical_and(xmat>0.5,xmat<0.8)
    yind = np.logical_and(ymat>0.5,ymat<0.8)
    ind = np.logical_and(xind,yind)

    u = np.ones((nx,ny))
    v = np.ones((nx,ny))
    u[ind] = 2
    v[ind] = 2

    return (x, y, xmat, ymat, u, v)

def burg2d(x, y, u, v, t, nu, cfl):

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    ti = 0
    dt = (cfl*dx*dy)/nu

    ts = []
    vels = []
    ts.append(ti)
    vels.append((u,v))

    while (ti < t):
        un = u.copy()
        vn = v.copy()
        ti += dt

        # Convection terms
        du_dx = (dt/dx)*(un[1:-1,1:-1] - un[:-2,1:-1])
        du_dy = (dt/dy)*(un[1:-1,1:-1] - un[1:-1,:-2])
        dv_dx = (dt/dx)*(vn[1:-1,1:-1] - vn[:-2,1:-1])
        dv_dy = (dt/dy)*(vn[1:-1,1:-1] - vn[1:-1,:-2])

        # Diffusion terms
        d2u_dx2 = (dt/dx**2)*(un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1])
        d2u_dy2 = (dt/dy**2)*(un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])
        d2v_dx2 = (dt/dx**2)*(vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1])
        d2v_dy2 = (dt/dy**2)*(vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2])

        u[1:-1,1:-1] = un[1:-1,1:-1] - un[1:-1,1:-1]*du_dx - vn[1:-1,1:-1]*du_dy + nu*(d2u_dx2 + d2u_dy2)
        v[1:-1,1:-1] = vn[1:-1,1:-1] - vn[1:-1,1:-1]*dv_dx - un[1:-1,1:-1]*dv_dy+ nu*(d2v_dx2 + d2v_dy2)

        u[:, [0,-1]] = 1
        u[[0,-1], :] = 1
        v[:, [0,-1]] = 1
        v[[0,-1], :] = 1
        ts.append(ti)
        vels.append((u.copy(),v.copy()))
        
    return (u, v, ts, vels)

def main():

    (x, y, xmat, ymat, u, v) = setupGrid(2, 2, 100, 100)
    (u, v, ts, vels) = burg2d(x, y, u, v, 0.75, 0.01, 0.0009)

    p.contourf(x, y, np.sqrt(u**2+v**2), cmap='jet')
    p.colorbar()
    p.streamplot(x, y, u, v)
    p.show()


if __name__ == "__main__":
    main()

