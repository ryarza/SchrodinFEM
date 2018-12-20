import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import gmsh
from mpl_toolkits.mplot3d import Axes3D
from trigauss import *
from matplotlib import rc
import math

from shapely.geometry.polygon import Polygon
from descartes import PolygonPatch

rc('text', usetex=True)
mesh = gmsh.Mesh()

# Order of quadrature and mesh to use.
gauord = 7
mesh.read_msh('pseudoslit2.msh')

E = mesh.Elmts[9][1]
V = mesh.Verts[:,:2]

ne = E.shape[0]
nv = V.shape[0]
X = V[:,0]
Y = V[:,1]

def gradphi(funcnum,r,s):
    if funcnum == 0:
        grad = np.array([-3.0+4.0*r+4.0*s,-3.0+4.0*r+4.0*s])
    elif funcnum == 1:
        grad = np.array([-1.0+4.0*r,0.0])
    elif funcnum == 2:
        grad = np.array([0.0,4.0*s-1.0])        
    elif funcnum == 3:
        grad = np.array([4.0-8.0*r-4.0*s,-4.0*r])       
    elif funcnum == 4:
        grad = np.array([4.0*s,4.0*r])      
    elif funcnum == 5:
        grad = np.array([-4.0*s,4.0-4.0*r-8.0*s])
    return grad

def phi(funcnum,r,s):
    if funcnum == 0:
        val = (1.0-r-s)*(1.0-2.0*r-2.0*s)
    elif funcnum == 1:
        val = r * ( 2.0*r - 1.0)
    elif funcnum == 2:
        val = s * ( 2.0*s - 1.0)
    elif funcnum == 3:
        val = 4.0*r*(1.0-r-s)
    elif funcnum == 4:
        val = 4.0*r*s
    elif funcnum == 5:
        val = 4.0*s*(1.0-r-s)
    return val

def getJ(x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,r,s):
    j11 = (-3.0 + 4.0*r + 4.0*s)*x0 + (4.0*r - 1.0)*x1 - 4.0*(-1.0 + 2.0*r + s)*x3 + 4.0*s*(x4 - x5)
    j12 = (-3.0 + 4.0*r + 4.0*s)*x0 + (4.0*s - 1.0)*x2 + 4.0*r*(x4 - x3) - 4.0*x5*(2.0*s + r - 1.0)
    j21 = (-3.0 + 4.0*r + 4.0*s)*y0 + (4.0*r - 1.0)*y1 - 4.0*(-1.0 + 2.0*r + s)*y3 + 4.0*s*(y4 - y5)
    j22 = (-3.0 + 4.0*r + 4.0*s)*y0 + (4.0*s - 1.0)*y2 + 4.0*r*(y4 - y3) - 4.0*y5*(2.0*s + r - 1.0)
    return np.array([[j11,j12],[j21,j22]])

def getdbasis(r,s):
    return np.array([[-3.0+4.0*r+4.0*s,-1.0+4.0*r,0.0,4.0-8.0*r-4.0*s,4.0*s,-4.0*s],[-3.0+4.0*r+4.0*s,0.0,4.0*s-1.0,-4.0*r,4.0*r,4.0-4.0*r-8.0*s]])

# Initialize arrays
AA = np.zeros((ne, 36))
AA2 = np.zeros((ne, 36))
IA = np.zeros((ne, 36))
JA = np.zeros((ne, 36))

qx,qw = trigauss(gauord)

# Main loop
for ei in range(0,ne):
    Aelem = np.zeros((6,6))
    A2elem = np.zeros((6,6))
    K = E[ei,:]
    x0, y0 = X[K[0]], Y[K[0]]
    x1, y1 = X[K[1]], Y[K[1]]
    x2, y2 = X[K[2]], Y[K[2]]
    x3, y3 = X[K[3]], Y[K[3]]
    x4, y4 = X[K[4]], Y[K[4]]
    x5, y5 = X[K[5]], Y[K[5]]

    # estimate the integral using quadrature
    for qp in range(0,len(qw)):
        r = qx[qp,0]
        s = qx[qp,1]
        w = qw[qp]
        J = getJ(x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,r,s)
        invJ = la.inv(J.T)
        detJ = la.det(J)
        dbasis = getdbasis(r,s)
        dphi = invJ.dot(dbasis)
        Aelem += w * (dphi.T).dot(dphi) * detJ
        phis = np.array([[phi(i,r,s) for i in range(0,6)]])
        A2elem += w * detJ * (phis.T).dot(phis)
    AA2[ei,:] = A2elem.ravel()
    AA[ei, :] = Aelem.ravel()
    IA[ei, :] = np.repeat(K,6)
    JA[ei, :] = np.tile(K,6)


# Assembly
A = sparse.coo_matrix((AA.ravel(), (IA.ravel(), JA.ravel())),dtype=complex)
A = A.tocsr()
A = A.tocoo()

A2 = sparse.coo_matrix((AA2.ravel(), (IA.ravel(), JA.ravel())),dtype=complex)
A2 = A2.tocsr()
A2 = A2.tocoo()

# Boundary conditions
tol = 1e-8
tolx = 1e-4
toly = 1e-10
#Dflag = np.logical_or.reduce((abs(X) < tol,
#                              abs(Y) < tol,
#                              abs(X-1.0) < tol,
#                              abs(Y-1.0) < tol))

sxw = 0.005
ssep = 0.15

Dflag = np.logical_or.reduce((
			      np.logical_and.reduce((X-sxw<tolx,X+sxw>-1.0*tolx,Y-0.2>-toly)),
			      np.logical_and.reduce((X-sxw<tolx,X+sxw>-1.0*tolx,Y-ssep<toly,Y+ssep>-toly)),
			      np.logical_and.reduce((X-sxw<tolx,X+sxw>-1.0*tolx,Y+0.2<toly)),
				abs(X+1.0) < tolx,
			      abs(Y-1.0) < tolx,
                           abs(X-1.0) < tolx,
                            abs(Y+1.0) < tolx
			    ))

gflag = Dflag
nopeflag = np.logical_not(gflag)

uboundarytest = np.zeros((nv))
uboundarytest[gflag] = 1.0
fig = plt.figure()
triang = tri.Triangulation(X,Y)
surf = plt.tripcolor(X,Y,uboundarytest, triangles=E[:,:3], cmap=plt.cm.viridis,linewidth=0.2,vmin=0,vmax=1)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$|\psi|^{2}$')
plt.xlim([-1,1])
plt.ylim([-1,1])
fig.colorbar(surf)
fig.tight_layout()
plt.show()

plt.cla()
plt.scatter(X[gflag],Y[gflag],color='k',marker='.')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()

for k in range(0, len(A.data)):
    i = A.row[k]
    j = A.col[k]
    if Dflag[i] or Dflag[j]:
        if i == j:
            A.data[k] = complex(1.0,0)
        else:
            A.data[k] = complex(0.0,0)

for k in range(0, len(A2.data)):
    i = A2.row[k]
    j = A2.col[k]
    if Dflag[i] or Dflag[j]:
        if i == j:
            A2.data[k] = complex(1.0,0)
        else:
            A2.data[k] = complex(0.0,0.0)

hbar, m = 1.0, 1.0
lx, ly = 1.0, 1.0
def uex(x, y, t):
    #nxs = [1, 2]#, 1]
    #nys = [1, 2]#, 2]
    #phis = []
    #for nx, ny in zip(nxs, nys):
    #    En = (((hbar*np.pi)**2)/(2*m*lx*ly))*(nx**2 + ny**2)
    #    phit = np.exp(-1j*t*En/hbar)
    #    N = (np.sqrt(2)/lx)*(np.sqrt(2)/ly)
    #    phi = np.sin(nx*x*np.pi)*np.sin(ny*y*np.pi)*phit
    #    phis.append(phi)
    #return sum(phis).astype(np.complex128)
    return (np.sin(x*np.pi)*np.sin(y*np.pi)*np.exp(-1j*t*((((hbar*np.pi)**2)/(2*m*lx*ly))*(1**2 + 1**2)))+np.sin(2*x*np.pi)*np.sin(2*y*np.pi)*np.exp(-1j*t*((((hbar*np.pi)**2)/(2*m*lx*ly))*(2**2 + 2**2))))/np.sqrt(2)

def ic(x, y):
    #return uex(x, y)
    #return ( np.sin(x*np.pi)*np.sin(y*np.pi)+np.sin(2*x*np.pi)*np.sin(2*y*np.pi) ) / np.sqrt(2)
	if ( x<-0.2 and x>-0.8 and y>-0.8 and y<0.8 ):
		return np.exp(complex(0.0,1.0) * 40.0 * x)/np.sqrt(0.6*1.6)
	else:
		return 0.0

imagu = complex(0.0,1.0)
nt = 600
tf = 0.045
dt = tf / ( nt - 1.0 )
#U = sla.spsolve(A2+A*(hbar*imagu*(dt*1)/(4.0*m)),A2-A*(hbar*imagu*(dt*1)/(4.0*m)))
#U = np.dot(sla.inv(A2+A*(hbar*imagu*(dt*1)/(4.0*m))),A2-A*(hbar*imagu*(dt*1)/(4.0*m))).tocoo()
LHS = A2+A*(hbar*imagu*(dt*1)/(4.0*m))
RHS = A2-A*(hbar*imagu*(dt*1)/(4.0*m))
psi0 = np.zeros((nv),dtype=complex)
for i in range(0,nv):
	psi0[i] = ic(X[i],Y[i])
psi = np.array(psi0,dtype=complex)
pdens = np.real(psi * np.conj(psi))
fig = plt.figure()
for i in range(0,nt):
    if 1:
        plt.clf()
	triang = tri.Triangulation(X,Y)
	surf = plt.tripcolor(X,Y,pdens, triangles=E[:,:3], cmap=plt.cm.viridis,linewidth=0.2,vmin=0,vmax=1)
	plt.xlim([-1,1])
	plt.ylim([-1,1])
		#plt.tricontour(triang, pdens, colors='k',vmin=0,vmax=1)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.title('$|\psi|^{2}$ at t='+str(i*dt))
	fig.colorbar(surf)
	p0 = Polygon([(-sxw,-1),(-sxw,-0.2),(sxw,-0.2),(sxw,-1)])
	p1 = Polygon([(-sxw,-0.15),(-sxw,0.15),(sxw,0.15),(sxw,-0.15)])
	p2 = Polygon([(-sxw,0.2),(-sxw,1),(sxw,1),(sxw,0.2)])
	pf0 = PolygonPatch(p0,color='k')
	pf1 = PolygonPatch(p1,color='k')
	pf2 = PolygonPatch(p2,color='k')
	ax = fig.gca()
	ax.add_patch(pf0)
	ax.add_patch(pf1)
	ax.add_patch(pf2)
	fig.tight_layout()
        #plt.pause(.000000001)
	plt.savefig("Figure"+str(i).zfill(6)+".png")
	print "Figure "+str(i)+" saved."
    #psi = U.dot(psi)
	psi = sla.spsolve(LHS,RHS.dot(psi))
    pdens = np.real(psi * np.conj(psi))
