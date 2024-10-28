import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import sys
from scipy.ndimage.filters import gaussian_filter
import scipy
import math
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
from skimage import transform
from scipy.optimize import minimize
import argparse
import visvis as vv
import trimesh

def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos( costheta )
    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )
    return [x,y,z]

def random_two_vector():
    """
    Generates a random 2D unit vector (direction) with a uniform circular distribution
    """
    theta = np.random.uniform(0,np.pi*2)

    x = np.cos(theta)
    y = np.sin(theta)

    return [x,y]

def random_two_vector1():

    alpha1 = np.random.uniform(0,np.pi)
    #alpha2 = np.random.uniform(-np.pi,0)

    alpha2 = np.random.uniform(0, 2*np.pi*2)


    dL = [np.cos(alpha1), np.sin(alpha1)]
    dV = [np.cos(alpha2), np.sin(alpha2)]


    return dL, dV

def crop_infinite_scene_plane(scene_plane, dims=[406,406]):

    BPx,BPy = np.where(scene_plane == np.max(scene_plane))

    p0xx, p0yy = np.median(BPx), np.median(BPy)

    cropped = scene_plane[int(p0xx-dims[0]/2):int(p0xx+dims[0]/2), int(p0yy-dims[1]/2):int(p0yy+dims[1]/2)]

    return cropped



def display_mesh(verts, faces, normals, texture, save_path):

    mesh = vv.mesh(verts, faces, normals, texture)
    f = vv.gca()
    mesh.colormap = vv.CM_JET
    f.axis.visible = False

    #f.bgcolor = None #1,1,1 #None
    mesh.edgeShading = 'smooth'
    #mesh.clim = np.min(texture),np.max(texture)
    #mesh.clim = -0.05,0.02
    vv.callLater(1.0, vv.screenshot, save_path, vv.gcf(), sf=2, bg='w')
    vv.colorbar()
    vv.view({'zoom': 0.005, 'azimuth': 0.0, 'elevation': 0})
    #vv.view({'zoom': 0.006, 'azimuth': -80.0, 'elevation': -5.0})
    vv.use().Run()

    return 0

def display_image(image, BP=None):

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image.T, cmap="gray", origin='lower')
    plt.colorbar(fraction=0.046, pad=0.04)
    if BP is not None:
        plt.scatter(BP[0], BP[1])

    plt.title('Image plane S')
    #plt.axis('off')
    # plt.savefig('/home/karim/Bureau/Results/specularity_gt.png')
    plt.show()

def compute_normalized_vector(i,j,k,x,y):

    I = i * np.ones(x.shape) - x
    J = j * np.ones(y.shape) - y
    K = k * np.ones(x.shape)

    norm =  np.sqrt(I ** 2 + J ** 2 + K ** 2)

    return  I/norm, J/norm, K/norm

def compute_specular_image(L,V,n,size,scale=1, sigma=0, diffuse=False):

    X = np.arange(-size[0]/2, size[0]/2 + 1, step=scale)
    Y = np.arange(-size[1]/2, size[1]/2 + 1, step=scale)

    x, y = np.meshgrid(X, Y, indexing='ij')

    Rx_norm, Ry_norm, Rz_norm = compute_normalized_vector(L[0], L[1], -L[2], x, y)
    Vx_norm, Vy_norm, Vz_norm = compute_normalized_vector(V[0], V[1], V[2], x, y)


    scalar_product = -Vx_norm * Rx_norm - Vy_norm * Ry_norm - Vz_norm * Rz_norm

    image = 1.0 * np.maximum(0.0, np.power(scalar_product, n))

    print('position of maximum:', np.median(np.where(image==np.max(image)), axis=1))#[0][:])



    if sigma != 0:
        image = gaussian_filter(image, sigma=sigma)

        image /= np.max(image)

    if diffuse is True:

        print('diffuse is true!')
        #Nx_norm, Ny_norm, Nz_norm = compute_normalized_vector(x, y, 1., x, y)
        Lx_norm, Ly_norm, Lz_norm = compute_normalized_vector(L[0], L[1], L[2], x, y)

        #i_diffuse = Lx_norm * Nx_norm + Ly_norm * Ny_norm + Lz_norm * Nz_norm
        i_diffuse = Lz_norm

        image += i_diffuse

    return image #i_diffuse

def compute_normalized_vector2(i,j,k,x,y,z):

    I = i * np.ones(x.shape) - x
    J = j * np.ones(y.shape) - y
    K = k * np.ones(z.shape) - z

    norm =  np.sqrt(I ** 2 + J ** 2 + K ** 2)

    return  I/norm, J/norm, K/norm

def compute_specular_image2(L,V,n,size,scale=1, sigma=0, diffuse=False, normal=[-7.65716481e-01,  5.76701098e-04, -6.43177998e-01], distance=1000):

    X = np.arange(-size[0]/2, size[0]/2 + 1, step=scale)
    Y = np.arange(-size[1]/2, size[1]/2 + 1, step=scale)

    x, y = np.meshgrid(X, Y, indexing='ij')

    a = normal[0]
    b = normal[1]
    c = normal[2]

    d= distance

    z = -(a/c)*x -(b/c)*y + d/c


    #z = 0.12861723162963065 * x + 0.2024845304814665 * y + 1.0964608113924048

    Rx_norm, Ry_norm, Rz_norm = compute_normalized_vector2(L[0], L[1], -L[2], x, y, z)
    Vx_norm, Vy_norm, Vz_norm = compute_normalized_vector2(V[0], V[1], V[2], x, y, z)


    scalar_product = -Vx_norm * Rx_norm - Vy_norm * Ry_norm - Vz_norm * Rz_norm

    image = 1.0 * np.maximum(0.0, np.power(scalar_product, n))

    world_position_BP = np.median(np.where(image==np.max(image)), axis=1)


    print('position of maximum:', world_position_BP)#[0][:])

    Nx = int(x[int(world_position_BP[0]),int(world_position_BP[1])])
    Ny = int(y[int(world_position_BP[0]),int(world_position_BP[1])])
    Nz = int(z[int(world_position_BP[0]),int(world_position_BP[1])])

    print([Nx,Ny,Nz])


    #print(x[int(world_position_BP[0]),int(world_position_BP[1])])
    #print(y[int(world_position_BP[0]),int(world_position_BP[1])])
    #print(z[int(world_position_BP[0]),int(world_position_BP[1])])

    normal = np.array([Nx, Ny, Nz]).astype(np.float64)# - np.array([-size[0]/2, -size[1]/2,-1000.])
    normal /= np.linalg.norm(normal)

    print("real world normal", normal)


    if sigma != 0:
        image = gaussian_filter(image, sigma=sigma)

        image /= np.max(image)

    if diffuse is True:

        print('diffuse is true!')
        #Nx_norm, Ny_norm, Nz_norm = compute_normalized_vector(x, y, 1., x, y)
        Lx_norm, Ly_norm, Lz_norm = compute_normalized_vector(L[0], L[1], L[2], x, y)

        #i_diffuse = Lx_norm * Nx_norm + Ly_norm * Ny_norm + Lz_norm * Nz_norm
        i_diffuse = Lz_norm

        image += i_diffuse

    return image #i_diffuse

def compute_diffuse_image(L,size,c=1,scale=1):

    X = np.arange(-size[0]/2, size[0]/2 + 1, step=scale)
    Y = np.arange(-size[1]/2, size[1]/2 + 1, step=scale)

    x, y = np.meshgrid(X, Y, indexing='ij')

    Lx_norm, Ly_norm, Lz_norm = compute_normalized_vector(L[0], L[1], L[2], x, y)

    return  c*Lz_norm


def determine_brightest_point(V, L):

    R = [L[0], L[1], -L[2]]
    mu = V[2] / (V[2] + L[2])
    lamda = (mu - 1) / mu

    return mu * np.array(R) + (1 - mu) * np.array(V), lamda


def change_coordinate_range(xOld, xLoOld, xHiOld, xLoNew, xHiNew):

    xNew = (xOld - xLoOld) / (xHiOld - xLoOld) * (xHiNew - xLoNew) + xLoNew

    return xNew


def determine_brightest_point_from_image(image):

    BPx,BPy = np.where(image == np.max(image))

    p0xx, p0yy = np.median(BPx), np.median(BPy)

    return [p0xx, p0yy]

def extract_isocontour(image, level):

    contour = measure.find_contours(image, level)[0]

    return contour

def extract_all_isocontours(image, level):

    contours = measure.find_contours(image, level)

    return contours

def plot_all_isocontours(image, level):

    contours = extract_all_isocontours(image, level)
    plt.imshow(image.T, cmap='gray', origin='lower')

    from matplotlib.pyplot import cm
    color = iter(cm.Accent(np.linspace(0, 1, len(contours))))

    for contour in contours:
        c = next(color)
        plt.plot(contour[:,0], contour[:,1], linewidth=2, c=c)

    plt.show()



def densify_isocontour_by_factor(a, factor):
    """Densify a 2D array using np.interp.

    Parameters
    ----------
    a : array
        A 2D array of points representing a polyline/polygon boundary.
    fact : number
        The factor to densify the line segments by.
    """
    a = np.squeeze(a)
    n_fact = len(a) * factor
    b = np.arange(0, n_fact, factor)
    b_new = np.arange(n_fact - 1)  # where we want to interpolate
    c0 = np.interp(b_new, b, a[:, 0])
    c1 = np.interp(b_new, b, a[:, 1])
    n = c0.shape[0]
    c = np.zeros((n, 2))
    c[:, 0] = c0
    c[:, 1] = c1

    return c


def determine_k(t, c, n):

    #tau = (t / c) ** (1 / float(n))

    tau = np.power((t / c),(1/n))

    print('tau=', tau)

    return 1 - tau ** 2


def real_coefficients(k, lmda, V):

    V_norm = np.sqrt(V[0] ** 2 + V[1] ** 2 + V[2] ** 2)

    a = [0, 0, 0, 0, 0, 0]

    a[5] = k * lmda ** 2 * V_norm ** 4
    a[4] = -2 * V[0] * k * lmda * V_norm ** 2 * (lmda + 1)
    a[3] = V_norm ** 2 * ((k * (lmda ** 2 + 1)) - (lmda - 1) ** 2)

    a[2] = (V_norm ** 2 * k * (lmda ** 2 + 1)) - (V[2] ** 2 * (lmda - 1) ** 2) + (4 * V[0] ** 2 * k * lmda)
    a[1] = -2 * V[0] * k * (lmda + 1)
    a[0] = k

    #m = np.max(np.asarray(a))
    #print(m)

    return a


def vector_norm(vec):

    norm = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

    return [vec[0] / norm, vec[1] / norm, vec[2] / norm]


def apply_quartic_fitting(x, y, a):

    return a[0]*(x**2 + y**2)**2 + a[1]*x*(x**2 + y**2) + a[2]*x**2 + a[3]*y**2 + a[4]*x + a[5]

def apply_quartic_fitting_L_equals_V(x, y, Vz, k):

    #Vz = V[2]
    alpha = (2-k)/k

    gamma1 = Vz**2 * (alpha + np.sqrt(alpha**2 -1))
    gamma2 = Vz**2 * (alpha - np.sqrt(alpha**2 -1))

    return (x**2 + y**2-gamma1)*(x**2 + y**2-gamma2)


def apply_ellipse_fitting(x, y, a):
    return a[0] * x ** 2 + a[1] * x * y + a[2] * y ** 2 + a[3] * x + a[4] * y + a[5]


def plot_3Dscene1(V, L, imag, P0):

    fig = plt.figure(figsize=(16, 16))

    plt.rcParams["figure.figsize"] = (12, 12)

    nx, ny = imag.shape

    min_val = imag.min()
    max_val = imag.max()

    # colormap = plt.cm.YlOrRd
    #colormap = plt.cm.PuRd  # Purples
    colormap = plt.cm.gray

    X, Y = np.mgrid[-nx/2:nx/2, -ny/2:ny/2]

    Z = np.zeros((nx, ny))
    #fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.grid(False)
    #ax.plot_surface(X, Y, Z, color='k', alpha=1.0, shade=False)
    ax.plot_surface(X, Y, Z, alpha=0.25, zorder=0, rstride=1, cstride=1, facecolors=colormap((imag - min_val) / (max_val - min_val)), shade=False)
    #ax.set_title("3D scene")

    R = [L[0],L[1],-L[2]]

    p0 = V
    p1 = L
    p2 = R

    center = (P0[0], P0[1], 0)

    fr = [(0,0,0), (0,0,0)]
    #fr = [center, center, center, center]
    to = [(p0[0], p0[1], p0[2]), (p1[0], p1[1], p1[2])]
    cr = ["cyan", "blue"]

    #for (from_, to_, colr_) in zip(fr, to, cr):
    #    ax.quiver(*from_, *to_, colors=colr_, length=1)

    ax.scatter(p0[0], p0[1], p0[2], color='cyan', s=70)
    #ax.text(p0[0]+35, p0[1]+5, p0[2], 'V', size=30, zorder=2, color='k')

    ax.scatter(p1[0], p1[1], p1[2], color='b', s=70)
    #ax.text(p1[0]-30, p1[1] , p1[2], 'L', size=30, zorder=2, color='k')


    #ax.scatter(P0[0], P0[1], 0, color='red', s=70)
    #ax.text(P0[0]-25, P0[1]+5 , 0, 'P0', size=30, zorder=3, color='red')


    #ax.text(0, 0, 10+(V[2]+L[2])/2, 'N', size=30, zorder=2, color='k')

    ax.view_init(17, 100)


    plt.show()


def plot_3Dscene(V, L, imag, P0):

    fig = plt.figure(figsize=(16, 16))

    plt.rcParams["figure.figsize"] = (12, 12)

    nx, ny = imag.shape

    min_val = imag.min()
    max_val = imag.max()

    # colormap = plt.cm.YlOrRd
    #colormap = plt.cm.PuRd  # Purples
    colormap = plt.cm.gray

    X, Y = np.mgrid[-nx/2:nx/2, -ny/2:ny/2]

    Z = np.zeros((nx, ny))
    #fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.05, zorder=0, rstride=1, cstride=1,
                    facecolors=colormap((imag - min_val) / (max_val - min_val)), shade=False)
    ax.set_title("3D scene")

    R = [L[0],L[1],-L[2]]

    p0 = V
    p1 = L
    p2 = R

    center = (P0[0], P0[1], 0)

    fr = [(0,0,0), (0,0,0), (0,0,0), (0,0,0)]
    #fr = [center, center, center, center]
    to = [(p0[0], p0[1], p0[2]), (p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2]), (0, 0, (V[2]+L[2])/2)]
    cr = ["cyan", "blue", "magenta", "green"]

    for (from_, to_, colr_) in zip(fr, to, cr):
        ax.quiver(*from_, *to_, colors=colr_, length=1)

    ax.scatter(p0[0], p0[1], p0[2], color='cyan', s=70)
    ax.text(p0[0]+35, p0[1]+5, p0[2], 'V', size=30, zorder=2, color='k')

    ax.scatter(p1[0], p1[1], p1[2], color='b', s=70)
    ax.text(p1[0]-30, p1[1] , p1[2], 'L', size=30, zorder=2, color='k')

    ax.scatter(p2[0], p2[1], p2[2], color='magenta', s=70)
    ax.text(p2[0]-20, p2[1] , p2[2], 'R', size=30, zorder=2, color='k')

    ax.scatter(P0[0], P0[1], 0, color='red', s=70)
    ax.text(P0[0]-25, P0[1]+5 , 0, 'P0', size=30, zorder=3, color='red')


    ax.text(0, 0, 10+(V[2]+L[2])/2, 'N', size=30, zorder=2, color='k')

    ax.view_init(17, 100)


    plt.show()


def rotation_matrix(theta):

    theta = np.radians(theta)

    mat = np.identity(3)

    mat[0,0]  = math.cos(theta)
    mat[0,1]  = -math.sin(theta)
    mat[1,0]  = math.sin(theta)
    mat[1,1]  = math.cos(theta)

    return mat

def simulate_projective_transformation(theta=0, scale=1, translation=[0,0], v=[0,0], K=[1,1,0]):

    theta = np.radians(theta)

    Hs = np.identity(3) # similarity transformation
    Hs[0,0]  = scale*math.cos(theta)
    Hs[0,1]  = -scale*math.sin(theta)
    Hs[1,0]  = scale*math.sin(theta)
    Hs[1,1]  = scale*math.cos(theta)
    Hs[0,2]  = translation[0]
    Hs[1,2]  = translation[1]

    Hp = np.identity(3)
    Hp[2,0] = v[0]
    Hp[2,1] = v[1]

    #For the parameters of K, see https://towardsdatascience.com/camera-intrinsic-matrix-with-example-in-python-d79bf2478c12

    Ha = np.identity(3)
    Ha[0,0] = K[0]
    Ha[1,1] = K[1]/math.sin(K[2])
    Ha[0,1] = -K[0]/math.tan(K[2])

    H = np.dot(np.dot(Hs,Ha), Hp)

    return H

def recover_rotation_from_homography_matrix(H):

    u, _, vh = np.linalg.svd(H[0:2, 0:2])
    R = u @ vh
    angle = math.atan2(R[1,0], R[0,0])

    #angle = math.atan2(H[1, 0], H[0, 0])

    return np.degrees(angle)

#https://stackoverflow.com/questions/8927771/computing-camera-pose-with-homography-matrix-based-on-4-coplanar-points
# The camera rotation will be the inverse of the homography rotation and the traslation will be -1*(homography_traslation) * scale_factor.

def cameraPoseFromHomography(H):

    H1 = H[:, 0]
    H2 = H[:, 1]
    H3 = np.cross(H1, H2)

    norm1 = np.linalg.norm(H1)
    norm2 = np.linalg.norm(H2)
    tnorm = (norm1 + norm2) / 2.0

    T = H[:, 2] / tnorm

    return np.mat([H1, H2, H3, T])

def center_image(image, cX, cY):
  height, width = image.shape
  wi=(width/2)
  he=(height/2)

  offsetX = (wi-cX)
  offsetY = (he-cY)
  T = np.float32([[1, 0, offsetX], [0, 1, offsetY]])
  centered_image = cv2.warpAffine(image, T, (width, height), flags=cv2.INTER_CUBIC)

  centered_image /= np.max(centered_image)

  return centered_image


def display_2_images_side_by_side(image1, image2, title1='image1', title2='image2'):

    size = image1.shape#[1000,1000]

    plt.close('all')

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image1.T, cmap='RdPu', origin='lower', extent=[-size[0] / 2., size[0] / 2., -size[1] / 2., size[1] / 2.])
    ax[1].imshow(image2.T, cmap='RdPu', origin='lower', extent=[-size[0] / 2., size[0] / 2., -size[1] / 2., size[1] / 2.])
    ax[0].set_title(title1, fontsize=10)
    ax[1].set_title(title2, fontsize=10)

    plt.show()
    #plt.close('all')

    #plt.close(fig)




def recover_scene_parameters(a, V):

    V_norm = np.sqrt(V[0] ** 2 + V[1] ** 2 + V[2] ** 2)
    lamda = -np.sqrt(np.absolute(a[5] / (a[0] + sys.float_info.epsilon))) / V_norm ** 2
    # k = np.divide (V_norm**2 * (lamda-1)**2 * a[5], V_norm**2 * (lamda**2+1)**2 * a[5] -a[2])
    k = a[0]

    return lamda, k, np.sqrt(1 - k)


def retrieve_n(t, c, tau):

    return (np.log(t) - np.log(c)) / (np.log(tau))

def rotate_180(array):
    M, N = array.shape
    out = np.zeros(array.shape)
    for i in range(M):
        for j in range(N):
            out[i, N-1-j] = array[M-1-i, j]
    return out

def determine_isocontour_radius(x, y):

    return np.sqrt(x**2 + y**2)

### For L = V , we retrieve n and Vz using this function

def retrieve_n_Vz(image, t=0.5):

    c = np.max(image)

    epsilon = 1e-64 #sys.float_info.epsilon

    P0 = determine_brightest_point_from_image(image)
    isocontour1 =  extract_isocontour(image, epsilon)#0.00001*sys.float_info.epsilon)

    #print(sys.float_info.epsilon)
    print(epsilon)

    print('Brightest point:', P0)

    Vz = determine_isocontour_radius(isocontour1[0,0]-P0[0], isocontour1[0,1]-P0[1])

    isocontour2 = extract_isocontour(image, t)

    r1 = determine_isocontour_radius(isocontour2[0, 0] - P0[0], isocontour2[0, 1] - P0[1])

    k = np.divide( 2*(r1/Vz), (r1/Vz)**2+1 )**2

    tau = np.sqrt(1-k)

    n = np.divide(np.log(t)-np.log(c), np.log(tau))

    return n, Vz









def plot_ellipse(a, image, gtx, gty, BP, theta, title = 'Ellipse contours (ground truth)'):

    size = image.shape

    x = np.arange(-size[0]/2, size[0]/2, 1)-BP[0]
    y = np.arange(-size[1]/2, size[1]/2, 1)-BP[1]

    x1 = math.cos(theta)*x-math.sin(theta)*y
    y1 = math.sin(theta)*x+math.cos(theta)*y

    [X, Y] = np.meshgrid(x1, y1)

    BP_rotated = [0., 0.]

    BP_rotated[0] = math.cos(theta) * BP[0] - math.sin(theta) * BP[1]
    BP_rotated[1] = math.sin(theta) * BP[0] + math.cos(theta) * BP[1]

    fig, ax = plt.subplots(1, 1)

    #if V is not None:
    #    Q = apply_quartic_fitting_L_equals_V(X, Y, V, k)

    #else:
    Q = apply_ellipse_fitting(X, Y, a)

    ax.scatter(gtx, gty, s=3, color='red')

    ax.plot(0, 0, marker="o", markersize=4, markeredgecolor="red", markerfacecolor="red")

    #CS = ax.contour(X, Y, Q, cmap='cool', zorder=0.99)
    #ax.clabel(CS, inline=True, fontsize=10)

    ax.contour(X, Y, Q, 0, colors='chartreuse', zorder=0.99)

    image = ndimage.rotate(image, (theta*180)/math.pi, reshape=False)

    ax.imshow(image.T, extent=[-BP_rotated[0]-(size[0] / 2.), -BP_rotated[0]+(size[0] / 2.), -BP_rotated[1]-(size[1] / 2.),\
                               -BP_rotated[1]+(size[1] / 2.)], cmap='gray', origin='lower')

    #ax.imshow(image.T, extent=[-BP_rotated[0]-2*size[0], -BP_rotated[0]+2*size[0], -BP_rotated[1]-2*size[1],\
    #                           -BP_rotated[1]+2*size[1]], cmap='gray', origin='lower')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()


def display_fitting_result(a, image, isocontourX, isocontourY, fitting = 'ellipse', allcontours=True):

    size = image.shape

    x = np.arange(0, size[0], 1)
    y = np.arange(0, size[1], 1)

    [X, Y] = np.meshgrid(x, y)

    fig, ax = plt.subplots(1, 1)

    if fitting == 'ellipse':

        Q = apply_ellipse_fitting(X, Y, a)
    else:
        Q = apply_quartic_fitting(X, Y, a)

    if allcontours is True:

        CS = ax.contour(X, Y, Q, cmap='winter', zorder=0.99)
        ax.clabel(CS, inline=True, fontsize=10)
    else:

        ax.contour(X, Y, Q, 0, colors='mediumspringgreen', zorder=0.99)

    ax.scatter(isocontourX, isocontourY, s=8, color='red')#, zorder=0.99)

    ax.imshow(image.T, cmap='gray', origin='lower')

    ax.set_title('fitting result')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()

def display_fitting_result_aligned_with_brightest_point(a, image, BP, isocontourX, isocontourY, fitting = 'ellipse', allcontours=True):

    size = image.shape

    x = np.arange(-size[0]/2, size[0]/2, 1)-BP[0]
    y = np.arange(-size[1]/2, size[1]/2, 1)-BP[1]
    [X, Y] = np.meshgrid(x, y)

    plt.close('all')

    fig, ax = plt.subplots(1, 1)

    if fitting == 'ellipse':

        Q = apply_ellipse_fitting(X, Y, a)
    else:
        Q = apply_quartic_fitting(X, Y, a)

    if allcontours is True:

        CS = ax.contour(X,  Y, Q, cmap='winter', zorder=0.99)
        ax.clabel(CS, inline=True, fontsize=10)
    else:

        ax.contour(X, Y, Q, 0, colors='mediumspringgreen', zorder=0.99)

    ax.scatter(isocontourX, isocontourY, s=8, color='red')#, zorder=0.99)

    ax.scatter(0, 0, s=10, color='deeppink')



    ax.imshow(image.T, extent=[-BP[0]-(size[0] / 2.), -BP[0]+(size[0] / 2.), -BP[1]-(size[1] / 2.),\
                               -BP[1]+(size[1] / 2.)], cmap='gray', origin='lower')

    ax.set_title('fitting result')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()



def plot_quartic(a, image, gtx, gty, BP, theta, title = 'Quartic contours (ground truth)', V=None, k=None):

    size = image.shape

    x = np.arange(-size[0]/2, size[0]/2, 1)-BP[0]
    y = np.arange(-size[1]/2, size[1]/2, 1)-BP[1]

    x1 = math.cos(theta)*x-math.sin(theta)*y
    y1 = math.sin(theta)*x+math.cos(theta)*y

    [X, Y] = np.meshgrid(x1, y1)

    BP_rotated = [0., 0.]

    BP_rotated[0] = math.cos(theta) * BP[0] - math.sin(theta) * BP[1]
    BP_rotated[1] = math.sin(theta) * BP[0] + math.cos(theta) * BP[1]

    fig, ax = plt.subplots(1, 1)

    if V is not None:
        Q = apply_quartic_fitting_L_equals_V(X, Y, V, k)

    else:
        Q = apply_quartic_fitting(X, Y, a)

    ax.scatter(gtx, gty, s=3, color='red')

    ax.plot(0, 0, marker="o", markersize=4, markeredgecolor="red", markerfacecolor="red")

    #CS = ax.contour(X, Y, Q, cmap='jet', zorder=0.99)
    #ax.clabel(CS, inline=True, fontsize=10)

    ax.contour(X, Y, Q, 0, colors=['chartreuse', 'magenta'], zorder=0.99)

    #ax.contour(X, Y, Q, 0, colors='chartreuse', zorder=0.99)

    image = ndimage.rotate(image, (theta*180)/math.pi, reshape=False)

    ax.imshow(image.T, extent=[-BP_rotated[0]-(size[0] / 2.), -BP_rotated[0]+(size[0] / 2.), -BP_rotated[1]-(size[1] / 2.),\
                               -BP_rotated[1]+(size[1] / 2.)], cmap='gray', origin='lower')

    #ax.imshow(image.T, extent=[-BP_rotated[0]-2*size[0], -BP_rotated[0]+2*size[0], -BP_rotated[1]-2*size[1],\
    #                           -BP_rotated[1]+2*size[1]], cmap='gray', origin='lower')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()
#def align_origin_with_the_brightest_point(image):




def fit_ellipse_Fitzgibbon(x, y):

    D = np.array([x ** 2, x * y, y ** 2, x, y, x * 0 + 1]).T

    S = np.dot(D.T, D)

    C = np.zeros((6, 6))

    C[0, 2] = -2
    C[2, 0] = -2
    C[1, 1] = 1

    geval, gevec = scipy.linalg.eig(S, C)

    low = np.where(geval == np.min(geval))

    return gevec[:, low]


####################################################################################################################################
# Reference for improved ellipse fitting: http://andrewd.ces.clemson.edu/courses/cpsc482/papers/HF98_stableLeastSquaresEllipses.pdf
# We use this method for ellipse fitting which is a stable (improved version) of the method of Fitzgiborn (original formulation)
####################################################################################################################################

def fit_ellipse_improved_Fitzgibbon(x, y):

    D1 = np.array([x ** 2, x * y, y ** 2]).T #quadratic part of the design matrix
    D2 = np.array([x, y, x * 0 + 1]).T #linear part of the design matrix

    S1 = D1.T @ D1 #quadratic part of the scatter matrix
    S2 = D1.T @ D2 #combined part of the scatter matrix
    S3 = D2.T @ D2 #linear part of the scatter matrix

    T = - np.linalg.inv(S3) @ S2.T #for getting a2 from a1

    M = S1 + S2 @ T #reduced scatter matrix

    M = np.array([M[2,:]/2, - M[1,:], M[0,:]/ 2]) #premultiply by inv(C1)

    geval, gevec = np.linalg.eig(M) #solve eigensystem

    #cond = 4 * np.multiply(gevec[0,:],gevec[2,:]) - np.power(gevec[1,:],2) #evaluate a^T.C.a

    loc = np.where(4 * np.multiply(gevec[0,:],gevec[2,:]) - np.power(gevec[1,:],2) > 0)[0] #evaluate a^T.C.a

    if loc.size == 0:

        #print('the condition is not satisfied!')

        return np.array([1,0,1,0,0,-1]).reshape((6, 1))

    else:
        #print('okay')

        a1 = gevec[:, loc[0]].reshape((3, 1)) #eigenvector for min. pos. eigenvalue
        return np.concatenate((a1, T@a1)) #ellipse coefficients




def _check_data_dim(data, dim):
    if data.ndim != 2 or data.shape[1] != dim:
        raise ValueError('Input data must have shape (N, %d).' % dim)

class BaseModel(object):

    def __init__(self):
        self.params = None


class EllipseModel(BaseModel):

    def estimate(self, data):
        """Estimate circle model from data using total least squares.
        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        References
        ----------
        .. [1] Halir, R.; Flusser, J. "Numerically stable direct least squares
               fitting of ellipses". In Proc. 6th International Conference in
               Central Europe on Computer Graphics and Visualization.
               WSCG (Vol. 98, pp. 125-132).
        """
        # Original Implementation: Ben Hammel, Nick Sullivan-Molina
        # another REFERENCE: [2] http://mathworld.wolfram.com/Ellipse.html
        _check_data_dim(data, dim=2)

        # to prevent integer overflow, cast data to float, if it isn't already
        float_type = np.promote_types(data.dtype, np.float64)
        data = data.astype(float_type, copy=False)

        x = data[:, 0]
        y = data[:, 1]

        # Quadratic part of design matrix [eqn. 15] from [1]
        D1 = np.vstack([x ** 2, x * y, y ** 2]).T
        # Linear part of design matrix [eqn. 16] from [1]
        D2 = np.vstack([x, y, np.ones_like(x)]).T

        # forming scatter matrix [eqn. 17] from [1]
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2

        # Constraint matrix [eqn. 18]
        C1 = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])

        try:
            # Reduced scatter matrix [eqn. 29]
            M = np.linalg.inv(C1) @ (S1 - S2 @ np.linalg.inv(S3) @ S2.T)
        except np.linalg.LinAlgError:  # LinAlgError: Singular matrix
            return False

        # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors
        # from this equation [eqn. 28]
        eig_vals, eig_vecs = np.linalg.eig(M)

        # eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = 4 * np.multiply(eig_vecs[0, :], eig_vecs[2, :]) \
               - np.power(eig_vecs[1, :], 2)
        a1 = eig_vecs[:, (cond > 0)]
        # seeks for empty matrix
        if 0 in a1.shape or len(a1.ravel()) != 3:
            return False
        a, b, c = a1.ravel()

        # |d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a2 = -np.linalg.inv(S3) @ S2.T @ a1
        d, f, g = a2.ravel()

        A, B, C, D, F, G = a, b, c, d, f, g

        # eigenvectors are the coefficients of an ellipse in general form
        # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 (eqn. 15) from [2]
        b /= 2.
        d /= 2.
        f /= 2.

        # finding center of ellipse [eqn.19 and 20] from [2]
        x0 = (c * d - b * f) / (b ** 2. - a * c)
        y0 = (a * f - b * d) / (b ** 2. - a * c)

        # Find the semi-axes lengths [eqn. 21 and 22] from [2]
        numerator = a * f ** 2 + c * d ** 2 + g * b ** 2 \
                    - 2 * b * d * f - a * c * g
        term = np.sqrt((a - c) ** 2 + 4 * b ** 2)
        denominator1 = (b ** 2 - a * c) * (term - (a + c))
        denominator2 = (b ** 2 - a * c) * (- term - (a + c))
        width = np.sqrt(2 * numerator / denominator1)
        height = np.sqrt(2 * numerator / denominator2)

        # angle of counterclockwise rotation of major-axis of ellipse
        # to x-axis [eqn. 23] from [2].
        phi = 0.5 * np.arctan((2. * b) / (a - c))
        if a > c:
            phi += 0.5 * np.pi

        self.params = np.nan_to_num([A, B, C, D, F, G, x0, y0, width, height, phi]).tolist()
        self.params = [float(np.real(x)) for x in self.params]
        return True





def fit_ellipse_lstsq(x, y):

    x = x.flatten()
    y = y.flatten()

    A = np.array([x** 2, x * y, y** 2, x, y, x * 0 + 1])
    ATA = np.dot(A.T, A)

    u, s, vh = np.linalg.svd(ATA, hermitian=True)

    return u[:, u.shape[1] - 1]


# determine the rotation angle of isocontour (and specular highlight) using ellipse fitting
# we determine counterclockwise rotation angle between the principal axis of symmetry and the X-axis
# For more details, see: https://stackoverflow.com/questions/67537630/ellipse-fitting-to-determine-rotation-python

def determine_rotation(isocontour):

    e = fit_ellipse_Fitzgibbon(isocontour[:, 0], isocontour[:, 1])[:, 0, 0]

    if e[5] > 0:
        e *= -1
        e /= e[5]

    #theta = (math.pi / 2) + 0.5 * math.atan(e[1]/(e[0]-e[2]))
    theta = 0.5 * math.atan(e[1] / (e[0] - e[2]))

    return (theta * 180) / math.pi


def rotate_vector(vec, rotation_degrees):
    rotation_radians = np.radians(rotation_degrees)

    rotation_axis = np.array([0, 0, 1])

    rotation_vector = rotation_radians * rotation_axis

    rotation = R.from_rotvec(rotation_vector)

    return rotation.apply(vec)


def align_vector(vec, P0, theta):

    # theta must be in degrees

    aligned_vec = [vec[0] - P0[0], vec[1] - P0[1], vec[2]]

    return rotate_vector(aligned_vec, theta)


def align_isocontour(isocontour, dims, P0, theta):

    #theta must be in radians

    isox = change_coordinate_range(isocontour[:, 0] - P0[0], 0, dims[0], -dims[0] / 2, dims[0] / 2)
    isoy = change_coordinate_range(isocontour[:, 1] - P0[1], 0, dims[1], -dims[1] / 2, dims[1] / 2)

    #isox = isox - P0[0]
    #isoy = isoy - P0[1]

    isoxx = math.cos(theta)*isox - math.sin(theta)*isoy

    isoyy = math.sin(theta) * isox + math.cos(theta) * isoy

    return isoxx, isoyy

def align_isocontour_image(isocontour, dims, P0, theta, world_coordinates=True):

    #theta must be in radians

    if world_coordinates is True:

        isox = change_coordinate_range(isocontour[:, 0] - P0[0], 0, dims[0], -dims[0] / 2, dims[0] / 2)
        isoy = change_coordinate_range(isocontour[:, 1] - P0[1], 0, dims[1], -dims[1] / 2, dims[1] / 2)

    else:

        isox = isocontour[:, 0] - P0[0]
        isoy = isocontour[:, 1] - P0[1]

        print("hello world")

    isoxx = math.cos(theta)*isox - math.sin(theta)*isoy

    isoyy = math.sin(theta)*isox + math.cos(theta)*isoy

    return isoxx, isoyy

def rotate_isocontour_around_brightest_point(isocontour,p0,theta): #rotate x,y around p0 by theta (rad)

    xr=math.cos(theta)*(isocontour[:,0]-p0[0])-math.sin(theta)*(isocontour[:,1]-p0[1]) + p0[0]
    yr=math.sin(theta)*(isocontour[:,0]-p0[0])+math.cos(theta)*(isocontour[:,1]-p0[1]) + p0[1]

    return np.concatenate((xr.reshape(-1, 1), yr.reshape(-1, 1)), axis=1)


import cv2
def rot_image_around_point(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def plot_isocurve(isocontour, image, isovalue):

    fig = plt.figure(figsize=(10, 10))
    plt.plot(isocontour[:,0],isocontour[:,1], color='r', linewidth=2)
    plt.imshow(image.T, cmap='gray', origin='lower')
    plt.title('Isocurve: I =  %1.2f' % isovalue)
    plt.show()

def display_image_in_world_coordinates(image, BP=None):

    fig = plt.figure(figsize=(10, 10))
    plt.title('Image plane S in real world coordinates')
    size = image.shape
    plt.imshow(image.T, origin='lower', extent=[-size[0] / 2., size[0] / 2., -size[1] / 2., size[1] / 2.], cmap='gray')
    plt.colorbar(fraction=0.046, pad=0.04)

    if BP is not None:
        plt.scatter(BP[0], BP[1])

    plt.show()


def plot_many_isocontours(image):

    size = image.shape

    x = np.arange(0, size[0])
    y = np.arange(0, size[1])

    [X, Y] = np.meshgrid(x, y)

    fig, ax = plt.subplots(1, 1)

    CS = ax.contour(X, Y, image.T, cmap='jet', zorder=0.99)
    ax.clabel(CS, inline=True, fontsize=10)

    #ax.contour(X, Y, Q, 0, colors='chartreuse', zorder=0.99)


    ax.imshow(image.T, cmap='gray', origin='lower')


    plt.show()

## We determine ellipse parameters using
# To compute a, b, c, d, e, and f, we establish correspondences between coefficients of ellipse fitting using Fitzgibborn and
# the above coefficients defined in: https://stackoverflow.com/questions/67537630/ellipse-fitting-to-determine-rotation-python
# input: vector of coefficients for ellipse fitting using Fitzgibborn method

# Outputs:
# A: semi major axis length
# B: semi minor axis length

def determine_ellipse_parameters(e):

    a = -np.divide(e[0],e[5])
    b = -np.divide(e[1],2*e[5])
    c = -np.divide(e[2],e[5])
    d = -np.divide(e[3],2*e[5])
    f = -np.divide(e[4],2*e[5])

    #g = -1.

    #numerator = 2*( a*f**2 + c*d**2 + g*b**2 -2*b*d*f -a*c*g )

    numerator = 2*( a*f**2 + c*d**2 - b**2 - 2*b*d*f + a*c)

    denB = (b**2 - a*c) * ( np.sqrt( (a-c)**2 + 4*b**2 ) - (a+c) )
    denA = (b**2 - a*c) * ( -np.sqrt( (a-c)**2 + 4*b**2 ) - (a+c) )

    A = np.sqrt(numerator / denA)
    B = np.sqrt(numerator / denB)

    center = [0,0]
    center[0] = (c*d - b*f) / (b**2 - a*c)
    center[1] = (a*f - b*d) / (b**2 - a*c)

    return A, B, center

def plothistogram(img_gray):
    vals = img_gray.flatten()
    # plot histogram with 255 bins
    b, bins, patches = plt.hist(vals, 255)
    plt.xlim([70,255])
    plt.show()

def compute_p_norm(Gx,Gy,p=2):

    return np.power(Gx**p+Gy**p,1/p)


def image_skeleton(img_gray):

    Gx, Gy = np.gradient(img_gray)

    pp = 100

    grad_norm2 = compute_p_norm(Gx,Gy,2)
    grad_normp = compute_p_norm(Gx, Gy, pp)

    ct2 = np.log(100 + grad_norm2)
    ctp = np.log(100 + grad_normp)

    ct2 = grad_norm2
    ctp = grad_normp

    from skimage import filters

    edge_roberts = filters.roberts(ct2)
    edge_sobel = filters.sobel(ct2)

    from skimage import feature
    canny = feature.canny(img_gray)

    #ct[ct<=1.063]=0

    from skimage.morphology import skeletonize
    #skeleton = skeletonize(ct2)

    plt.subplot(121), plt.imshow(ct2, cmap='PiYG')
    plt.subplot(122), plt.imshow(edge_sobel , cmap='gray')
    #plt.subplot(122), plt.imshow(skeleton, cmap='gray')
    plt.show()


    #plt.imshow(skeleton, cmap='gray')#/np.max(abs(fft2)))
    #plt.imshow(ct, cmap='jet')
    #plt.colorbar()
    #plt.show()

    return ct2
