import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from skimage import measure
import math
#import visvis as vv
import trimesh
import os
#from mayavi import mlab
import pyvista as pv
from scipy.ndimage.filters import gaussian_filter
#import skfmm

from scipy import ndimage


def compute_specularity_ratio(dilated_specular_mask, foreground):
    x = len(np.where(dilated_specular_mask != 0)[0])
    y = len(np.where(foreground != 0)[0])

    return x / y


def plot_error_bar(x,mean,std, title='error bar', savepath='./Error_bar.png'):

    fig = plt.figure()

    plt.errorbar(x, mean, std, linestyle='None', marker='o')
    plt.xlabel('distance from outer boundary (pixels)', fontsize=10)
    plt.ylabel('angular error (degrees)', fontsize=10)
    plt.legend()
    plt.title(title, fontsize=15)
    plt.savefig(savepath, dpi=200)

    #plt.show()


def plot_error_bars(x, mean2, std2, mean3, std3, title='Error bar', savepath='./Error_bar.png'):

    fig = plt.figure()

    #plt.errorbar(x, mean1, std1, linestyle='None', marker='o', label = 'Local planarity assumption')
    eb1 = plt.errorbar(x, mean2, std2, linestyle='None', marker='^', label='Image inpainting')
    eb2 = plt.errorbar(x, mean3, std3, linestyle='None', marker='o', label='Ours')
    eb2[-1][0].set_linestyle('--')
    plt.xlabel('distance from outer boundary (pixels)', fontsize=10)
    plt.ylabel('angular error (degrees)', fontsize=10)
    plt.legend()
    plt.title(title, fontsize=15)
    plt.savefig(savepath, dpi=200)

def compute_rmse_error(u1, u2, specular_mask, title='error', show = 'False'):

    #error = np.zeros(u1.shape)

    error = np.sqrt(np.absolute(u1**2-u2**2))
   # error = error*specular_mask

    R = np.where(specular_mask!=0)


    if show == 'True':

        plt.imshow( error*specular_mask)
        plt.colorbar()
        plt.title(title)
        plt.show()

    return np.min(error[R]), np.max(error[R]), np.mean(error[R]), np.std(error[R])

def dot_product(u1, u2, u3, v1, v2, v3, R):

    dot_prod = np.zeros(u1.shape)
    dot_prod[R] = u1[R] * v1[R] + u2[R] * v2[R] + u3[R] * v3[R]

    return dot_prod[R]

def normalize_vectors(u1,u2,u3,R):

    norm = np.sqrt(u1[R] * u1[R] + u2[R] * u2[R] + u3[R] * u3[R])

    u1[R] =  u1[R] / norm #np.sqrt(u1[R] * u1[R] + u2[R] * u2[R] + u3[R] * u3[R])
    u2[R] = u2[R] / norm #np.sqrt(u1[R] * u1[R] + u2[R] * u2[R] + u3[R] * u3[R])
    u3[R] = u3[R] / norm #np.sqrt(u1[R] * u1[R] + u2[R] * u2[R] + u3[R] * u3[R])

    return u1, u2, u3





def compute_angular_error(u1, u2, u3, v1, v2, v3, specular_mask, title='Angular error', show = 'False'):

    dot_prod = np.zeros(u1.shape)
    acos = np.zeros(u1.shape)
    theta = np.zeros(u1.shape)

    R = np.where(specular_mask != 0)

    dot_prod[R] = dot_product(u1, u2, u3, v1, v2, v3, R)  #u1[R]*v1[R] + u2[R]*v2[R] + u3[R]*v3[R]

    acos[R] = np.acos(dot_prod[R])

    #error = np.sqrt(np.absolute(u1**2-u2**2))
   # error = error*specular_mask

    theta[R] = np.degrees( acos[R] ) # / (len_u[R] * len_v[R]) )

    ang_error = np.absolute(theta)



    if show == 'True':

        plt.imshow(ang_error)#error*specular_mask)
        plt.colorbar()
        plt.title(title)
        plt.show()
    return np.min(ang_error[R]), np.max(ang_error[R]), np.mean(ang_error[R]), np.std(ang_error[R])



    #return np.min(error[R]), np.max(error[R]), np.mean(error[R]), np.std(error[R])


def plot_with_contours(image, contours, title='title'):
    plt.imshow(image, cmap='gray')
    for i, contour in enumerate(contours):
        # contour = smooth_contour(contour, 1000, 0.0)
        plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)
    plt.title(title)
    plt.colorbar()
    plt.show()


def display_2_images_side_by_side(image1, image2, title1='image1', title2='image2'):

    size = image1.shape#[1000,1000]

    plt.close('all')

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image1, cmap='RdPu', origin='lower', extent=[-size[0] / 2., size[0] / 2., -size[1] / 2., size[1] / 2.])
    ax[1].imshow(image2, cmap='RdPu', origin='lower', extent=[-size[0] / 2., size[0] / 2., -size[1] / 2., size[1] / 2.])
    ax[0].set_title(title1, fontsize=10)
    ax[1].set_title(title2, fontsize=10)

    plt.show()



def iterative_relaxation(U_initial, R, initial_value=0., n_iters=2000):

    U_init = U_initial.copy()

    if initial_value != -1:

        U_init[R] = initial_value

    n_iter = n_iters

    for it in range(n_iter):
        U_init_n = U_init

        U_init[R[0], R[1]] = (U_init_n[R[0]+1, R[1]] + U_init_n[R[0]-1, R[1]] + U_init_n[R[0], R[1]+1] \
                               + U_init_n[R[0], R[1]-1]) / 4

    del U_init_n

    #print("Laplacian equation is solved")
    return U_init


def normal_estimation_direct(K, cx, cy):

    direct_normal = np.linalg.inv(K) @ np.array([cx, cy, 1.])
    N = direct_normal/np.linalg.norm(direct_normal)

    return N


def local_planarity_refining(Nx_init, Ny_init, Nz_init, K, spec_mask, show='False'):

    from skimage.measure import label

    completed_region_Nx = Nx_init.copy()
    completed_region_Ny = Ny_init.copy()
    completed_region_Nz = Nz_init.copy()

    labeled_image, count = label(spec_mask, return_num=True)

    #local_component = np.zeros(specular_mask.shape)

    for i in range(1,count+1):
        #local_component[labeled_image==i] = 1

        r = np.where(labeled_image==i)

        cx = np.median(r[0])
        cy = np.median(r[1])

        completed_region_Nx[r] = normal_estimation_direct(K, cx, cy)[0]
        completed_region_Ny[r] = normal_estimation_direct(K, cx, cy)[1]
        completed_region_Nz[r] = normal_estimation_direct(K, cx, cy)[2]

        #print('gt:', Nx[int(cx), int(cy)])
        #print('estimated:', normal_estimation_direct(K, cx, cy)[0])



        #plt.imshow(local_component, cmap='gray')
        #plt.scatter(cy,cx, s=6)
        #plt.show()

        #local_component = np.zeros(specular_mask.shape)

    #print(count)

    if show == 'True':

        plt.imshow(completed_region_Nx)
        plt.axis('off')
        plt.title("Nx filled with local planarity assumption")
        plt.show()

        plt.imshow(completed_region_Ny)
        plt.axis('off')
        plt.title("Ny filled with local planarity assumption")
        plt.show()

        plt.imshow(completed_region_Nz)
        plt.axis('off')
        plt.title("Nz filled with local planarity assumption")
        plt.show()

    return completed_region_Nx, completed_region_Ny, completed_region_Nz





def compute_boundaries_normal(Nx, Ny, Nz, specular_mask, Cx, Cy, K, show='False'):

    #inner_boundary = np.zeros(Nx.shape)

    Nx_init = Nx * (1-specular_mask)
    Ny_init = Ny * (1-specular_mask)
    Nz_init = Nz * (1-specular_mask)

    Omega = specular_mask.copy()



    #mask_dilation(mask, radius=0, Structure="cross")
    #inner = np.zeros(specular_mask.shape)


    for i, cx in enumerate(Cx):

        #print("ground truth:", Nx[int(Cx[i]), int(Cy[i])], Ny[int(Cx[i]), int(Cy[i])], Nz[int(Cx[i]), int(Cy[i])] )
        #print("Estimation:", normal_estimation_direct(K, Cx[i], Cy[i]))
        Nx_init[int(Cx[i]), int(Cy[i])] = normal_estimation_direct(K, Cx[i], Cy[i])[0]
        Ny_init[int(Cx[i]), int(Cy[i])] = normal_estimation_direct(K, Cx[i], Cy[i])[1]
        Nz_init[int(Cx[i]), int(Cy[i])] = normal_estimation_direct(K, Cx[i], Cy[i])[2]


        Omega[int(Cx[i]), int(Cy[i])]   = 0

    if show == 'True':

        plt.imshow(Nx_init, cmap='gray')
        plt.colorbar()
        plt.title('Nx_init')
        plt.show()

        plt.imshow(Omega, cmap='gray')
        plt.colorbar()
        plt.title('Omega')
        plt.show()

    return Nx_init, Ny_init, Nz_init, Omega

def threshold_based_masking(img, threshold = 70):
    from scipy import ndimage

    background = np.zeros(img.shape)
    background[img <= threshold] = 1

    foreground = ndimage.binary_fill_holes(1 - background).astype(int)

    return foreground, 1-foreground


def perfect_dilation(mask, radius=0,  show='False'):

    #mask[np.where(mask!=0)] = 1

    dilated_mask = np.zeros(mask.shape)
    phi_ext = ndimage.distance_transform_edt(1-mask)
    dilated_mask[phi_ext<=radius] = 1
    #phi_int = ndimage.distance_transform_edt(mask)
    if show == 'True':
        plt.imshow(dilated_mask + mask, cmap='gray')
        plt.title('Dilation result')
        plt.show()

    return dilated_mask, phi_ext #- phi_int
'''
def phi_v1(mask):

    phi_ext = skfmm.distance(np.max(mask)-mask)
    phi_int = skfmm.distance(mask)

    return  phi_ext - phi_int

def phi_narrow(mask, band=3):

    tmp = np.ones(mask.shape)
    tmp[mask!=0]= -1
    sgd = np.array(skfmm.distance(tmp, narrow=band), float)
    sgd[sgd==0] = band + 1

    return  sgd#np.array(skfmm.distance(tmp, narrow=band), float)
'''

def mask_dilation(mask, radius=0, Structure = "cross"):
    from scipy import ndimage
    if Structure == "square":
        struct = ndimage.generate_binary_structure(2, 2)
    else:
        struct = ndimage.generate_binary_structure(2, 1)
    dilated_mask = ndimage.binary_dilation(mask, structure=struct, iterations=radius).astype(mask.dtype)

    return dilated_mask

def smooth_normal_vector_field(Nx, Ny, Nz, foreground, sigma=4):
    ### smooth normal vector field

    Nx = gaussian_filter(Nx * foreground, sigma=sigma)
    Ny = gaussian_filter(Ny * foreground, sigma=sigma)
    Nz = gaussian_filter(Nz * foreground, sigma=sigma)

    norm = np.sqrt(Nx**2+Ny**2+Nz**2)
    norm[norm==0]=1

    return Nx/norm, Ny/norm, Nz/norm



def smooth_contour(contour, nb_points=10000, smooth_factor=0.0):
    from scipy.interpolate import splprep, splev

    tck, u = splprep(contour.T, u=None, s=smooth_factor, per=1)  # , k=2)
    u_new = np.linspace(u.min(), u.max(), nb_points)
    x_new, y_new = splev(u_new, tck, der=0)

    return np.vstack((x_new, y_new)).T

def extend_contour(contour, scale=1):
    extended_contour = contour.copy()

    N = contour.shape[0]
    looped_contour = np.zeros((N + 2, 2))

    looped_contour[0, :] = contour[N - 1, :]
    looped_contour[N + 1, :] = contour[0, :]
    looped_contour[1:N + 1, :] = contour

    for i in range(1, N):
        dp = looped_contour[i + 1, :] - looped_contour[i - 1, :]

        Tangent = dp / (np.linalg.norm(dp) + sys.float_info.epsilon)

        nx = Tangent[1]
        ny = -Tangent[0]

        extended_contour[i, 0] += scale * nx
        extended_contour[i, 1] += scale * ny

    # extend the initial point

    extended_contour[0, 0] = 0.5 * (extended_contour[1, 0] + extended_contour[N - 1, 0])
    extended_contour[0, 1] = 0.5 * (extended_contour[1, 1] + extended_contour[N - 1, 1])

    return extended_contour

def area(vs):
    a = 0
    x0, y0 = vs[0]
    for [x1, y1] in vs[1:]:
        dx = x1 - x0
        dy = y1 - y0
        a += 0.5 * (y0 * dx - x0 * dy)
        x0 = x1
        y0 = y1
    return a


def retrieve_synthetic_specular_mask(im_path):
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2GRAY)
    # depth0 = np.load('./Depths/'+depthSet[30])

    from skimage import filters, measure
    dog_filtered = filters.difference_of_gaussians(im, low_sigma=2, high_sigma=40, mode='nearest')
    specular_mask = np.zeros(dog_filtered.shape)
    specular_mask[dog_filtered >= 0.14] = 1
    # contours = measure.find_contours(specular_mask, 0)

    return specular_mask

def solve_heat_equation(inner_boundary, outer_boundary, R, n_iters = 60):

    #R = np.where(inner_boundary+outer_boundary == 0)
    #R = np.where(specular_mask != 0)

    u = np.zeros(outer_boundary.shape)  #u(i)
    u_n = np.zeros(outer_boundary.shape) #u(i+1)

    u[np.where(outer_boundary != 0)] =  1

    n_iter = n_iters

    for it in range (n_iter):

        u_n = u

        u[R[0],R[1]]= (u_n[R[0]+1,R[1]] + u_n[R[0]-1,R[1]] +  u_n[R[0],R[1]+1] \
        + u_n[R[0],R[1]-1]) / 4


        del u_n
    return u

def solve_heat_equation2(inner_boundary, outer_boundary, R, n_iters = 60):

    #R = np.where(inner_boundary+outer_boundary == 0)
    #R = np.where(specular_mask != 0)

    u = np.zeros(outer_boundary.shape)  #u(i)
    u_n = np.zeros(outer_boundary.shape) #u(i+1)

    u = outer_boundary #[np.where(outer_boundary != 0)] =  1

    n_iter = n_iters

    for it in range (n_iter):

        u_n = u

        u[R[0],R[1]]= (u_n[R[0]+1,R[1]] + u_n[R[0]-1,R[1]] +  u_n[R[0],R[1]+1] \
        + u_n[R[0],R[1]-1]) / 4


        del u_n
    return u

def solve_Poisson_equation(Z, R):

    N_xx, N_yy = np.gradient(Z)

    grad_norm = np.sqrt(N_xx ** 2 + N_yy ** 2)

    grad_norm[np.where(grad_norm == 0)] = 1  ## to avoid dividing by zero

    # Normalization
    Nx = np.divide(N_xx, grad_norm)
    Ny = np.divide(N_yy, grad_norm)


    den = np.absolute(Nx) + np.absolute(Ny)

    den[np.where(den == 0)] = 1  ## to avoid dividing by zero

    L0 = np.zeros(Z.shape)
    L1 = np.zeros(Z.shape)

    for it in range(100):
        L0_n = L0
        L0[R[0], R[1]] = (1 + np.absolute(Nx[R[0], R[1]]) *\
                          L0_n[(R[0] - np.sign(Nx[R[0], R[1]])).astype(int), R[1]] \
                          + np.absolute(Ny[R[0], R[1]]) * L0_n[R[0], (R[1] - np.sign(Ny[R[0], R[1]])).astype(int)])  / den[R[0], R[1]]
    return L0


def determine_closest_point_to_BP(cx,cy,GT_Data):

    X_GT = GT_Data[:,6:8]
    from scipy import spatial
    BP = [cx,  cy]
    distance, index = spatial.KDTree(X_GT).query(BP)

    vertex_index = int(index+1)
    GT_normal = GT_Data[index,3:6]
    closest_point = X_GT[index,:]


    return vertex_index, GT_normal, closest_point


def determine_closest_point_to_BP2(cx,cy,mesh, points):
    m = trimesh.load_mesh(mesh)
    normals = m.vertex_normals

    #X_GT = GT_Data[:,6:8]
    from scipy import spatial
    BP = [cx,  cy]
    distance, index = spatial.KDTree(points).query(BP)

    vertex_index = int(index)
    GT_normal = normals[index,:]
    closest_point = points[index,:]

    return vertex_index, GT_normal, closest_point

def compute_centroid_mask(Cx, Cy, im_path, show='True'):
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2GRAY)
    mask_centroids = np.zeros(im.shape)
    for i, cx in enumerate(Cx):
        mask_centroids[int(np.round(Cx[i])), int(np.round(Cy[i]))] = i

    if show == 'True':
        plt.imshow(mask_centroids, cmap='gray')
        plt.show()
    return mask_centroids

def extend_contour(contour, scale=1):
    extended_contour = contour.copy()

    N = contour.shape[0]
    looped_contour = np.zeros((N + 2, 2))

    looped_contour[0, :] = contour[N - 1, :]
    looped_contour[N + 1, :] = contour[0, :]
    looped_contour[1:N + 1, :] = contour

    for i in range(1, N):
        dp = looped_contour[i + 1, :] - looped_contour[i - 1, :]

        Tangent = dp / (np.linalg.norm(dp) + sys.float_info.epsilon)

        nx = Tangent[1]
        ny = -Tangent[0]

        extended_contour[i, 0] += scale * nx
        extended_contour[i, 1] += scale * ny

    # extend the initial point

    extended_contour[0, 0] = 0.5 * (extended_contour[1, 0] + extended_contour[N - 1, 0])
    extended_contour[0, 1] = 0.5 * (extended_contour[1, 1] + extended_contour[N - 1, 1])

    return extended_contour



def fill_contour(contours, im_path, show='True'):
    from skimage.draw import polygon
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2GRAY)
    mask = np.zeros(im.shape)

    for i, contour in enumerate(contours):
        rr, cc = polygon(contour[:, 0], contour[:, 1], mask.shape)
        mask[rr, cc] = 1

    if show == 'True':

        plt.imshow(mask, cmap='gray')
        for i, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)
        plt.show()

    return mask

def robust_centroid_detection(img_path, level=0.9, sigma=3, min_area= 0, max_area=1000000, show='False' , ext_contour='False', ext_scale=1):

    image_gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    from scipy.ndimage.filters import gaussian_filter

    image_gray = gaussian_filter(image_gray, sigma=sigma)

    im = image_gray / np.max(image_gray)

    contours = measure.find_contours(im, level*np.max(im))

    #print('total number of contours:', len(contours))

    Cx = []
    Cy = []

    useful_contours = []

    for i, contour in enumerate(contours):

        contour = smooth_contour(contour, 2000, 1.0)
        ar = np.absolute(area(contour))

        if ext_contour == 'True':
            extended_contour = extend_contour(contour, scale=ext_scale)
        else:
            extended_contour = contour


        #print("contour area:", ar)
        if np.logical_and(ar < max_area, ar> min_area):
            useful_contours.append(extended_contour)
            Cx.append(np.median(contour[:, 1]))
            Cy.append(np.median(contour[:, 0]))

    if show == 'True':

        plt.imshow(im, cmap='gray')
        for i, contour in enumerate(useful_contours):
            #contour = smooth_contour(contour, 1000, 0.0)
            plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)
            plt.scatter(Cx[i], Cy[i])

        plt.show()

    print(len(contours), len(useful_contours))

    return Cx, Cy, image_gray, useful_contours, len(contours)

def fit_ellipse_lstsq(contour):
    x = contour[:, 0].flatten()
    y = contour[:, 1].flatten()

    A = np.array([x**2, x*y, y**2, x, y, np.ones_like(x)])
    ATA = np.dot(A, A.T)

    u, s, vh = np.linalg.svd(ATA, hermitian=True)

    return u[:, u.shape[1] - 1]

def find_fitted_ellipse_parameters(u):

    a = u[0]
    b = u[1]/2
    c = u[2]
    d = u[3]/2
    f = u[4]/2
    g = u[5]

    # finding center of ellipse
    x0 = (c * d - b * f) / (b ** 2. - a * c)
    y0 = (a * f - b * d) / (b ** 2. - a * c)

    # Find the semi-axes lengths
    numerator = a * f ** 2 + c * d ** 2 + g * b ** 2 - 2 * b * d * f - a * c * g
    term = np.sqrt((a - c) ** 2 + 4 * b ** 2)
    denominator1 = (b ** 2 - a * c) * (term - (a + c))
    denominator2 = (b ** 2 - a * c) * (- term - (a + c))
    width = np.sqrt(2 * numerator / (denominator1 + sys.float_info.epsilon))
    height = np.sqrt(2 * numerator / (denominator2 + sys.float_info.epsilon))

    # angle of counterclockwise rotation of major-axis of ellipse
    # to x-axis [eqn. 23] from [2].
    phi = 0.5 * np.arctan((2.*b) / (a-c))
    if a > c:
        phi += 0.5 * np.pi

    return x0, y0, width, height, phi


def plot_fitting_res(X, Y, e):


    # equation: 7.91x^2 + -0.213xy + 5.46y^2 -0.031x -0.0896y = 1
    eqn = e[0] * X**2 + e[1]*X*Y + e[2]*Y**2 + e[3]*X + e[4]*Y
    Z = -e[5]

    return eqn, [Z]

def Swap(arr, start_index, last_index):
    arr[:, [start_index, last_index]] = arr[:, [last_index, start_index]]

def ellipse_eccentricity(e_coeff):
    a = e_coeff[0]
    b = e_coeff[1]
    c = e_coeff[2]
    d = e_coeff[3]
    e = e_coeff[4]
    f = e_coeff[5]

    E = np.array([[a, b / 2, d / 2],
                  [b / 2, c, e / 2],
                  [d / 2, e / 2, f]])

    nu = -np.sign(np.linalg.det(E))
    #print(nu)

    numerator = 2*np.sqrt((a-c)**2 + b**2)
    den = nu*(a+c) + np.sqrt((a-c)**2 + b**2)

    e = np.sqrt(numerator/den)

    return e
def _check_data_dim(data, dim):
    if data.ndim != 2 or data.shape[1] != dim:
        raise ValueError('Input data must have shape (N, %d).' % dim)


class BaseModel(object):

    def __init__(self):
        self.params = None


####################################################################################################################################
# Reference for improved ellipse fitting: http://andrewd.ces.clemson.edu/courses/cpsc482/papers/HF98_stableLeastSquaresEllipses.pdf
# We use this method for ellipse fitting which is a stable (improved) version of the method of Fitzgiborn (original formulation)
# For similarity measure, we use: https://pypi.org/project/similaritymeasures/
####################################################################################################################################


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
        width = np.sqrt(2 * numerator / (denominator1 + sys.float_info.epsilon))
        height = np.sqrt(2 * numerator / (denominator2 + sys.float_info.epsilon))

        # angle of counterclockwise rotation of major-axis of ellipse
        # to x-axis [eqn. 23] from [2].
        phi = 0.5 * np.arctan((2. * b) / (a - c))
        if a > c:
            phi += 0.5 * np.pi

        self.params = np.nan_to_num([A, B, C, D, F, G, x0, y0, width, height, phi]).tolist()
        self.params = [float(np.real(x)) for x in self.params]

        return True

def text_file_to_array(text_file):

    with open(text_file) as f:
        mylist = []
        lines = f.read().splitlines()

        for line in lines:
            mylist.append(line.split(","))

    arr = np.array(list(mylist), dtype='float32')

    #The text files are in the format: < X, Y, Z, Nx, Ny, Nz, px, py >. e.g. X = arr[:,0], Y = arr[:,1] etc...
    return arr



def determine_closest_point_to_BP_liver(cx,cy,correspondences2D, correspondences3D, mesh):

    m = trimesh.load_mesh(mesh)
    normals = m.vertex_normals

    X_GT = correspondences2D - np.ones(correspondences2D.shape)
    from scipy import spatial
    BP = [cx,  cy]
    distance, index = spatial.KDTree(X_GT).query(BP)

    #vertex_index = int(index+1)
    vertex_index = int(correspondences3D[index,0] - 1)

    #GT_normal = GT_Data[index,3:6]
    GT_normal = normals[vertex_index, :]

    closest_point = X_GT[index,:]


    return vertex_index, GT_normal, closest_point



def dual_conic(e_coeff, K):
    a = e_coeff[0]
    b = e_coeff[1]
    c = e_coeff[2]
    d = e_coeff[3]
    e = e_coeff[4]
    f = e_coeff[5]

    E = np.array([[a, b / 2, d / 2],
                  [b / 2, c, e / 2],
                  [d / 2, e / 2, f]])

    #print('C:', E)


    #K_norm = K / K[0, 0]

    #C = K_norm.T @ E @ K_norm



    C = K.T @ E @ K

    #print('C prime:', C@np.linalg.inv(C.T))

    return C


def sorted_eigen(A):

    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = eigenValues.argsort()#[::-1]
    eig_vals_sorted = eigenValues[idx]
    eig_vecs_sorted = eigenVectors[:,idx]

    #print('eigenvalues:', eig_vals_sorted)

    return eig_vals_sorted, eig_vecs_sorted


def plane_normal(R1, theta, case):
    R2 = np.array([[math.cos(theta), 0., math.sin(theta)],
                   [0., 1., 0.],
                   [-math.sin(theta), 0., math.cos(theta)]])

    #Rc = np.linalg.inv(np.dot(R1, R2)) #@ R2
    Rc = R1 @ R2
    if case =='synthetic':
        Rc = R1# @ R2

    n = -Rc[:, 2]

    #print("Rc:", Rc)

    return n, R2, Rc

def Zisserman_method(a, K, case='non-synthetic'):
    C = dual_conic(a, K)

    e1, R1 = sorted_eigen(C)  # the column R1[:,i] is the eigenvector corresponding to the eigenvalue e[i]

    thetay1 = math.atan(np.sqrt((e1[1] - e1[0]) / (e1[2] - e1[1] + sys.float_info.epsilon)))
    thetay2 = -math.atan(np.sqrt((e1[1] - e1[0]) / (e1[2] - e1[1] + sys.float_info.epsilon)))

    n1, R2_1, Rc1 = plane_normal(R1, thetay1, case= case)
    n2, R2_2, Rc2 = plane_normal(R1, thetay2, case = case)

    if case == 'synthetic':
        n1, R2_1, Rc1 = plane_normal(R1, thetay1, case = 'synthetic')
        n2, R2_2, Rc2 = plane_normal(R1, thetay2, case = 'synthetic')

    return n1, n2



def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

'''
def display_mesh(verts, faces, normals, texture, range, save_path):

    mesh = vv.mesh(verts, faces, normals, texture)
    f = vv.gca()
    mesh.colormap = vv.CM_JET
    f.axis.visible = False

    #f.bgcolor = None #1,1,1 #None
    #mesh.edgeShading = 'smooth'
    #mesh.clim = np.min(texture),np.max(texture)
    mesh.clim = 0.,range #-range,range  # -0.05,0.05#2

    vv.callLater(1.0, vv.screenshot, save_path, vv.gcf(), sf=2, bg='w')
    vv.colorbar()
    vv.view({'zoom': 0.3, 'azimuth': 0.0, 'elevation': -90.0})
    #vv.view({'zoom': 0.006, 'azimuth': -80.0, 'elevation': -5.0})
    vv.use().Run()

    return 0

def display_mesh_liver(verts, faces, normals, texture, range, save_path):

    mesh = vv.mesh(verts, faces, normals, texture)
    f = vv.gca()
    mesh.colormap = vv.CM_JET
    f.axis.visible = False

    #f.bgcolor = None #1,1,1 #None
    #mesh.edgeShading = 'smooth'
    #mesh.clim = np.min(texture),np.max(texture)
    mesh.clim = 0.,range #-range,range  # -0.05,0.05#2

    vv.callLater(1.0, vv.screenshot, save_path, vv.gcf(), sf=2, bg='w')
    vv.colorbar()
    vv.view({'zoom': 0.004, 'azimuth': 0.0, 'elevation': -90.0})
    #vv.view({'zoom': 0.006, 'azimuth': -80.0, 'elevation': -5.0})
    vv.use().Run()

    return 0

'''
def ellipse_eccentricity(e_coeff):
    a = e_coeff[0]
    b = e_coeff[1]
    c = e_coeff[2]
    d = e_coeff[3]
    e = e_coeff[4]
    f = e_coeff[5]

    E = np.array([[a, b / 2, d / 2],
                  [b / 2, c, e / 2],
                  [d / 2, e / 2, f]])

    nu = -np.sign(np.linalg.det(E))
    #print(nu)

    numerator = 2*np.sqrt((a-c)**2 + b**2)
    den = nu*(a+c) + np.sqrt((a-c)**2 + b**2)

    e = np.sqrt(numerator/den)

    return e


def principal_curvatures(K_M,K_G):

    tmp = np.sqrt(np.absolute(K_M**2- K_G))
    k1 = K_M  - tmp
    k2 = K_M  + tmp

    return k1, k2

def normal_from_ellipse_axes(K, p1, p2):
    K = np.array(K)
    R = np.eye(3)
    t = np.array([[0], [1], [0]])
    P = K.dot(np.hstack((R, t)))

    import scipy.linalg as lin

    x = np.array(p1)#[300, 300, 1])
    y = np.array(p2)
    u1 = np.dot(lin.pinv(P), x)
    u1 = u1/u1[3]
    u2 = np.dot(lin.pinv(P), y)
    u2 = u2/u2[3]

    N = np.cross([u1[0],u1[1], u1[2] ],[u2[0],u2[1], u2[2] ])

    return N#/np.linalg.norm(N)


def principal_directions_conic_section(a, K):
    C = dual_conic(a, K)

    e, R = sorted_eigen(C)  # the column R1[:,i] is the eigenvector corresponding to the eigenvalue e[i]

    return e, R #R[:, 0], R[:, 1]


def principal_curvatures_from_ellipse(e_coeff, K):

        a = e_coeff[0]
        b = e_coeff[1]
        c = e_coeff[2]
        d = e_coeff[3]
        e = e_coeff[4]
        f = e_coeff[5]

        E = np.array([[a, b / 2, d / 2],
                      [b / 2, c, e / 2],
                      [d / 2, e / 2, f]])

        # K_norm = K / K[0, 0]

        # C = K_norm.T @ E @ K_norm

        C = K.T @ E @ K
        C1 = np.array([[E[0,0], E[0,1]],
                      [E[0,1], E[1,1]]])

        e, R = sorted_eigen(C1)

        return e


def add_patch(legend):
    from matplotlib.patches import Patch
    ax = legend.axes

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor='red', edgecolor='r'))
    labels.append("Saturated area")

    legend._legend_box = None
    legend._init_legend_box(handles, labels)
    legend._set_loc(legend._loc)
    legend.set_title(legend.get_title().get_text())



def scalar_products(v1,v3,v4):

    sp1_3 = np.dot(v1,v3)
    sp1_4 = np.dot(v1,v4)

    miin = min(np.absolute(sp1_3), np.absolute(sp1_4))
    maax = max(np.absolute(sp1_3), np.absolute(sp1_4))



    return miin, maax



def load_dot_mat_file(path):

    import scipy.io

    mat = scipy.io.loadmat(path)
    correspondences2D = mat['correspondences2D']
    correspondences3D = mat['correspondences3D']

    return correspondences2D, correspondences3D

def retrieve_principal_curvatures(K, R, x0, y0, k1, k2):

    #x0 /= K[0,0]

    T = np.zeros((4,4))
    T[0:3,0:3] = R
    T[3,3] = 1

    print(R)
    print(T)

    #R1 = K.T@R@K
    #X = R1 @ np.array([x0, y0, 1.])

    X = np.linalg.inv(R) @ np.array([x0, y0, 0.])
    #X = R @ np.array([0, 0, 1.])

    X1 = np.linalg.inv(K)@ np.array([x0, y0, 1.])

    print('yalla;' ,X, X1)

    #X =  np.linalg.inv(R) @ np.array([x0, y0, 1.])
    #X = np.linalg.inv(R) @ np.array([x0, y0, 1.])
    #kappa1 = 1/K[0,0] - K[0,2]/X[0]
    #kappa2 = 1/K[1,1] - K[1,2]/X[1]

    a = (X[0])/K[0,0] #(X[0] - K[0,2]) #
    b = (X[1])/K[1,1]#(X[1] - K[1,2]) #

    print('estimated new x y:', [a, b])

    kappa1 = (x0 - K[0,2])/(K[0,0])
    kappa2 = (y0 - K[1,2])/(K[1,1])

    x_new = kappa1 / k1
    y_new = kappa2 / k2

    print('true new x y:', [x_new, y_new])

    return kappa1, kappa2

def rotate_isophote(X,Y,R,x0,y0):

    R1 = np.array([[R[0,0], R[0,1], 0],
                  [R[1,0], R[1,1], 0],
                  [R[2,0], R[2,1], 1]])

    X_new = np.zeros(X.shape)
    Y_new = np.zeros(Y.shape)

    X0_new = np.linalg.inv(R) @ np.array([x0, y0, 1.])

    for i in range(len(X)):

        mapped = np.linalg.inv(R) @ np.array([X[i], Y[i], 1.])

        X_new[i] = mapped[0] - (X0_new[0] -x0)
        Y_new[i] = mapped[1] - (X0_new[1] -y0)

    plt.scatter(X,Y, color='red', s=5)
    plt.scatter(X_new, Y_new, color='green', s=5)
    plt.show()





def check_normals(mesh, cx, cy, K, corresp):

    m = trimesh.load_mesh(mesh)
    GT_normals = m.vertex_normals
    estimated_normal = normal_estimation_direct(K, cx, cy)

    from scipy import spatial
    distance, index = spatial.KDTree(corresp).query([cx,cy])

    #vertex_index = int(index + 1)
    vertex_index = int(index)

    print("ground truth normal:", GT_normals[vertex_index])
    print("estimated normal:", estimated_normal)

    angular_error = angle(GT_normals[vertex_index], estimated_normal)

    print("angular error:", angular_error)

def determine_blobs(mask):
    blobs_labels = measure.label(mask, background=0)
    blobs = []

    for i in range(1,np.max(blobs_labels)):
        component = np.zeros(mask.shape)
        component[blobs_labels==i] = i
        blobs.append(component)
        del component
    return blobs

def determine_blob_centroid(blob):

    X, Y = np.where(blob.T != 0)
    cx = np.median(X)
    cy = np.median(Y)

    return cx, cy

def depth_to_normal(depth_map):
    zy, zx = np.gradient(depth_map)
    normal = np.dstack((-zx, -zy, np.ones_like(depth_map)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    return normal

def get_surface_normal_by_depth(depth, K=None):
    """
    depth: (h, w) of float, the unit of depth is meter
    K: (3, 3) of float, the depth camere's intrinsic
    """
    K = [[1, 0], [0, 1]] if K is None else K
    fx, fy = K[0,0], K[1,1]

    dz_dv, dz_du = np.gradient(depth)  # u, v mean the pixel coordinate in the image
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    # normalize to unit vector
    normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
    # set default normal to [0, 0, 1]
    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    return normal_unit

def plot_surface(z):

    mlab.figure(bgcolor=(1, 1, 1))
    x, y = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    mlab.surf(x, y, z, warp_scale='auto')

    mlab.show()

def plot_surface_with_texture(z, color_image, texture='depth'):

    if texture == 'depth':

        cmap = plt.get_cmap('jet')
        rgba_img = cmap(z)
        tex_arr = 255 * rgba_img.astype(np.uint8)
        tex = pv.numpy_to_texture(tex_arr)
    else:

        tex = pv.numpy_to_texture(color_image)

    x, y = np.mgrid[0:z.shape[0], 0:z.shape[1]]

    #scale = max(z.shape[0],z.shape[1])/2
    scale=1
    curvsurf = pv.StructuredGrid(x, y, scale*z)

    curvsurf.texture_map_to_plane(inplace=True)

    curvsurf.plot(texture=tex)

def crop_image(image, cx,cy, crop_size=200):

    return image[int(cy)-crop_size:int(cy)+crop_size, int(cx)-crop_size:int(cx)+crop_size]


def plot_depth_completion1(cropped_depth, cropped_depth_with_hole):
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cropped_depth, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(np.min(cropped_depth), np.max(cropped_depth))

    plt.subplot(1, 2, 2)
    plt.imshow(cropped_depth_with_hole, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(np.min(cropped_depth), np.max(cropped_depth))

    plt.show()

def plot_depth_completion2(cropped_depth, cropped_depth_with_hole, vmin=0.9, vmax=1, cmap='hsv', title1='title1', title2='title2'):
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cropped_depth, cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(vmin, vmax)
    plt.title(title1)

    plt.subplot(1, 2, 2)
    plt.imshow(cropped_depth_with_hole, cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(vmin, vmax)

    plt.title(title2)

    plt.show()
