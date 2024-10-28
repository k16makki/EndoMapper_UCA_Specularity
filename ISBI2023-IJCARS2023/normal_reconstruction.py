import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import math
import sys
from skimage import measure
import xlsxwriter
from skimage.filters import difference_of_gaussians
from scipy import ndimage
import os

import skfmm
from mylib import*

from PIL import Image
import diplib as dip
import os

# 24, 30, 36
#######################################Input parameters (user interaction) ########################
# path = '/home/karim/Téléchargements/colonData/images/evaluation/Kidney/Images/U79L_4_18244.png'#/home/karim/Téléchargements/colonData/images/30.png'
# cnn_mask_path = '/home/karim/Téléchargements/colonData/images/evaluation/Kidney/Network/U79L_4_18244.png'#/home/karim/Téléchargements/colonData/images/outKarim/30.png'

image_id = 150#170#1 #5 #53 # 26 # 4



lv = 0.9

#path = '/home/karim/Téléchargements/colonData/simulated_images/' + str(image_id) + '.png'
#cnn_mask_path = '/home/karim/Téléchargements/colonData/images/outKarim/' + str(image_id) + '.png'
#normal_path = './Normals_4_Agniva/'

path = '/home/karim/Téléchargements/real_pink_plane/final_ICCV_dataset/Images/image47.jpg'
normal_path = './Normals_4_Agniva/pinkplaneresults/'





# path = '/home/karim/Téléchargements/colonData/tool_images/tool1.png'
# cnn_mask_path = '/home/karim/Téléchargements/colonData/tool_images/predicted_tool1.png'
###################################################################################################

output_path = os.path.dirname(os.path.abspath(path)) + '/elliptical_masks'
if not os.path.exists(output_path):
    os.makedirs(output_path)

dirname = os.path.dirname(os.path.abspath(path))

isovalue = 0.91
threshold = 200
sigma = 0  # No smoothing

# Camera intrinsics


K = np.array([[3.21895067e+03, 0.00000000e+00, 1.54738621e+03],
 [0.00000000e+00, 3.19730708e+03, 2.08899263e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


####################################################################################################
def compute_specular_mask(image,threshold):
    mask = np.zeros(image.shape)
    mask[image>threshold] =1

    plt.imshow(mask, cmap='gray')
    plt.show()

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


def export_binary_mask(arr, f_name, list_of_ellipses, dpi=200, resize_fact=1, plt_show=False):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(arr.shape[1] / dpi, arr.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr, cmap='gray')

    for i in range(len(list_of_ellipses)):
        ellipse = Ellipse((list_of_ellipses[i][0], list_of_ellipses[i][1]), list_of_ellipses[i][2],
                          list_of_ellipses[i][3], list_of_ellipses[i][4], edgecolor='none',
                          facecolor='white', lw=1, linestyle='-')  # , alpha=1)
        ax.add_patch(ellipse)
    plt.savefig(f_name, dpi=(dpi * resize_fact))
    if plt_show:
        plt.show()
    else:
        plt.close()


def fit_ellipse_lstsq(contour):
    x = contour[:, 0].flatten()
    y = contour[:, 1].flatten()

    A = np.array([x ** 2, x * y, y ** 2, x, y, x * 0 + 1])
    ATA = np.dot(A, A.T)

    u, s, vh = np.linalg.svd(ATA, hermitian=True)

    return u[:, u.shape[1] - 1]


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




def residuals(xc, yc, a, b, theta, data):
    ctheta = math.cos(theta)
    stheta = math.sin(theta)

    x = data[:, 0]
    y = data[:, 1]

    N = data.shape[0]

    def fun(t, xi, yi):
        ct = math.cos(t)
        st = math.sin(t)
        xt = xc + a * ctheta * ct - b * stheta * st
        yt = yc + a * stheta * ct + b * ctheta * st
        return (xi - xt) ** 2 + (yi - yt) ** 2

    residuals = np.empty((N,), dtype=np.double)

    # initial guess for parameter t of closest point on ellipse
    t0 = np.arctan2(y - yc, x - xc) - theta

    # determine shortest distance to ellipse for each point
    for i in range(N):
        xi = x[i]
        yi = y[i]
        # faster without Dfun, because of the python overhead
        from scipy import optimize
        t, _ = optimize.leastsq(fun, t0[i], args=(xi, yi))
        residuals[i] = np.sqrt(fun(t, xi, yi))

    return residuals


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


    K_norm = K / K[0, 0]

    C = K_norm.T @ E @ K_norm

    #C = K.T @ E @ K

    return C


def sorted_eigen(A):

    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = eigenValues.argsort()#[::-1]
    eig_vals_sorted = eigenValues[idx]
    eig_vecs_sorted = eigenVectors[:,idx]


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

    print("Rc:", Rc)

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


def Swap(arr, start_index, last_index):
    arr[:, [start_index, last_index]] = arr[:, [last_index, start_index]]


def plot_contours(image, contours):
    fig, ax = plt.subplots()
    ax.imshow(image / np.max(image), cmap=plt.cm.gray)
    colors = plt.cm.jet(np.linspace(0, 1, len(contours)))

    for i, contour in enumerate(contours):
        #ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)



        cx = np.median(contour[:, 0])
        cy = np.median(contour[:, 1])

        ax.scatter(cy,cx)

    ax.axis('off')
    plt.show()


def plot_2contours(image, contours1, contours2):
    fig, ax = plt.subplots()
    ax.imshow(image / np.max(image), cmap=plt.cm.gray)
    # colors = plt.cm.jet(np.linspace(0, 1, len(contours1)))

    for i, contour in enumerate(contours1):
        ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2, label='region 1' if i == 0 else '')

    for i, contour in enumerate(contours2):
        ax.plot(contour[:, 1], contour[:, 0], color='green', linewidth=2, label='region 2' if i == 0 else '')

    plt.legend(fontsize='xx-large')

    ax.axis('off')

    plt.show()


def capture_specularities(path, threshold=200, sigma=1, isovalue=0.9):
    patch_image = cv2.imread(path)
    # patch_image= cv2.fastNlMeansDenoisingColored(patch_image, None, 11, 6, 7, 21)
    img_gray = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY)

    mask = np.zeros(img_gray.shape)
    mask[img_gray > threshold] = 1

    masked_image = (img_gray * mask) / np.max(img_gray * mask)

    masked_image = gaussian_filter(masked_image, sigma=sigma)
    masked_image /= np.max(masked_image)

    # contours = measure.find_contours(masked_image, isovalue)

    img_gray = gaussian_filter(img_gray.astype(np.uint), sigma=sigma)
    contours = measure.find_contours(img_gray / np.max(img_gray), isovalue)

    plt.subplot(121), plt.imshow(img_gray, cmap='gray')
    plt.subplot(122), plt.imshow(masked_image, cmap='gray')
    plt.show()

    return contours, img_gray


def plothistogram(img_gray, min=200, max=256, title='image histogram'):
    vals = img_gray.flatten()

    # plot histogram with 255 bins
    b, bins, patches = plt.hist(vals, 400, [min, max], color='orangered')
    # plt.xlim([70,255])
    plt.title(title)
    plt.show()


def robust_elliptic_contours(img_path, level=0.015):  # 0.029):

    image_color = cv2.imread(img_path)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # image_gray = ndimage.zoom(image_gray, 2)

    diff_img0 = difference_of_gaussians(image_gray, 2, high_sigma=4)
    diff_img = diff_img0 #difference_of_gaussians(diff_img0, 1, high_sigma=2)

    ref = np.log(1 + np.absolute(diff_img))
    # ref = np.absolute(diff_img)

    # plt.imshow(ref, cmap='hsv')
    # plt.title('difference of gaussians')
    # plt.show()
    contours = measure.find_contours(ref, level)

    im = image_gray / np.max(image_gray)


    plt.imshow(ref, cmap='gray')#'hsv')
    # plt.title('output')
    plt.show()

    return contours, image_gray

def robust_elliptic_contours_thresholding(img_path, level = 0.75):

    image_color = cv2.imread(img_path)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)


    im = image_gray / np.max(image_gray)
    im = im #gaussian_filter(im,1)
    im/=np.max(im)


    contours = measure.find_contours(im, level*np.max(im))


    plt.imshow(im, cmap='gray')
    # plt.title('output')
    plt.show()


    return contours, image_gray



def smooth_contour(contour, nb_points=1000, smooth_factor=0.0):
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


#

def determine_connected_components(blobs, min_blob_area=30, max_blob_area=35):
    blobs_labels = measure.label(blobs, background=0)

    # remove connected components

    free_connected_components = blobs_labels.copy()

    for i in range(1, np.max(blobs_labels) - 1):
        pixels = len(np.where(blobs_labels == i)[0])

        if np.logical_or(pixels >= max_blob_area, pixels <= min_blob_area):
            free_connected_components[free_connected_components == i] = 0

    # plt.figure(figsize=(9, 3.5))
    # plt.subplot(121)
    # plt.imshow(free_connected_components, cmap='nipy_spectral')
    # plt.axis('off')

    # plt.subplot(122)
    # plt.imshow(blobs_labels, cmap='nipy_spectral')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

    contours = measure.find_contours(blobs, 0.)
    free_connected_contours = measure.find_contours(free_connected_components, 0.)

    for i, contour in enumerate(free_connected_contours):

        if cv2.contourArea(contour.astype(int)) == 0.:  # cv2.isContourConvex(contour.astype(np.int)) is False:
            # print(i)
            del free_connected_contours[i]



    return blobs_labels, free_connected_contours, free_connected_components


def discard_hybrid_components(image_gray, cnn_mask):
    im = image_gray * (cnn_mask / np.max(cnn_mask))
    thresholds_im = threshold_multiotsu(im)
    regions = np.digitize(im, bins=thresholds_im)
    return True


def robust_elliptic_contours2(img_path, cnn_mask_path):  # level=0.029):

    image_color = cv2.imread(img_path)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # image_gray = gaussian_filter(image_gray, sigma=1)

    # image_gray = ndimage.zoom(image_gray, 2)

    diff_img0 = difference_of_gaussians(image_gray, 1, high_sigma=2)
    diff_img1 = difference_of_gaussians(image_gray, 1, high_sigma=3)

    # diff_img = difference_of_gaussians(diff_img0, 1, high_sigma=2)

    # ref = np.log(1 + np.absolute(diff_img))

    cnn_mask = Image.open(cnn_mask_path)  # )

    connected_components, contours, free_connected_components = determine_connected_components(
        cnn_mask / np.max(cnn_mask))
    # binary = img_gray.astype(float)
    # thresh2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)

    # image = ref * (cnn_mask / np.max(cnn_mask))

    im = image_gray * (cnn_mask / np.max(cnn_mask))

    from skimage.filters import threshold_multiotsu

    thresholds_im = threshold_multiotsu(im)

    print('Thresholds original image:', thresholds_im)

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(im, bins=thresholds_im)

    brightest_region = np.zeros(im.shape)
    brightest_region[regions == 2] = 1

    darkest_region = np.zeros(im.shape)
    darkest_region[regions == 1] = 1

    image = diff_img0 * darkest_region  # (cnn_mask / np.max(cnn_mask))
    image1 = diff_img1 * brightest_region  # (cnn_mask / np.max(cnn_mask))

    thresholds = threshold_multiotsu(image)
    thresholds1 = threshold_multiotsu(image1)

    print('Thresholds:', thresholds)
    print('Thresholds1:', thresholds1)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

    # Plotting the original image.
    ax[0].imshow(im, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    ax[1].hist(im.ravel(), bins=255)
    ax[1].set_title('Histogram')
    for thresh in thresholds_im:
        ax[1].axvline(thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(regions, cmap='jet')
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')

    plt.subplots_adjust()

    plt.show()

    contours = measure.find_contours(image, thresholds[1])
    contours1 = measure.find_contours(image1, thresholds1[0])

    return contours, contours1, image_gray


from shapely.geometry.point import Point
from shapely import affinity


def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr


def compute_ellipse_intersection_area(ell1, ell2):
    # elli = [cxi, cyi, ai, bi, thetai]

    ellipse1 = create_ellipse((ell1[0], ell1[1]), (ell1[2], ell1[3]), ell1[4])
    ellipse2 = create_ellipse((ell2[0], ell2[1]), (ell2[2], ell2[3]), ell2[4])

    intersect = ellipse1.intersection(ellipse2)

    print('area of intersect:', intersect.area)


def estimate_albedo_illumination(gray_img):
    # normalizing the image to have maximum of one
    E = gray_img / np.max(gray_img)
    # compute the average of the image brightness
    Mu1 = np.mean(E)
    # compute the average of the image brightness square
    Mu2 = np.mean(np.mean(np.power(E, 2)))
    # now lets compute the image's spatial gradient in x and y directions
    Ex, Ey = np.gradient(E)
    # normalize the gradients to be unit vectors
    Exy = np.sqrt(Ex ** 2 + Ey ** 2)
    nEx = np.divide(Ex, Exy + sys.float_info.epsilon)  # to avoid dividing by 0
    nEy = np.divide(Ey, Exy + sys.float_info.epsilon)
    # computing the average of the normalized gradients
    avgEx = np.mean(nEx)
    avgEy = np.mean(nEy)
    # now lets estimate the surface albedo

    gamma = np.sqrt((6 * (math.pi ** 2) * Mu2) - (48 * (Mu1 ** 2)))
    albedo = gamma / math.pi

    print("error source", (4 * Mu1) / (gamma+sys.float_info.epsilon))

    # estimating the slant
    slant = math.acos(min(1, (4 * Mu1) / gamma))

    # estimating the tilt
    tilt = math.atan(avgEy / avgEx)
    if tilt < 0:
        tilt = tilt + math.pi

    # the illumination direction will be ...
    I = [math.cos(tilt) * math.sin(slant), math.sin(tilt) * math.sin(slant), math.cos(slant)]

    return albedo, I, slant, tilt


def normal_disambiguiting(N1, N2, I):
    dot1 = N1[0] * I[0] + N1[1] * I[1] + N1[2] * I[2]
    dot2 = N2[0] * I[0] + N2[1] * I[1] + N2[2] * I[2]

    print("dot product 1:", dot1)
    print("dot product 2:", dot2)

    disambiguited_normal = [0, 0, 1]

    if (1 - np.absolute(dot1)) <= (1 - np.absolute(dot2)):
        print("The normal is N1")
        disambiguited_normal = N1
    else:
        print("The normal is N2")
        disambiguited_normal = N2

    return disambiguited_normal


def rotate_ellipse(contour, theta, xc, yc, sc):
    X = contour[:, 0] - xc
    Y = contour[:, 1] - yc

    theta = -theta

    isoxx = math.cos(theta) * X - math.sin(theta) * Y
    isoyy = math.sin(theta) * X + math.cos(theta) * Y

    aligned_contour = np.concatenate((isoxx.reshape(-1, 1), isoyy.reshape(-1, 1)), axis=1)
    ell_aligned = EllipseModel()
    ell_aligned.estimate(aligned_contour)

    fig, ax = plt.subplots()

    if ell_aligned.params is not None:
        a, b, c, d, f, g, xc, yc, width, height, thet = ell_aligned.params

        A = max(height - sc, width - sc)
        B = min(height - sc, width - sc)
        ex = np.sqrt(1 - (B ** 2 / A ** 2))
        # print('eccentricity aligned:', ex)

        # print("anal:", B ** 2 / A ** 2)

        # ellipse = Ellipse((xc, yc), 2 * width , 2 * height, thet * 180 / np.pi, edgecolor='magenta',facecolor='magenta', lw=3, linestyle='-', zorder=0.99, alpha=0.7)  # , label='fitted ellipse' if i == 0 else '')
        # ax.add_patch(ellipse)

    ax.scatter(X, Y, label='detected')
    ax.scatter(isoxx, isoyy, label='aligned')
    ax.legend()
    plt.show()


# compute_ellipse_intersection_area([0, 0, 2, 4, 10], [1, -1, 3, 2, 50])
########### Main ####################



#contours, img_gray = robust_elliptic_contours(path, level=0.029)
contours, img_gray = robust_elliptic_contours_thresholding(path , level = lv)
plot_contours(img_gray, contours)



image_color = cv2.imread(path)
img_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

plt.imshow(img_gray, cmap='gray')
plt.show()
compute_specular_mask(img_gray/np.max(img_gray),isovalue)

albedo, illumination_direction, slant, tilt = estimate_albedo_illumination(img_gray)
print("illumination direction:", illumination_direction)

# plt.imshow(img_gray, cmap='gray')
# plt.axis('off')
# plt.show()

mask = np.zeros(img_gray.shape)


from matplotlib.patches import Ellipse

colors = plt.cm.plasma(np.linspace(0, 1, len(contours)))

def plot_fitting_res(X, Y, e):


    # equation: 7.91x^2 + -0.213xy + 5.46y^2 -0.031x -0.0896y = 1
    eqn = e[0] * X**2 + e[1]*X*Y + e[2]*Y**2 + e[3]*X + e[4]*Y
    Z = -e[5]

    return eqn, [Z]




if not os.path.exists(normal_path):
    os.makedirs(normal_path)

workbook = xlsxwriter.Workbook(normal_path + 'normals_image' + f"{image_id:02}" + '.xlsx')

# The workbook object is then used to add new
# worksheet via the add_worksheet() method.
worksheet = workbook.add_worksheet()

worksheet.write('A1', 'Px')
worksheet.write('B1', 'Py')
worksheet.write('C1', 'N1x')
worksheet.write('D1', 'N1y')
worksheet.write('E1', 'N1z')
worksheet.write('F1', 'N2x')
worksheet.write('G1', 'N2y')
worksheet.write('H1', 'N2z')
worksheet.write('I1', 'Ndx')
worksheet.write('J1', 'Ndy')
worksheet.write('K1', 'Ndz')

worksheet.write('L1', 'Nx_dir')
worksheet.write('M1', 'Ny_dir')
worksheet.write('N1', 'Nz_dir')

import matplotlib as mpl
import matplotlib.cm as cm

norm = mpl.colors.Normalize(vmin=0, vmax=1)
m = cm.ScalarMappable(norm=norm, cmap=cm.jet)

fig, ax = plt.subplots()
ax.imshow(img_gray, cmap=plt.cm.gray)

j = 0

sc = 3.0

list_of_ellipses = []

x = np.linspace(0, img_gray.shape[0], img_gray.shape[0])
y = np.linspace(0, img_gray.shape[1], img_gray.shape[1])
X, Y = np.meshgrid(x, y)

for i, contour in enumerate(contours):

    print('i =', i)
    Swap(contour, 0, 1)
    first_contour = contour.copy()
    contour = smooth_contour(contour, 1000, 0.0)

    extended_contour = extend_contour(contour, scale=sc)

    ell = EllipseModel()
    ell.estimate(extended_contour)

    e0 = fit_ellipse_Fitzgibbon(extended_contour[:, 0], extended_contour[:, 1])[:, 0, 0]

    e_lsq = fit_ellipse_lstsq(extended_contour)

    if ell.params is not None:
        # ell.estimate(extended_contour)

        a, b, c, d, f, g, xc, yc, width, height, theta = ell.params

        print('coefficients:', [a, b, c, d, f, g])
        print('coefficients fitzgibbon:', e0)


        ellipse_area = math.pi * width * height

        # residus = np.round(abs(residuals(xc, yc, width, height, theta, extended_contour)), 5)

        residus = np.max(abs(residuals(xc, yc, width, height, theta, extended_contour)))

        # print('residuals:', residus)

        # if img_gray[int(yc), int(xc)] >= 0.5 * np.max(img_gray):
        # if ellipse_area <= 300:
        if residus <= 10.0 :

        #if np.logical_and(residus <= 1.0, ellipse_area <= 120):
            # if np.logical_and(ellipse_area >= 10, ellipse_area <= 40):
            # if residus <= 0.5:#41:#, ellipse_area <= 50):

            # rotate_ellipse(extended_contour, theta, xc, yc, sc)

            ar = np.absolute(area(contour))

            # print('blob area=', ar)

            j += 1

            # A = max(height-sc, width-sc)
            # B = min(height-sc, width-sc)

            A = max(height, width)
            B = min(height, width)
            ex = np.sqrt(1 - (B ** 2 / A ** 2))
            print('eccentricity:', ex)

            # ax.scatter(xc, yc, c=ex, cmap='jet', norm=norm)

            ###  0.5759636374732415

            # ellipse = Ellipse((xc, yc), 2*(width-sc), 2*(height-sc), theta*180 / np.pi, edgecolor=colors[i], facecolor=colors[i], lw=3, linestyle='-', \
            #                  zorder=0.99, alpha=0.7)#, label='fitted ellipse' if i == 0 else '')
            # ax.add_patch(ellipse)

            ellipse = Ellipse((xc, yc), 2 * (width - sc), 2 * (height - sc), theta * 180 / np.pi, zorder=0.99,
                              alpha=0.4, edgecolor='orangered', facecolor='orangered', lw=3)
            #ax.add_artist(ellipse)
            eqn, [Z] = plot_fitting_res(X, Y, np.array([a, b, c, d, f, g]))
            #eqn, [Z] = plot_fitting_res(X, Y, e_lsq)
            ax.contour(X, Y, eqn, [Z])


            # ellipse.set_facecolor(m.to_rgba(ex))
            # ellipse.set_edgecolor(m.to_rgba(ex))

            list_of_ellipses.append([xc, yc, 2 * (width - sc), 2 * (height - sc), theta * 180 / np.pi])

            ellipse_extended = Ellipse((xc, yc), 2 * width, 2 * height, theta * 180 / np.pi, edgecolor='orangered',
                                       facecolor='orangered', lw=3, linestyle='-', \
                                       alpha=0.4, label='fitted ellipse ex' if i == 0 else '')
            # ax.add_patch(ellipse_extended)

            # ax.add_artist(ellipse_extended)
            # ellipse_extended.set_facecolor(m.to_rgba(ex))
            # ellipse_extended.set_edgecolor(m.to_rgba(ex))
            ellipse_extended.set_facecolor('orangered')
            ellipse_extended.set_edgecolor('orangered')

            #N1, N2 = Zisserman_method(np.array([a, b, c, d, f, g]), K, case='synthetic')
            #N1, N2 = Zisserman_method(e0, K)
            N1, N2 = Zisserman_method(e_lsq, K, case='synthetic')


            #print("ok")
            #print(N1)
            #print(N11)


            # print('N1:', np.array(N1)/N1[2])
            # print('N2:', np.array(N2)/N2[2])

            #if N1[2] <= 0:
            #    N1 *= -1
            #if N2[2] <= 0:
            #    N2 *= -1

            Nd = normal_disambiguiting(N1, N2, illumination_direction)

            direct_normal = np.linalg.inv(K) @ np.array([xc, yc,1.])
            direct_normal = direct_normal/np.linalg.norm(direct_normal)

            print("direct normal:", direct_normal)

            # if Nd[2] <= 0:
            #    Nd *= -1

            nd = [Nd[0] / Nd[2], Nd[1] / Nd[2]]
            ndx = nd[0] / np.linalg.norm(nd)
            ndy = nd[1] / np.linalg.norm(nd)

            n1 = [N1[0] / N1[2], N1[1] / N1[2]]
            n1x = n1[0] / np.linalg.norm(n1)
            n1y = n1[1] / np.linalg.norm(n1)

            n2 = [N2[0] / N2[2], N2[1] / N2[2]]
            n2x = n2[0] / np.linalg.norm(n2)
            n2y = n2[1] / np.linalg.norm(n2)

            # ax.plot(first_contour[:,0], first_contour[:,1], color='red', linewidth=2, label='marching squares isophote' if i == 0 else '')
            # ax.plot(contour[:,0], contour[:,1], color='orange', linewidth=2, linestyle='--', label='spline interpolated isophote' if i == 0 else '')

            ax.plot(extended_contour[:, 0], extended_contour[:, 1], color='chartreuse', linewidth=2, linestyle='--',
                    label='extended isophote' if i == 0 else '')



            ax.quiver(xc, yc, -ndx, -ndy, angles='xy', scale_units='xy', color='red', width=3, scale=0.01,
                      units='xy', label='good normal' if i == 0 else '')

            ax.scatter(xc, yc, s=70)#, zorder=0.99)

            worksheet.write(j, 0, xc)
            worksheet.write(j, 1, yc)
            worksheet.write(j, 2, N1[0])
            worksheet.write(j, 3, N1[1])
            worksheet.write(j, 4, N1[2])

            worksheet.write(j, 5, N2[0])
            worksheet.write(j, 6, N2[1])
            worksheet.write(j, 7, N2[2])

            worksheet.write(j, 8, Nd[0])
            worksheet.write(j, 9, Nd[1])
            worksheet.write(j, 10, Nd[2])

            worksheet.write(j, 11, direct_normal[0])
            worksheet.write(j, 12, direct_normal[1])
            worksheet.write(j, 13, direct_normal[2])
        #else:

        #    intensity = free_connected_components[int(np.median(contour[:, 1])), int(np.median(contour[:, 0]))]
        #    print('intensity:', intensity)
        #    free_connected_components[free_connected_components == intensity] = 0

plt.axis('off')
#ax.legend(fontsize='xx-large')
# plt.legend( fontsize = 'xx-large')
smap = plt.cm.ScalarMappable(cmap='jet', norm=norm)
# cbar = fig.colorbar(smap, ax=ax, fraction=0.046, pad=0.04)
plt.show()

workbook.close()


