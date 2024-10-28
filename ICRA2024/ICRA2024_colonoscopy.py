import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure
import glob, re
from pathlib import Path
import os
import xlsxwriter
import math
import sys

##### Inputs: ###################################

#K = np.array([[9.5206967601396957e+02, 0, 8.8447392341747241e+02],
#              [0, 9.5206967601396957e+02, 5.5368748726315528e+02],
#              [0, 0, 1]])

K = np.array([[533.3333, 0., 960.0],
              [0., 533.3333,  540.0],
              [0., 0., 1.]])


data_path = '/home/karim/Téléchargements/colonData/Blender_simulation/SpecularImages/'

#data_path = '/home/karim/Téléchargements/colonData/SpecularData/'+sequence_name+'/'
###################################################

dist_tolerence = 2.0
###################################################
def smooth_contour(contour, nb_points=1000, smooth_factor=0.0):
    from scipy.interpolate import splprep, splev

    tck, u = splprep(contour.T, u=None, s=smooth_factor, per=1, k=3)
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

def robust_centroid_detection_colonoscopy(img_path, show = 'False'):

    from skimage.filters import difference_of_gaussians
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    #diff_img0 = difference_of_gaussians(image, 1, high_sigma=2)
    #diff_img1 = difference_of_gaussians(image, 1, high_sigma=3)
    diff_img2 = difference_of_gaussians(image, 1, high_sigma=9)

    binary = np.exp(diff_img2) > 1.25 #0.15

    contours = measure.find_contours(binary, 0)

    from scipy.ndimage.filters import gaussian_filter
    image = gaussian_filter(image, sigma=2)
    image = image/np.max(image)
    contours = measure.find_contours(image, 0.9)

    useful_contours = []
    Cx = []
    Cy = []
    for i, contour in enumerate(contours):
        # contour = smooth_contour(contour, 1000, 0.0)

        if np.logical_and(contour.shape[0] <= 45, contour.shape[0] >= 4):
                contour = smooth_contour(contour, 1000, 0.0)
                useful_contours.append(contour)

                Cx.append(np.median(contour[:, 1]))
                Cy.append(np.median(contour[:, 0]))

    #plt.figure(figsize=(8, 7))
    #plt.subplot(2, 2, 1)
    #plt.imshow(image, cmap=plt.cm.gray)
    #plt.title('Original')
    #plt.axis('off')

    #plt.subplot(2, 2, 2)
    #plt.title('Global Threshold')
    #plt.imshow(np.exp(diff_img2), cmap=plt.cm.jet)
    #plt.axis('off')

    #plt.subplot(2, 2, 3)
    #plt.imshow(binary, cmap=plt.cm.gray)
    #plt.title('Niblack Threshold')
    #plt.axis('off')

    #plt.subplot(2, 2, 4)
    #plt.imshow(diff_img2, cmap=plt.cm.jet)
    #plt.title('Sauvola Threshold')
    #plt.axis('off')

    #plt.show()

    if show == 'True':

        plt.imshow(image, cmap='gray')
        for i, contour in enumerate(useful_contours):
            #if contour.shape[0] >= 4:
            #    contour = smooth_contour(contour, 1000, 0.0)
            plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)

            plt.scatter(Cx[i], Cy[i])

        plt.show()

    return Cx, Cy, image, useful_contours, len(contours)


def robust_centroid_detection(img_path, level=0.9,  min_area= 0, max_area=10000, show='False'):

    image_gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

    im = image_gray / np.max(image_gray)

    contours = measure.find_contours(im, level*np.max(im))

    #print('total number of contours:', len(contours))

    Cx = []
    Cy = []

    useful_contours = []

    for i, contour in enumerate(contours):

        contour = smooth_contour(contour, 1000, 0.0)
        ar = np.absolute(area(contour))

        #print("contour area:", ar)
        if np.logical_and(ar < max_area, ar> min_area):
            useful_contours.append(contour)
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




def normal_estimation_direct(K, cx, cy):

    direct_normal = np.linalg.inv(K) @ np.array([cx, cy, 1.])
    N = direct_normal/np.linalg.norm(direct_normal)

    return N

def ellipse_centroid(e_coeff):
    a = e_coeff[0]
    b = e_coeff[1]
    c = e_coeff[2]
    d = e_coeff[3]
    f = e_coeff[4]
    g = e_coeff[5]

    b /= 2.
    d /= 2.
    f /= 2.

    x0 = (c * d - b * f) / (b ** 2. - a * c)
    y0 = (a * f - b * d) / (b ** 2. - a * c)

    return x0, y0



def distance_between_2_points(p1x, p1y, p2x, p2y):

    return np.sqrt((p1x-p2x)**2 + (p1y-p2y)**2)


def fitting_residuals(e_coeff, contour):

    a = e_coeff[0]
    b = e_coeff[1]
    c = e_coeff[2]
    d = e_coeff[3]
    e = e_coeff[4]
    f = e_coeff[5]

    residuals = 0

    for i in range(contour.shape[0]):
        X = contour[i,0]
        Y = contour[i,1]
        residuals += a * X**2 + b * (X*Y) + c * Y**2 + d*X + e*Y + f
    return np.absolute(residuals)

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

def fit_ellipse_lstsq(contour):
    x = contour[:, 0].flatten()
    y = contour[:, 1].flatten()

    A = np.array([x ** 2, x * y, y ** 2, x, y, x * 0 + 1])
    ATA = np.dot(A, A.T)

    u, s, vh = np.linalg.svd(ATA, hermitian=True)

    return u[:, u.shape[1] - 1]


def plot_fitting_res(X, Y, e):


    # equation: 7.91x^2 + -0.213xy + 5.46y^2 -0.031x -0.0896y = 1
    eqn = e[0] * X**2 + e[1]*X*Y + e[2]*Y**2 + e[3]*X + e[4]*Y
    Z = -e[5]

    return eqn, [Z]

def Swap(arr, start_index, last_index):
    arr[:, [start_index, last_index]] = arr[:, [last_index, start_index]]

def normal_disambiguiting(N1, N2, I):
    dot1 = N1[0] * I[0] + N1[1] * I[1] + N1[2] * I[2]
    dot2 = N2[0] * I[0] + N2[1] * I[1] + N2[2] * I[2]

    #print("dot product 1:", dot1)
    #print("dot product 2:", dot2)

    disambiguited_normal = [0, 0, 1]

    if (1 - np.absolute(dot1)) <= (1 - np.absolute(dot2)):
        #print("The normal is N1")
        disambiguited_normal = N1
    else:
        #print("The normal is N2")
        disambiguited_normal = N2

    return disambiguited_normal

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.degrees( math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))) )


def blob_area_statistics(imageSet, level=0.9):
    areas = []

    for t in range(len(imageSet)):
        image_gray = cv2.cvtColor(cv2.imread(imageSet[t]), cv2.COLOR_BGR2GRAY)
        im = image_gray / np.max(image_gray)

        contours = measure.find_contours(im, level * np.max(im))

        for i, contour in enumerate(contours):
            contour = smooth_contour(contour, 1000, 0.0)
            ar = np.absolute(area(contour))
            areas.append(ar)

    return np.max(areas), np.min(areas), np.median(areas), np.mean(areas)

def plot_error(error, save_path, title='error_curve.png', distance='False'):
    from matplotlib.ticker import FormatStrFormatter
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.title(title, {'color': 'k', 'fontsize': 20})
    plt.xlabel('specularity index', {'color': 'k', 'fontsize': 20})
    plt.ylabel('angular error (degrees)', {'color': 'k', 'fontsize': 20})
    plt.grid()

    if distance == 'True':
        #plt.yscale('symlog')
        plt.ylabel('distance (pixels)', {'color': 'k', 'fontsize': 20})

    #plt.ylim(-np.max(error), np.max(error))

    plt.plot(error, 'm')
    plt.savefig(save_path, dpi=150)
    plt.show()



### Main ##########

imageSet = sorted(glob.glob(data_path + '*.png'), \
                                     key=lambda x: float(re.findall("(\d+)", x)[-1]))

maskSet = sorted(glob.glob(data_path+'outKarim/' + '*.png'), \
                                     key=lambda x: float(re.findall("(\d+)", x)[0]))

#Cx, Cy, image_gray, useful_contours, lenght = robust_centroid_detection_colonoscopy(imageSet[0], maskSet[0], show='True' )
#Cx, Cy, image_gray, useful_contours, lenght = robust_centroid_detection_colonoscopy(imageSet[0], show = 'True')


## create output path in which centroids and normals will be stored as xlsx files
path1 = Path(data_path)
output_path = os.path.join(path1.parent.absolute(), 'laparoscopy_output_normals/')
print(output_path)

if not os.path.exists(output_path):
       os.makedirs(output_path)

display_fitting = 'True'

if display_fitting == 'True':
    image_gray = cv2.cvtColor(cv2.imread(imageSet[0]), cv2.COLOR_BGR2GRAY)
    x = np.linspace(0, image_gray.shape[0], image_gray.shape[0])
    y = np.linspace(0, image_gray.shape[1], image_gray.shape[1])
    X, Y = np.meshgrid(x, y)

error =[]

distance_centroids =[]

total_specularities = 0
#used_specularities = 0

'''
maax, miin, mediaan, meaan = blob_area_statistics(imageSet, level=0.9)

print(miin, maax, mediaan, meaan)

for t in range(112, 120):
    Cx, Cy, image_gray, useful_contours, lenn = robust_centroid_detection(imageSet[t], level=0.9,  min_area= miin, \
                                                                     max_area=maax/2, show='True')

'''



for t in range(0,len(imageSet)): #(134, 143):

    print('image ', t+1)
    prefix = imageSet[t].split('/')[-1].split('.')[0]

    workbook = xlsxwriter.Workbook(output_path + prefix + '.xlsx')

    worksheet = workbook.add_worksheet()

    worksheet.write('A1', 'Px')
    worksheet.write('B1', 'Py')
    worksheet.write('C1', 'Nx')
    worksheet.write('D1', 'Ny')
    worksheet.write('E1', 'Nz')
    worksheet.write('F1', 'Npfcx')
    worksheet.write('G1', 'Npfcy')
    worksheet.write('H1', 'Npfcz')


    Cx, Cy, image_gray, contours, len_contours = robust_centroid_detection_colonoscopy(imageSet[t], show='True')
    #Cx, Cy, image_gray, contours, len_contours = robust_centroid_detection(imageSet[t], level=0.94, min_area=0, max_area=10000, show='False')
    total_specularities += len_contours

    if display_fitting == 'True':

        plt.imshow(image_gray, cmap='gray')

    for i in range(len(Cx)):


        N = normal_estimation_direct(K, Cx[i], Cy[i])

        isophote = contours[i]
        Swap(isophote, 0, 1)
        extended_isophote = extend_contour(isophote, scale=5)
        e_lsq = fit_ellipse_lstsq(extended_isophote)


        x0, y0 = ellipse_centroid(e_lsq)

        print('contour centroid:')
        print(Cx[i], Cy[i])

        print('ellipse centroid:')

        print(x0, y0)

        ecc = ellipse_eccentricity(e_lsq)
        print('ellipse eccentricity:', ecc)

        residus = fitting_residuals(e_lsq, extended_isophote)

        print('fitting residuals:', residus)



        dist_centroids = distance_between_2_points(Cx[i], Cy[i], x0, y0)



        N1, N2 = Zisserman_method(e_lsq, K, case='synthetic')

        Nd = normal_disambiguiting(N1, N2, N)

        #print("Direct normal:", N)
        #print("Disambiguited normal:", Nd)

        angular_error = np.absolute(angle(N, Nd))


        if np.absolute(angular_error - 180.0) <= 3.0:
            angular_error = 180 - angular_error

        if np.logical_and(np.logical_and(np.absolute(angular_error) <= 1.0, ecc<=0.95),dist_centroids <= dist_tolerence):#residus <= 1e-08):

            if display_fitting == 'True':
                eqn, [Z] = plot_fitting_res(X, Y, e_lsq)
                plt.contour(X, Y, eqn, [Z], color='red')
                #plt.scatter(x0, y0)

            error.append(angular_error)
            distance_centroids.append(dist_centroids)
            #used_specularities += 1

            worksheet.write(i + 1, 0, Cx[i])
            worksheet.write(i + 1, 1, Cy[i])

            #print("angle between methods:", angular_error)

            worksheet.write(i+1, 2, N[0])
            worksheet.write(i+1, 3, N[1])
            worksheet.write(i+1, 4, N[2])

            worksheet.write(i+1, 5, Nd[0])
            worksheet.write(i+1, 6, Nd[1])
            worksheet.write(i+1, 7, Nd[2])

    workbook.close()

    if display_fitting == 'True':

        plt.show()


plot_error(error, os.path.join(path1.parent.absolute(), 'colonoscopy.png'), title='angular difference (direct VS ours)')
plot_error(distance_centroids, os.path.join(path1.parent.absolute(), 'colonoscopy_distance_centroids.png'), title='Euclidean distance between centroids ', distance='True')

print("ratio of used specularities:", len(error)/total_specularities)
