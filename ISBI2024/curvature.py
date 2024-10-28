import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure
import sys
from matplotlib.patches import Ellipse
from library import*


### Main

### Input data ##############
K = np.array([[1111.11, 0.00000000e+00, 2000.0],
            [0.00000000e+00, 1111.11, 2000.0],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

image_id = 33

data_path = './Images/'+str(image_id)+'.png'
GT_Data = text_file_to_array('./Text_files/'+str(image_id)+'.txt')
mesh = './Meshes/'+str(image_id)+'.ply'
#######################################################################

## Read computed principal curvatures
k1 = np.load('./Medima_tools-main/Specular_curvature_results/'+str(image_id)+'/k1.npy')
k2 = np.load('./Medima_tools-main/Specular_curvature_results/'+str(image_id)+'/k2.npy')

PrincipalDir1 = np.load('./Medima_tools-main/Specular_curvature_results/'+str(image_id)+'/PrincipalDir1.npy')
PrincipalDir2 = np.load('./Medima_tools-main/Specular_curvature_results/'+str(image_id)+'/PrincipalDir2.npy')

print(PrincipalDir1.shape)

#if image_id == 33:
#    K_M = np.load('./Medima_tools-main/Specular_curvature_results/'+str(image_id)+'/mean_curvature_Meyer.npy')
#    K_G = np.load('./Medima_tools-main/Specular_curvature_results/'+str(image_id)+'/Gauss_curvature_Meyer.npy')

#    k2m, k1m = principal_curvatures(K_M, K_G)

#########################################################################

output_path = './colon_like_curvature_results'

if not os.path.exists(output_path):
    os.makedirs(output_path)


Cx, Cy, image_gray, contours, lenn = robust_centroid_detection(data_path, level=0.90, sigma=1, min_area= 2\
                                                                      , max_area=2000, show='True' )



fig, ax = plt.subplots()
ax.imshow(image_gray, cmap=plt.cm.gray)

## isophote extension amount
sc = 100


import matplotlib
cmap = matplotlib.cm.jet
norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1.)

vertx_indices=[]
eccentricity_ell = []
eccentricity_curvature = []

sp1 = []
sp2 = []


for i in range(len(Cx)):

    print("specularity ", i)

    # N = normal_estimation_direct(K, Cx[i], Cy[i])

    isophote = contours[i]
    Swap(isophote, 0, 1)
    extended_isophote = extend_contour(isophote, scale=sc)



    #e_lsq = fit_ellipse_lstsq(extended_isophote)

    ell = EllipseModel()
    ell.estimate(extended_isophote)

    if ell.params is not None:

        a, b, c, d, f, g, xc, yc, width, height, theta = ell.params

        N_sightline = normal_estimation_direct(K, xc, yc)
        #print("sightline based normal:", N_sightline)

        N1, N2 = Zisserman_method(np.array([a, b, c, d, f, g]), K, case='synthetic')
        #print("isophote based normal:", N1)

        vertex_index, GT_normal, closest_point = determine_closest_point_to_BP(xc, yc, GT_Data)
        #print('ground truth normal:', GT_normal)
        #print('N1:', N1)



        angular_error = np.absolute(angle(N_sightline, N1))
        #angular_error = np.absolute(angle(N1, GT_normal))
        print('angular error:', angular_error)

    #ecc = ellipse_eccentricity(e_lsq)
    #print('ellipse eccentricity:', ecc)

    #x0, y0, width, height, theta = find_fitted_ellipse_parameters(e_lsq)
        if angular_error <= 1.:

            #ax.plot(isophote[:,0], isophote[:,1], color='chartreuse')

            #ax.scatter(xc,yc,s=6)

            vertx_indices.append(vertex_index)
            BPx = xc
            BPy = yc

            e, R = principal_directions_conic_section(np.array([a, b, c, d, f, g]), K)

            A = max(2 * (height - sc) , 2 * (width - sc))
            B = min(2 * (height - sc) , 2 * (width - sc))
            ecc2 = np.sqrt(1 - (B ** 2 / A ** 2))
            print(A)
            print(B)
            print('eccentricity2:', ecc2)

            eccentricity_ell.append(ecc2)

            #eccx = np.sqrt(A ** 2 - B ** 2)/A
            #print('eccentricityx:', eccx)

            k_min =  min(np.absolute(k1[int(vertex_index - 1)]), np.absolute(k2[int(vertex_index - 1)]))
            k_max =  max(np.absolute(k1[int(vertex_index - 1)]), np.absolute(k2[int(vertex_index - 1)]))

            ecc4 = np.sqrt(1 - (k_min ** 2 / k_max** 2))

            #print('eccentricity4:', ecc4)

            #eccentricity_curvature.append(ecc4)


            ecc3 = np.sqrt(1 - (k2[int(vertex_index - 1)] ** 2 / k1[int(vertex_index - 1)] ** 2))

            #print('eccentricity3:', ecc3)

            ee = principal_curvatures_from_ellipse(np.array([a, b, c, d, f, g]), K)

            ecc5 = np.sqrt(1 - (min(ee[0]**2,ee[1]**2)  / max(ee[0]**2,ee[1]**2)))

            #print('eccentricity conic:', ecc5)

            eccentricity_curvature.append(ecc3)

            #print('k_min:', min(k1[int(vertex_index - 1)], k2[int(vertex_index - 1)]))
            #print('k_max:', max(k1[int(vertex_index - 1)], k2[int(vertex_index - 1)]))

            #print('k_min Meyer:', min(k1m[int(vertex_index - 1)], k2m[int(vertex_index - 1)]))
            #print('k_max Meyer:', max(k1m[int(vertex_index - 1)], k2m[int(vertex_index - 1)]))

            #ellipse = Ellipse((xc, yc), 2 * (width - sc), 2 * (height - sc), theta * 180 / np.pi, zorder=0.99,
            #              alpha=1.0, edgecolor=None, facecolor=cmap(norm(ecc3)) , lw=1)
            ellipse = Ellipse((xc, yc), 2 * (width - sc), 2 * (height - sc), theta * 180 / np.pi, zorder=0.99,
                              alpha=0.5, edgecolor=None, facecolor='chartreuse', lw=1)
            ax.add_patch(ellipse)

            principalDir1 = PrincipalDir1[int(vertex_index - 1),:]
            principalDir2 = PrincipalDir2[int(vertex_index - 1),:]


            principalDir1conic = R[:, 0]
            principalDir2conic = R[:, 1]

            print(np.linalg.norm(principalDir1))
            print(np.linalg.norm(principalDir2))

            sp1.append(scalar_products(principalDir1conic, principalDir1, principalDir2)[0])
            sp2.append(scalar_products(principalDir1conic, principalDir1, principalDir2)[1])

            print('principalDir1conic:', principalDir1conic)
            print('principalDir2conic:', principalDir2conic)
            print('principalDir1:', principalDir1)
            print('principalDir2:', principalDir2)

            print(np.dot(principalDir1conic, principalDir1))
            print(np.dot(principalDir1conic, principalDir2))

            print(np.dot(principalDir2conic, principalDir1))
            print(np.dot(principalDir2conic, principalDir2))

            #print("eigenvalues:", e)

            #print('N1:', N1)

            print('ground truth normal:', GT_normal)
            print("Estimated normal:", np.array(N_sightline))
            print("Estimated normal conic:", np.array(N1))
            #print("Recheck normal:", np.linalg.inv(R) @ np.array(N_sightline))



            kappa1, kappa2 = retrieve_principal_curvatures(K, R, xc, yc, k1[int(vertex_index - 1)], k2[int(vertex_index - 1)])

            print("kappa1, kappa2:", kappa1, kappa2)

            print("k1, k2:", k1[int(vertex_index - 1)], k2[int(vertex_index - 1)])
            rotate_isophote(extended_isophote[:, 0], extended_isophote[:, 1], R, xc, yc)




            #print("scalar product1:", np.dot(principalDir1, GT_normal))
            #print("scalar product2:", np.dot(principalDir2, GT_normal))

            ## compute perspective projection and plot of principal directions

            #### determine ellipse major and minor axes and plotting them

            rmajor = max(2 * (width - sc), 2 * (height - sc)) / 2
            angle1 = theta * 180 / np.pi
            if angle1 > 90:
                angle1 = angle1 - 90
            else:
                angle1 = angle1 + 90
            x1 = xc + math.cos(math.radians(angle1)) * rmajor
            y1 = yc + math.sin(math.radians(angle1)) * rmajor
            x2 = xc + math.cos(math.radians(angle1+90)) * rmajor
            y2 = yc + math.sin(math.radians(angle1+90)) * rmajor

            x3 = xc - math.cos(math.radians(angle1)) * rmajor
            y3 = yc - math.sin(math.radians(angle1)) * rmajor

            x4 = xc - math.cos(math.radians(angle1+90)) * rmajor
            y4 = yc - math.sin(math.radians(angle1+90)) * rmajor

            ax.plot([x3,x1],[y3,y1], c='black')
            ax.plot([x4,x2],[y4,y2], c='black')

            #major_axis = [math.cos(math.radians(angle1)), math.sin(math.radians(angle1))]
            #minor_axis = [math.cos(math.radians(angle1+90)), math.sin(math.radians(angle1+90))]

            pdir1 = [principalDir1[0]/principalDir1[2], principalDir1[1]/principalDir1[2]]
            pdir2 = [principalDir2[0]/principalDir2[2], principalDir2[1]/principalDir2[2]]

            #u1 = [-math.cos(math.radians(angle1)), -math.sin(math.radians(angle1)), -1.]
            #u2 = [math.cos(math.radians(angle1+90)), math.sin(math.radians(angle1+90)), 1.]

            nn = np.cross(principalDir1, principalDir2)
            NN = np.cross(principalDir1conic, principalDir2conic)

            #Ncross = [nn[0],nn[1],1.]
            #Ncross = nn/np.linalg.norm(nn)

            #N3D = normal_from_ellipse_axes(K, [xc+math.cos(math.radians(angle1)),\
            #                                   yc+math.sin(math.radians(angle1)),1.]\
            #                               , [xc+math.cos(math.radians(angle1+90)), yc+math.sin(math.radians(angle1+90)), 1.])

            #N3D = normal_from_ellipse_axes(K, [xc,yc,1.], [xc + math.cos(math.radians(angle1)), \
            #xc + math.sin(math.radians(angle1)), 1.], [xc + math.cos(math.radians(angle1+90)), \
            #                                           xc + math.sin(math.radians(angle1+90)),1.])

            print('ground truth normal:', GT_normal)
            print("sightline based normal:", N_sightline)
            print("isophote based normal:", N1)
            print('curvature based normal conic:', NN)
            print('curvature based normal:', nn)



            length = 15

            ax.quiver(BPx, BPy, length * pdir1[0], length * pdir1[1], color='red')
            ax.quiver(BPx, BPy, length * pdir2[0], length * pdir2[1], color='red')




            #ax.quiver(BPx, BPy, -10 * GT_normal[0]/GT_normal[2], -10 * GT_normal[1]/GT_normal[2], linestyle='dashed', color='green')
            #ax.quiver(BPx, BPy, -length * N_sightline[0] / N_sightline[2], -length * N_sightline[1] / N_sightline[2], color='blue')
            ax.quiver(BPx, BPy, 10 * GT_normal[0] / GT_normal[2], 10 * GT_normal[1] / GT_normal[2],
                      linestyle='dashed', color='green')




plt.axis('off')


#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()

plt.savefig(os.path.join(output_path, "fitted_ellipses.png"), dpi=500)#, pad_inches=0.5)
plt.show()

print(vertx_indices)

m = trimesh.load_mesh(mesh)
print("shapes:")
print(m.vertices.shape)
print(m.faces.shape)
texture = np.zeros(m.vertices.shape[0])

for idx in vertx_indices:
    texture[int(idx-1)] = 100.

from scipy.ndimage.filters import gaussian_filter

texture = gaussian_filter(texture, sigma=4)

#display_mesh(m.vertices, m.faces, m.vertex_normals, texture, np.max(texture),
#                 os.path.join(output_path, "specular_points.png"))

#### Results
plt.plot(eccentricity_ell, label='Ellipse eccentricity')
plt.plot(eccentricity_curvature, label='Principal curvature ratio')
plt.title('Eccentricity VS princ. curvature ratio')
plt.legend()
plt.show()

plt.plot(sp1, label='scalar product 1')
plt.plot(sp2, label='scalar product 2')
plt.title('estimated (ellipse based) vs  ground truth principal directions')
plt.legend()
plt.show()
