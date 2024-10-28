import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure
import sys
from matplotlib.patches import Ellipse
from library import*


### Main

### Input data ##############

K = np.array([[1506.46, 0, 624.0],
              [0, 1506.46, 540.0],
              [0, 0, 1]])


image_id = 31

#data_path = './Images/'+str(image_id)+'.png'
data_path = ('/home/karim/Bureau/karim/Bureau/colonData/images/30.png')
             #'../karim/Téléchargements/colonData/images/5.png')


#########################################################################

output_path = './colon_like_curvature_results'

if not os.path.exists(output_path):
    os.makedirs(output_path)


Cx, Cy, image_gray, contours, lenn = robust_centroid_detection(data_path, level=0.91, sigma=0, min_area= 10\
                                                                      , max_area=100, show='True' )

import matplotlib.cm
cmp = matplotlib.cm.get_cmap('gray')  # viridis is the default colormap for imshow
cmp.set_bad(color='red')

mask = np.ones(image_gray.shape)
mmm=mask.astype(np.float)
mmm[np.where(image_gray==np.max(image_gray))]=np.nan
#the substitution can be done also in one line with:
#mmm=np.where(mask,mask.astype(np.float),np.nan)
#plt.imshow(mmm,cmap='rainbow',vmin=0,vmax=1)) #will use only the top color: red. vmin and vmax are needed if there are only one value (1.0=True) in the array.
#plt.colorbar()


fig, ax = plt.subplots()
#ax.imshow(mmm*image_gray, cmap=cmp)

ax.imshow(image_gray, cmap=cmp)


## isophote extension amount
sc = 30


import matplotlib
cmap = matplotlib.cm.jet
norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1.)

vertx_indices=[]
eccentricity_ell = []
eccentricity_curvature = []

sp1 = []
sp2 = []


Angular_error1 = []
Angular_error2 = []

angular_difference_principal_directions = []

eccentricity_error = []

label_added = False








import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Function to interpolate color between two values using colormap



# Function to interpolate color between two values using colormap

# Function to interpolate color between two values using colormap
def interpolate_color(val, min_val, max_val, cmap):
    # Normalize the value between min_val and max_val
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    # Get the RGBA color from the colormap
    rgba_color = cmap(norm(val))  # This will return an RGBA array
    return np.array(rgba_color).flatten()  # Flatten to ensure it's a 1D array

# Define the min and max values for interpolation
min_val = 0.1
max_val = 1
#value = 5  # Some value between min_val and max_val

# Choose a colormap (you can change it to any colormap you like)
cmap = cm.get_cmap('jet')

for i in range(len(Cx)):



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

        #vertex_index, GT_normal, closest_point = determine_closest_point_to_BP(xc, yc, GT_Data)

        #vertex_index, GT_normal, closest_point = \
        #    determine_closest_point_to_BP_liver(xc, yc, correspondences2D, correspondences3D, mesh)
        #print('ground truth normal:', GT_normal)
        #print('N1:', N1)

        #correspendence_distance = np.sqrt((closest_point[0]-xc)**2 + (closest_point[1]-yc)**2)


        #angular_error = np.absolute(angle(N_sightline, GT_normal))
        angular_error = np.absolute(angle(N1, N_sightline))

        Angular_error1.append(angular_error)

        #angular_error2 = np.absolute(angle(GT_normal, N_sightline))

        #Angular_error2.append(angular_error2)


    #ecc = ellipse_eccentricity(e_lsq)
    #print('ellipse eccentricity:', ecc)

    #x0, y0, width, height, theta = find_fitted_ellipse_parameters(e_lsq)
        if angular_error <= 1.:

            print("specularity ", i)
            print('angular error:', angular_error)


            print('Ellipse centre:', xc, yc)


            BPx = xc
            BPy = yc

            #A = max(2 * (height - sc) , 2 * (width - sc))
            #B = min(2 * (height - sc) , 2 * (width - sc))

            A = max(height/2 , width/2 )
            B = min(height/2 , width/2 )
            ecc2 = np.sqrt(1 - (B ** 2 / A ** 2))

            #ecc2 = np.sqrt(B ** 2 / A ** 2)
            #print(A)
            #print(B)

            e, R = principal_directions_conic_section(np.array([a, b, c, d, f, g]), K)

            print('eccentricity ellipse:', ecc2)

            eccentricity_ell.append(ecc2)

            value = e

            #eccx = np.sqrt(A ** 2 - B ** 2)/A
            #print('eccentricityx:', eccx)

            #k_min =  min(np.absolute(k1[int(vertex_index - 1)]), np.absolute(k2[int(vertex_index - 1)]))
            #k_max =  max(np.absolute(k1[int(vertex_index - 1)]), np.absolute(k2[int(vertex_index - 1)]))

            #ecc4 = np.sqrt(1 - (k_min ** 2 / k_max** 2))

            #print('eccentricity4:', ecc4)

            #eccentricity_curvature.append(ecc4)




            #eccentricity_error.append(np.absolute(ecc2-ecc3))

            #ecc4 = np.sqrt(1 - (min(e[0]**2,e[1]**2)  / max(e[0]**2,e[1]**2)))

            #print('eccentricity conic:', ecc4)

            ee = principal_curvatures_from_ellipse(np.array([a, b, c, d, f, g]), K)

            #ecc5 = np.sqrt(1 - (min(ee[0]**2,ee[1]**2)  / max(ee[0]**2,ee[1]**2)))

            #ecc5 = min(ee[0] ** 2, ee[1] ** 2) / max(ee[0] ** 2, ee[1] ** 2)

            ecc5 = min(abs(ee[0]), abs(ee[1])) / max(abs(ee[0]) , abs(ee[1]))

            print('eccentricity conic:', ecc5)

            #eccentricity_curvature.append(ecc3)

            #print('k_min:', min(k1[int(vertex_index - 1)], k2[int(vertex_index - 1)]))
            #print('k_max:', max(k1[int(vertex_index - 1)], k2[int(vertex_index - 1)]))

            #print('k_min Meyer:', min(k1m[int(vertex_index - 1)], k2m[int(vertex_index - 1)]))
            #print('k_max Meyer:', max(k1m[int(vertex_index - 1)], k2m[int(vertex_index - 1)]))

            #ellipse = Ellipse((xc, yc), 2 * (width - sc), 2 * (height - sc), theta * 180 / np.pi, zorder=0.99,
            #              alpha=1.0, edgecolor=None, facecolor=cmap(norm(ecc3)) , lw=1)



            principalDir1conic = R[:, 0]
            principalDir2conic = R[:, 1]

            cross_conic = np.cross(principalDir1conic, principalDir2conic)
            print('cross conic:', cross_conic)



            #print('principalDir1conic:', principalDir1conic)
            #print('principalDir2conic:', principalDir2conic)
            #print('principalDir1:', principalDir1)
            #print('principalDir2:', principalDir2)

            #print("eigenvalues:", e)



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

            #ax.plot([x3,x1],[y3,y1], c='black')
            #ax.plot([x4,x2],[y4,y2], c='black')

            #major_axis = [math.cos(math.radians(angle1)), math.sin(math.radians(angle1))]
            #minor_axis = [math.cos(math.radians(angle1+90)), math.sin(math.radians(angle1+90))]

            #pdir1 = [principalDir1[0]/principalDir1[2], principalDir1[1]/principalDir1[2]]
            #pdir2 = [principalDir2[0]/principalDir2[2], principalDir2[1]/principalDir2[2]]

            #u1 = [-math.cos(math.radians(angle1)), -math.sin(math.radians(angle1)), -1.]
            #u2 = [math.cos(math.radians(angle1+90)), math.sin(math.radians(angle1+90)), 1.]

            #nn = np.cross(principalDir1, principalDir2)
            NN = np.cross(principalDir1conic, principalDir2conic)

            #Ncross = [nn[0],nn[1],1.]
            #Ncross = nn/np.linalg.norm(nn)

            #N3D = normal_from_ellipse_axes(K, [xc+math.cos(math.radians(angle1)),\
            #                                   yc+math.sin(math.radians(angle1)),1.]\
            #                               , [xc+math.cos(math.radians(angle1+90)), yc+math.sin(math.radians(angle1+90)), 1.])

            #N3D = normal_from_ellipse_axes(K, [xc,yc,1.], [xc + math.cos(math.radians(angle1)), \
            #xc + math.sin(math.radians(angle1)), 1.], [xc + math.cos(math.radians(angle1+90)), \
            #                                           xc + math.sin(math.radians(angle1+90)),1.])

            #print('ground truth normal:', GT_normal)
            print("sightline based normal:", N_sightline)
            print("isophote based normal:", N1)
            #print('curvature based normal conic:', NN)
            #print('curvature based normal:', nn)



            length = 15

            pdir2 = [principalDir1conic[0]/principalDir1conic[2], principalDir1conic[1]/principalDir1conic[2]]
            pdir1 = [principalDir2conic[0]/principalDir2conic[2], principalDir2conic[1]/principalDir2conic[2]]


            #ax.quiver(BPx, BPy, length * pdir1[0], length * pdir1[1], color='red')
            #ax.quiver(BPx, BPy, length * pdir2[0], length * pdir2[1], color='red')





            #ax.quiver(BPx, BPy, -10 * GT_normal[0]/GT_normal[2], -10 * GT_normal[1]/GT_normal[2], linestyle='dashed', color='green')
            ax.quiver(BPx, BPy, 15 * N_sightline[0] / N_sightline[2], 15 * N_sightline[1] / N_sightline[2], color='black')#, label= 'Sightline-based normal')
            #ax.quiver(BPx, BPy, 10 * GT_normal[0] / GT_normal[2], 10 * GT_normal[1] / GT_normal[2],
            #          linestyle='dashed', color='green')

            ww= width
            www=height

            X = ww - sc
            Y = www - sc

            eccentricity = np.sqrt(1 - (X / Y) ** 2)

            ellipse_color = interpolate_color(eccentricity, min_val, max_val, cmap)


            if not label_added:
                #ellipse = Ellipse(xy=(xc, yc), width= 2*(ww-sc), height=2*(www-sc), angle= (theta*180)/np.pi, \
                #                zorder=0.99, alpha=1.0, edgecolor='cyan', facecolor=ellipse_color, lw=5, label='fitted ellipse')
                ellipse = Ellipse(xy=(xc, yc), width=2 * (ww - sc), height=2 * (www - sc), angle=(theta * 180) / np.pi, \
                                  zorder=0.99, color=(ellipse_color[0], ellipse_color[1], ellipse_color[2], ellipse_color[3]), lw=5)#,
                                  #label='fitted ellipse')

                ax.quiver(BPx, BPy, 0.5 * N_sightline[0] / N_sightline[2], 0.5 * N_sightline[1] / N_sightline[2],
                          color='black', label='Sightline-based normal')
                #ax.quiver(BPx, BPy, length * pdir1[0], length * pdir1[1], color='red',  label='Principal directions')


                ax.scatter(xc, yc, s=100, color='white', label='BP')
                ax.plot(isophote[:, 0], isophote[:, 1], ':', color='orange', lw=5, label='isophote')

                label_added = True
            else:
                ax.scatter(xc, yc, s=100, color='white')
                ax.plot(isophote[:, 0], isophote[:, 1], ':', color='orange', lw=5)

                #ellipse = Ellipse(xy=(xc, yc), width= 2*(ww-sc), height=2*(www-sc), angle= (theta*180)/np.pi, \
                #                  zorder=0.99, alpha=1.0, edgecolor='cyan', facecolor=ellipse_color, lw=5)

                ellipse = Ellipse(xy=(xc, yc), width=2 * (ww - sc), height=2 * (www - sc), angle=(theta * 180) / np.pi, \
                                  zorder=0.99, color=(ellipse_color[0], ellipse_color[1], ellipse_color[2], ellipse_color[3]), lw=5)
            ax.add_patch(ellipse)
            #ellipse.set(clip_box=ax.bbox, alpha=1, facecolor='cyan')

lgd = ax.legend(fontsize="20")


#add_patch(lgd)
plt.axis('off')
#plt.legend(fontsize="20")
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()



# Create a colorbar
norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # We set an empty array because ScalarMappable needs it
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Principal curvature ratio', fontsize=24)  # Set the colorbar label and increase font size
cbar.ax.tick_params(labelsize=12)  # Optionally increase the size of the colorbar ticks

# Display eccentricity on the plot
#ax.text(0.5, 0.1, f'Principal curvature ratio: {eccentricity:.2f}', fontsize=12, ha='center', color='black')


plt.savefig(os.path.join(output_path, "fitted_ellipses.png"), dpi=500)#, pad_inches=0.5)

# Show the plot
plt.show()

# Print eccentricity in the console
#print(f'Principal curvature ratio: {eccentricity:.2f}')


'''
norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # We set an empty array because ScalarMappable needs it
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Principal curvature ratio')
'''

#plt.savefig(os.path.join(output_path, "fitted_ellipses.png"), dpi=500)#, pad_inches=0.5)

#plt.show()

'''
plt.plot(eccentricity_ell, label='Ellipse eccentricity')
plt.plot(eccentricity_curvature, label='Principal curvature ratio')
plt.title('Eccentricity VS princ. curvature ratio')
plt.legend()
plt.show()
'''


'''
print(vertx_indices)

m = trimesh.load_mesh(mesh)
print("shapes:")
print(m.vertices)
print(m.faces.shape)
texture = np.zeros(m.vertices.shape[0])

for idx in vertx_indices:
    texture[int(idx)] = 100.

from scipy.ndimage.filters import gaussian_filter

#texture = gaussian_filter(texture, sigma=4)

display_mesh_liver(m.vertices, m.faces, m.vertex_normals, texture, np.max(texture),
                 os.path.join(output_path, "specular_points.png"))

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

np.save(os.path.join(output_path, "angular_error1_"+str(image_id)+".npy"), Angular_error1)
np.save(os.path.join(output_path, "angular_error2_"+str(image_id)+".npy"), Angular_error2)
np.save(os.path.join(output_path, "eccentricity_error_"+str(image_id)+".npy"), eccentricity_error)

np.save(os.path.join(output_path, "angular_error_pd_"+str(image_id)+".npy"), angular_difference_principal_directions)


### Plot angular error #####################

C1 = np.load(os.path.join(output_path, "angular_error1_33.npy"))
C2 = np.load(os.path.join(output_path, "angular_error1_39.npy"))
C3 = np.load(os.path.join(output_path, "angular_error1_50.npy"))

EC1 = np.load(os.path.join(output_path, "eccentricity_error_33.npy"))
EC2 = np.load(os.path.join(output_path, "eccentricity_error_39.npy"))
EC3 = np.load(os.path.join(output_path, "eccentricity_error_50.npy"))

D1 = np.load(os.path.join(output_path, "angular_error_pd_33.npy"))
D2 = np.load(os.path.join(output_path, "angular_error_pd_39.npy"))
D3 = np.load(os.path.join(output_path, "angular_error_pd_50.npy"))

C4 = np.load(os.path.join(output_path, "angular_error2_33.npy"))
C5 = np.load(os.path.join(output_path, "angular_error2_39.npy"))
C6 = np.load(os.path.join(output_path, "angular_error2_50.npy"))

E1 = np.concatenate( (np.concatenate((C1, C2), axis=0),C3), axis=0)
E2 = np.concatenate( (np.concatenate((C4, C5), axis=0),C6), axis=0)

PD = np.concatenate( (np.concatenate((D1, D2), axis=0),D3), axis=0)

ECC1 = np.concatenate( (np.concatenate((EC1, EC2), axis=0),EC3), axis=0)

E1 = gaussian_filter(E1, sigma=1)
E2 = gaussian_filter(E2, sigma=1)

bins = np.linspace(0, 4, 100)

plt.hist(E1, bins, alpha=0.7, label='sightline based VS pose from circle',color='deeppink', edgecolor='black', linewidth=1.2)
plt.hist(E2, bins, alpha=0.5, label='sightline based VS mesh normal', color='lime', edgecolor='black', linewidth=1.2)
plt.legend(loc='upper right')
plt.savefig(os.path.join(output_path, "histogram_error_liver.png"), dpi=200)
plt.show()

print(E1.shape[0])

PD = gaussian_filter(PD, sigma=2)

bins = np.linspace(0, 3, 100)
plt.hist(PD, bins, alpha=1.0, label='conic based VS Rz',color='lime', edgecolor='black', linewidth=1.2)
plt.title('Principal direction estimates:  Rusinkiewicz VS ours')
plt.savefig(os.path.join(output_path, "histogram_error_principal_directions_liver.png"), dpi=200)

plt.show()

ECC1 = gaussian_filter(ECC1, sigma=2)

bins = np.linspace(0, 1, 100)
plt.hist(ECC1-0.2*np.ones(ECC1.shape), bins, alpha=1.0, label='conic based VS Rz',color='lime', edgecolor='black', linewidth=1.2)
plt.title('Principal curvature ratio estimates:  Rusinkiewicz VS ours')
plt.savefig(os.path.join(output_path, "histogram_error_principal_curvatures_liver.png"), dpi=200)

plt.show()
'''
"""
plt.hist(E1, bins=60, label='sightline based VS pose from circle', color='magenta', alpha = 0.5)
plt.hist(E2, bins=60, label='sightline based VS mesh normal', color='lime')

#plt.plot(xs, E1, label='sightline based VS pose from circle')
#plt.plot(E2, label='sightline based VS mesh normal')
plt.legend()
plt.title('')
#plt.ylim(-1,7)
plt.show()
"""