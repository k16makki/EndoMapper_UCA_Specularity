import glob, re

import matplotlib.pyplot as plt
from library import*
##pip install scikit-fmm --break-system-packages


### Input data ##############

images_path = './Images/' ## path to the folder containing all the image set
meshes_path = './Meshes/' ## path to the folder containing all the meshes
depth_path = './Depths/' ## path to the folder containing all the depths
normal_path = './Normals_gt/' ## path to the folder containing all the ground truth normals

K = np.array([[1111.11, 0., 2000.0],
              [0., 1111.11, 2000.0],
              [0., 0., 1.]])

Rt = np.array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.]])

H = K@Rt
##########################################################

imageSet = sorted(glob.glob(images_path + '*.png'), \
                                     key=lambda x: float(re.findall("(\d+)", x)[0]))
meshSet = sorted(glob.glob(meshes_path + '*.ply'), \
                                     key=lambda x: float(re.findall("(\d+)", x)[0]))

depthSet = sorted([filename for filename in os.listdir('./Depths/') if not filename.startswith('correspondences') \
                   and filename.endswith('npy')], key=lambda x: float(re.findall("(\d+)", x)[0]))

correspondSet = sorted([filename for filename in os.listdir('./Depths/') if filename.startswith('correspondences')], \
            key=lambda x: float(re.findall("(\d+)", x)[0]))

NxSet = sorted(glob.glob(normal_path+'Nx/' + '*.npy'), key=lambda x: float(re.findall("(\d+)", x)[0]))
NySet = sorted(glob.glob(normal_path+'Ny/' + '*.npy'), key=lambda x: float(re.findall("(\d+)", x)[0]))
NzSet = sorted(glob.glob(normal_path+'Nz/' + '*.npy'), key=lambda x: float(re.findall("(\d+)", x)[0]))

#print(imageSet)
#print(meshSet)
#print(depthSet)
#print(correspondSet)
#print(NxSet)
#print(NySet)
#print(NzSet)

image_idx = 30

Depth_image = np.load("./Depths/"+depthSet[image_idx])#cv2.cvtColor(cv2.imread(imageSet[image_idx]), cv2.COLOR_BGR2GRAY)
corresp = np.load("./Depths/"+correspondSet[image_idx])
Nx_gt = np.load(NySet[image_idx])
Ny_gt = np.load(NxSet[image_idx])
Nz_gt = np.load(NzSet[image_idx])


######################################################################################################################

Cx, Cy, image_gray, useful_contours, lenn =\
    robust_centroid_detection(imageSet[image_idx], level=0.8, sigma=3, min_area= 0, \
                              max_area=1000000, show='False', ext_contour='False', ext_scale=1)

#specular_mask = retrieve_synthetic_specular_mask(imageSet[30])

image = cv2.imread(imageSet[image_idx])
color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

foreground, background = threshold_based_masking(image_gray, threshold = 70)

Nx_gt, Ny_gt, Nz_gt = smooth_normal_vector_field(Nx_gt, Ny_gt, Nz_gt, foreground, sigma=2)


#plt.imshow(foreground, cmap='gray')
#plt.title('foreground')
#plt.colorbar()
#plt.show()

n_iters_Laplace = 3000

specular_mask = fill_contour(useful_contours, imageSet[image_idx], show='False')


## Initialize errors

ang_error_planar_assumption_min = []
ang_error_planar_assumption_max = []
ang_error_planar_assumption_mean = []
ang_error_planar_assumption_std = []
ang_error_inpainting_min = []
ang_error_inpainting_max = []
ang_error_inpainting_mean = []
ang_error_inpainting_std = []
ang_error_ours_min = []
ang_error_ours_max = []
ang_error_ours_mean = []
ang_error_ours_std = []

radii = []

spec_ratio = []





for i in range(0, 70, 3):


    dilated_specular_mask, distance_map = perfect_dilation(specular_mask, radius=i, show='False') #phi_Euclidean(specular_mask)

    spec_ratio.append(compute_specularity_ratio(dilated_specular_mask, foreground))

    #dilated_specular_mask = mask_dilation(specular_mask, radius=20)


    Nx_init, Ny_init, Nz_init, Omega = compute_boundaries_normal(Nx_gt, Ny_gt, Nz_gt, dilated_specular_mask, Cy, Cx, K,  show='False')
    #compute_boundaries(normals_from_depth[:,:,0], Cx, Cy, K, component='Nx')

    Nx_refined_plane, Ny_refined_plane, Nz_refined_plane = local_planarity_refining(Nx_init, Ny_init, Nz_init, K, dilated_specular_mask, show='False')

    R = np.where(Omega!=0)


    #Nx_refined_laplace = iterative_relaxation(Nx_init, R, initial_value=0, n_iters=n_iters_Laplace)
    #Ny_refined_laplace = iterative_relaxation(Ny_init, R, initial_value=0, n_iters=n_iters_Laplace)
    #Nz_refined_laplace = iterative_relaxation(Nz_init, R, initial_value=0, n_iters=n_iters_Laplace)

    Nx_refined_laplace = iterative_relaxation(Nx_refined_plane.copy(), R, initial_value=-1, n_iters=n_iters_Laplace)
    Ny_refined_laplace = iterative_relaxation(Ny_refined_plane.copy(), R, initial_value=-1, n_iters=n_iters_Laplace)
    Nz_refined_laplace = iterative_relaxation(Nz_refined_plane.copy(), R, initial_value=-1, n_iters=n_iters_Laplace)

    Nx_refined_laplace, Ny_refined_laplace, Nz_refined_laplace =  normalize_vectors(Nx_refined_laplace, Ny_refined_laplace, Nz_refined_laplace,R)

    #display_2_images_side_by_side(Nx_refined_laplace, Nx_gt, title1='Nx refined', title2='Nx_gt')

    #plot_with_contours(1000*(Nx_refined_laplace-Nx_gt), useful_contours, title='relative error')




    # Image_inpainting: https://github.com/aGIToz/PyInpaint

    print("inpainting")

    R_inpainting = np.where(dilated_specular_mask!=0)
    Nx_refined_inpainting = iterative_relaxation(Nx_gt, R_inpainting, initial_value=0., n_iters=n_iters_Laplace)
    Ny_refined_inpainting = iterative_relaxation(Ny_gt, R_inpainting, initial_value=0., n_iters=n_iters_Laplace)
    Nz_refined_inpainting = iterative_relaxation(Nz_gt, R_inpainting, initial_value=0., n_iters=n_iters_Laplace)

    Nx_refined_inpainting, Ny_refined_inpainting, Nz_refined_inpainting =  \
        normalize_vectors(Nx_refined_inpainting, Ny_refined_inpainting, Nz_refined_inpainting,R_inpainting)



    #plot_with_contours(Nx_refined_laplace, useful_contours, title='Nx refined with Laplace')
    #plot_with_contours(Nx_refined_plane, useful_contours, title='local_planarity')
    #plot_with_contours(Nx_refined_inpainting, useful_contours, title='inpainting')

    print('i =', i)



    min1, max1, mean1, std1 = compute_angular_error(Nx_refined_plane , Ny_refined_plane, Nz_refined_plane, Nx_gt, Ny_gt, Nz_gt, \
                          dilated_specular_mask, title='Angular error (theta): local planarity', show = 'False')

    min2, max2, mean2, std2 = compute_angular_error(Nx_refined_inpainting , Ny_refined_inpainting, Nz_refined_inpainting, Nx_gt, Ny_gt, Nz_gt, \
                          dilated_specular_mask, title='Angular error (theta): inpainting', show = 'False')

    min3, max3, mean3, std3 = compute_angular_error(Nx_refined_laplace , Ny_refined_laplace, Nz_refined_laplace, Nx_gt, Ny_gt, Nz_gt, \
                          dilated_specular_mask, title='Angular error (theta): ours', show = 'False')

    print('min, max, mean, std (RMSE), local planarity assumption:', min1, max1, mean1, std1)
    print('min, max, mean, std (RMSE), inpainting:', min2, max2, mean2, std2)
    print('min, max, mean, std (RMSE), ours:', min3, max3, mean3, std3)

    ang_error_planar_assumption_min.append(min1)
    ang_error_planar_assumption_max.append(max1)
    ang_error_planar_assumption_mean.append(mean1)
    ang_error_planar_assumption_std.append(std1)
    ang_error_inpainting_min.append(min2)
    ang_error_inpainting_max.append(max2)
    ang_error_inpainting_mean.append(mean2)
    ang_error_inpainting_std.append(std2)
    ang_error_ours_min.append(min3)
    ang_error_ours_max.append(max3)
    ang_error_ours_mean.append(mean3)
    ang_error_ours_std.append(std3)

    radii.append(i)




#plot_error_bar(radii,ang_error_planar_assumption_mean,error=[ang_error_planar_assumption_min, ang_error_planar_assumption_max]\
#               , title='Error bar local planarity assumption (angular error)')

fig = plt.figure()

plt.plot(radii,ang_error_planar_assumption_mean, label='Local planarity assumption', color='deeppink')
plt.plot(radii,ang_error_inpainting_mean, label='Image inpainting (Fast Marching)', color='orange')
plt.plot(radii,ang_error_ours_mean, label='Ours', color='springgreen')

plt.legend()
plt.grid()
plt.xlabel('distance from outer boundary (pixels)',  fontsize=10)
plt.ylabel('angular error (degrees)',  fontsize=10)
plt.title("Angular error over extended specular mask (mean)",  fontsize=15)
plt.savefig('./errors_refinement.png',  dpi=200)
#plt.show()


fig = plt.figure()

plt.plot(radii,ang_error_inpainting_mean, label='Image inpainting (Fast Marching)', color='orange')
plt.plot(radii,ang_error_ours_mean, label='Ours', color='springgreen')

plt.legend()
plt.grid()
plt.xlabel('distance from outer boundary (pixels)',  fontsize=10)
plt.ylabel('angular error (degrees)',  fontsize=10)
plt.title("Angular error over extended specular mask (mean)",  fontsize=15)
plt.savefig('./errors_refinement2.png',  dpi=200)
#plt.show()


fig = plt.figure()

plt.plot(spec_ratio,ang_error_inpainting_mean, label='Image inpainting (Fast Marching)', color='orange')
plt.plot(spec_ratio,ang_error_ours_mean, label='Ours', color='springgreen')

plt.legend()
plt.grid()
plt.xlabel('specularity ratio (specular pixels / object pixels)',  fontsize=10)
plt.ylabel('angular error (degrees)',  fontsize=10)
plt.title("Angular error over extended specular mask (mean)",  fontsize=15)
plt.savefig('./errors_refinement3.png',  dpi=200)
#plt.show()



plot_error_bar(radii,ang_error_planar_assumption_mean, ang_error_planar_assumption_std,\
               title='Error bar (local planarity assumption)', savepath='./Error_bar_planar.png')

plot_error_bar(radii,ang_error_inpainting_mean, ang_error_inpainting_std,\
               title='Error bar (image inpainting)', savepath='./Error_bar_inpainting.png')

plot_error_bar(radii,ang_error_ours_mean, ang_error_ours_std,\
               title='Error bar (ours)', savepath='./Error_bar_ours.png')

plot_error_bars(radii, ang_error_inpainting_mean, ang_error_inpainting_std, \
                ang_error_ours_mean, ang_error_ours_std, title='Error bars', savepath='./Error_bars.png')

print("The End")

#plot_with_contours(Nx_gt*(1-specular_mask), useful_contours, title='incomplete image')

