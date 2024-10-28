# -*- coding: utf-8 -*-

"""
  ©
  Author: Karim Makki
"""

import visvis as vv
import trimesh
import numpy as np
import os
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
import argparse
import timeit
from scipy.ndimage.interpolation import map_coordinates

## Import tools for computing curvature on explicit surfaces (for comparison purposes)
import slam_curvature as scurv
import CurvatureCubic as ccurv
import CurvatureWpF as WpFcurv
import CurvatureISF as ISFcurv
from trimesh import curvature
import DiffGeoOps as diffgeo


#### Compute the adjoint of the Hessian matrix (faster than the numpy version defined below)
#### Reference: Ron Goldman, Curvature formulas for implicit curves and surfaces, Computer Aided Geometric Design 22 (2005) 632–658




def display_mesh(verts, faces, normals, texture, range, save_path):

    mesh = vv.mesh(verts, faces, normals, texture)
    f = vv.gca()
    mesh.colormap = vv.CM_JET
    f.axis.visible = False

    #f.bgcolor = None #1,1,1 #None
    #mesh.edgeShading = 'smooth'
    #mesh.clim = np.min(texture),np.max(texture)
    mesh.clim =  -range,range  #

    #mesh.clim = -0.05, 0.05  # 2 -range,range

    vv.callLater(1.0, vv.screenshot, save_path, vv.gcf(), sf=2, bg='w')
    vv.colorbar()
    ## for blender simulated sequence
    #vv.view({'zoom': 0.3, 'azimuth': 0.0, 'elevation': -90.0})
    #for liver
    vv.view({'zoom': 0.004, 'azimuth': 0.0, 'elevation': -90.0})
    #vv.view({'zoom': 0.006, 'azimuth': -80.0, 'elevation': -5.0})
    vv.use().Run()

    return 0


###    Affect texture value to each vertex by spline interpolation, for the different modes as explained in: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html

def texture_spline_interpolation3D(verts, texture):

    return map_coordinates(texture,[verts[:,0],verts[:,1],verts[:,2]],order=3, mode='nearest')


#### Affect texture value to each vertex by averaging neighbrhood information
def texture_mean_avg_interpolation3D(verts, texture):

    X = np.rint(verts[:,0]).astype(int)
    Y = np.rint(verts[:,1]).astype(int)
    Z = np.rint(verts[:,2]).astype(int)

    return (texture[X-1,Y,Z] + texture[X+1,Y,Z] + texture[X,Y-1,Z] + texture[X,Y+1,Z] + texture[X,Y,Z-1] + texture[X,Y,Z+1])/6

#### Affect texture value to each vertex by nearest neighbour interpolation
def texture_nearest_neigh_interpolation3D(verts, texture):

    return texture[np.rint(verts[:,0]).astype(int),np.rint(verts[:,1]).astype(int),np.rint(verts[:,2]).astype(int)]



def plot_histogram(data, title, opath):

    import matplotlib.pyplot as plt

    _ = plt.hist(data, bins='auto')
    plt.title(title)
    plt.savefig(opath, dpi=500)
    #plt.show()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('-in', '--mask', help='3D shape binary mask, as NIFTI file', type=str, required = True)
    parser.add_argument('-o', '--output', help='output directory', type=str, default = './Specular_curvature_results')
    #parser.add_argument('-dmap', '--dmap', help='distance_map: 0 if Euclidean, 1 if geodesic distance map, and 2 if binary step function', type=int, default = 1)

    args = parser.parse_args()

    im_id = 50

    # Example of use : python3 fast_Gaussian_curvature_3D.py -in ./3D_data/stanford_bunny_binary.nii.gz

    #output_path = args.output+'/'+str(im_id)

    ### output folder for liver

    output_path = args.output + '/liver/' + str(im_id)

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    start_time = timeit.default_timer()


    elapsed = timeit.default_timer() - start_time
    print("The proposed method takes (in seconds):\n")
    print(elapsed)


    #display_mesh(verts, faces, normals, gaussian_curv, os.path.join(output_path, "Gaussian_curature_Makki.png"))


##To compare results with other methods defining the surface explicitly, please uncomment one of the following blocks #################


# #######################################################################################################################################
# ############### To compare results with the Trimesh Gaussian curvature, please uncomment this block ##################################



    #meshpath = '/home/karim/Bureau/reconstruction/Curvature/Meshes/'+str(im_id)+'.ply'

    # mesh path for liver

    meshpath = '/home/karim/Bureau/reconstruction/Curvature/liver/experiments/' + str(im_id)\
                +'/registered_model/view0/liver_rotated.obj'

    m = trimesh.load_mesh(os.path.join(output_path, meshpath))

    print(m.vertices.shape)
    texture = np.zeros(m.vertices.shape[0])
    display_mesh(m.vertices, m.faces, m.vertex_normals, texture, 1,
                                  os.path.join(output_path, "specular_points.png"))

    start_time = timeit.default_timer()

     #tr_gaussian_curv = curvature.discrete_gaussian_curvature_measure(m, m.vertices, 2)
    #tr_gaussian_curv = curvature.discrete_gaussian_curvature_measure(m, m.vertices, 4)

    #PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(m)
    #gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]


    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(m)
    gaussian_curv =  PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])

    np.save(os.path.join(output_path, "PrincipalDir1.npy"), PrincipalDir1)
    np.save(os.path.join(output_path, "PrincipalDir2.npy"), PrincipalDir2)

    display_mesh(m.vertices, m.faces, m.vertex_normals, mean_curv, 0.4, os.path.join(output_path, "mean_curvature.png"))
    display_mesh(m.vertices, m.faces, m.vertex_normals, gaussian_curv, 0.04, os.path.join(output_path, "gaussian_curvature.png"))
    display_mesh(m.vertices, m.faces, m.vertex_normals, PrincipalCurvatures[0, :], 0.3, os.path.join(output_path, "min_curvature.png"))
    display_mesh(m.vertices, m.faces, m.vertex_normals, PrincipalCurvatures[1, :], 0.3, os.path.join(output_path, "max_curvature.png"))

    np.save(os.path.join(output_path, "k1.npy"), PrincipalCurvatures[0, :])
    np.save(os.path.join(output_path, "k2.npy"), PrincipalCurvatures[1, :])

    np.save(os.path.join(output_path, "mean_curvature.npy"), mean_curv)
    np.save(os.path.join(output_path, "Gaussian_curvature.npy"), gaussian_curv)

"""

    #gaussian_curv = WpFcurv.GetCurvatures(m.vertices, m.faces)[0]
    #mean_curv = WpFcurv.GetCurvatures(m.vertices, m.faces)[1]
    #k1 = mean_curv + np.sqrt(np.absolute(mean_curv**2-gaussian_curv))
    #k2 = mean_curv - np.sqrt(np.absolute(mean_curv**2 - gaussian_curv))

    #plot_histogram(PrincipalCurvatures[0, :], 'Min curvature histogram', os.path.join(output_path, "min_curvature_histogram.png"))
    #plot_histogram(PrincipalCurvatures[1, :], 'Max curvature histogram', os.path.join(output_path, "max_curvature_histogram.png"))
    #plot_histogram(mean_curv, 'Mean curvature histogram', os.path.join(output_path, "mean_curvature_histogram.png"))
    #plot_histogram(gaussian_curv, 'Gaussian curvature histogram', os.path.join(output_path, "gaussian_curvature_histogram.png"))




    #gaussian_curv = WpFcurv.GetCurvatures(m.vertices, m.faces)[0]

    # Cubic method
    #A_mixed, mean_curvature_normal_operator_vector = diffgeo.calc_A_mixed(m.vertices, m.faces)

    # Meyer method
    gaussian_curv_Meyer = ccurv.CurvatureCubic(m.vertices,m.faces)[0] #diffgeo.get_gaussian_curvature(m.vertices, m.faces, A_mixed)
    mean_curv_Meyer = ccurv.CurvatureCubic(m.vertices,m.faces)[1]#diffgeo.get_mean_curvature(m.vertices, m.faces, A_mixed)

    # Iterative fitting method

    #gaussian_curv = ISFcurv.CurvatureISF2(m.vertices, m.faces)[0]

    elapsed = timeit.default_timer() - start_time

    print("The Trimesh method takes (in seconds):\n")

    print(elapsed)

    display_mesh(m.vertices, m.faces, m.vertex_normals, gaussian_curv_Meyer, 12,
                 os.path.join(output_path, "gaussian_curvature_Meyer.png"))

    display_mesh(m.vertices, m.faces, m.vertex_normals, mean_curv_Meyer, 12,
                 os.path.join(output_path, "mean_curvature_Meyer.png"))

    np.save(os.path.join(output_path, "mean_curvature_Meyer.npy"), mean_curv_Meyer)
    np.save(os.path.join(output_path, "Gauss_curvature_Meyer.npy"), gaussian_curv_Meyer)


    #display_mesh(m.vertices, m.faces, m.vertex_normals, k1, 12,
    #             os.path.join(output_path, "k1.png"))
    #display_mesh(m.vertices, m.faces, m.vertex_normals, k2, 12,
    #             os.path.join(output_path, "k2.png"))

    #np.savetxt(os.path.join(output_path, "min_curvature.txt"), PrincipalCurvatures[0, :], fmt='%d')
    #np.savetxt(os.path.join(output_path, "max_curvature.txt"), PrincipalCurvatures[1, :], fmt='%d')



    #np.savetxt(os.path.join(output_path, "min_curvature.txt"), k1, fmt='%d')
    #np.savetxt(os.path.join(output_path, "max_curvature.txt"), k2, fmt='%d')

    #np.save(os.path.join(output_path, "min_curvature.npy"), k1)
    #np.save(os.path.join(output_path, "max_curvature.npy"), k2)



#
# ########################################################################################################################################

# #######################################################################################################################################
# ##### To compare results with the Rusinkiewicz (v1) Gaussian curvature, please uncomment this block ###################################
#
#     m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
#
#     start_time = timeit.default_timer()
#     # Comptue estimations of principal curvatures
#     PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(m)
#     gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The Rusinkiewicz method v1 takes (in seconds):\n")
#     print(elapsed)
#
#     #gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)
#     display_mesh(m.vertices, m.faces, m.vertex_normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_Rusinkiewicz_v1.png"))
# #########################################################################################################################################


# #########################################################################################################################################
# ##### To compare results with the Rusinkiewicz (v2) Gaussian curvature, please uncomment this block #####################################
# ########################### Note that the second version is quite  faster than the first ################################################
#
#     m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
#
#     start_time = timeit.default_timer()
#
#     #K,H,VN = WpFcurv.GetCurvatures(m.vertices,m.faces)
#     gaussian_curv = WpFcurv.GetCurvatures(m.vertices,m.faces)[0]
#
#     elapsed = timeit.default_timer() - start_time
#
#
#     print("The Rusinkiewicz method v2 takes (in seconds):\n")
#     print(elapsed)
#     #print(np.min(gaussian_curv),np.max(gaussian_curv), np.sqrt(np.absolute(np.mean(gaussian_curv)-(1/R**2))))
#
#     #gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)
#     display_mesh(m.vertices, m.faces, m.vertex_normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_Rusinkiewicz_v2.png"))
# #########################################################################################################################################


# #########################################################################################################################################
# ##### To compare results with those of the cubic order algorithm, please uncomment this block ###########################################
#
#     m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
#
#     start_time = timeit.default_timer()
#
#     #K,H,VN = ccurv.CurvatureCubic(m.vertices,m.faces)
#     gaussian_curv = ccurv.CurvatureCubic(m.vertices,m.faces)[0]
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The cubic order algorithm takes (in seconds):\n")
#     print(elapsed)
#
#     #gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)
#     display_mesh(m.vertices, m.faces, m.vertex_normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_cubic_order.png"))
# ##########################################################################################################################################

# #########################################################################################################################################
# ##### To compare results with the iterative fitting method, please uncomment this block #################################################
#
#     m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
#
#     start_time = timeit.default_timer()
#
#     gaussian_curv = ISFcurv.CurvatureISF2(m.vertices,m.faces)[0]
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The iterative fitting method takes (in seconds):\n")
#     print(elapsed)
#
#     #gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)
#     display_mesh(m.vertices, m.faces, m.vertex_normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_iterative_fitting.png"))
# ##########################################################################################################################################

# #########################################################################################################################################
# ############## To compare results with the method of Meyer, please uncomment this block #################################################
#
#     m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
#
#     start_time = timeit.default_timer()
#
#     A_mixed, mean_curvature_normal_operator_vector = diffgeo.calc_A_mixed(m.vertices, m.faces)
#     gaussian_curv = diffgeo.get_gaussian_curvature(m.vertices, m.faces, A_mixed)
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The method of Meyer takes (in seconds):\n")
#     print(elapsed)
#
#     display_mesh(m.vertices, m.faces, m.vertex_normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_Meyer.png"))
# ##########################################################################################################################################
"""