
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import cv2
import matplotlib.pyplot as plt


def undistort_and_plot(image_path, points, K, D):
    """
    Undistort the image and points, then plot the undistorted points on the undistorted image.

    :param image_path: Path to the distorted image (string)
    :param points: List of points in the distorted image (numpy array of shape (N, 2))
    :param K: Camera intrinsic matrix (3x3 numpy array)
    :param D: Distortion coefficients (1x4 numpy array)
    :return: Undistorted points as an N x 2 numpy array
    """

    # Load the distorted image
    distorted_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Use a grayscale image for simplicity
    if distorted_img is None:
        print("Error: Could not load the image. Please check the path.")
        return None

    # Undistort the image
    undistorted_img = cv2.undistort(distorted_img, K, D)

    # Convert points to the format required by undistortPoints
    points_distorted = points.reshape(-1, 1, 2)  # Reshape to (N, 1, 2)

    # Undistort the points
    points_undistorted = cv2.undistortPoints(points_distorted, K, D)

    # Calculate undistorted pixel coordinates
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Calculate undistorted pixel coordinates
    points_undistorted_pixel = points_undistorted[:, 0, :] * np.array([fx, fy]) + np.array([cx, cy])

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(undistorted_img, cmap='gray')

    # Plot undistorted points
    plt.scatter(points_undistorted_pixel[:, 0], points_undistorted_pixel[:, 1],
                color='red', marker='o', label='Undistorted Points', zorder=5)

    plt.title('Undistorted Image with Undistorted Points')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.legend()
    plt.axis('off')  # Hide axes
    plt.show()

    # Return undistorted points as an N x 2 array
    return points_undistorted_pixel


def resize_image_and_points(image_path, points, new_size=(256, 256)):
    # Load the original image
    original_img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if original_img is None:
        raise ValueError("Image not found or could not be loaded.")

    # Get original dimensions
    original_height, original_width = original_img.shape[:2]

    # Resize the image to the new dimensions
    resized_img = cv2.resize(original_img, new_size)

    # Calculate the scale factors for width and height
    scale_x = new_size[0] / original_width
    scale_y = new_size[1] / original_height

    # Ensure points is a NumPy array of floats
    points = np.array(points, dtype=np.float64)

    # Initialize lists for interpolated and integer points
    interpolated_points = []

    # Apply the scaling to the pointset
    for point in points:
        x_new = point[0] * scale_x
        y_new = point[1] * scale_y

        # Append the floating-point coordinates
        interpolated_points.append([x_new, y_new])


    return resized_img, np.array(interpolated_points)


import matplotlib.image as mpimg


def display_image_with_points(image_path, points):
    # Load the image
    img = mpimg.imread(image_path)

    # Create a figure and axis to display the image
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img)

    # Extract the X and Y coordinates from the points array
    x_coords = [float(point[0]) for point in points]
    y_coords = [float(point[1]) for point in points]

    # Plot the points on the image
    ax.scatter(x_coords, y_coords, c='lime', s=40, marker='o')

    # Optionally, you can add titles, labels, or a grid
    ax.set_title('Image with Points Overlay')

    # Show the image with points
    plt.show()

def display_image_with_points2(img, points, cmap = 'gray', title = 'Downsampled Image with Brightest Points Overlay'):
    # Load the image
    #img = mpimg.imread(image_path)

    # Create a figure and axis to display the image
    fig, ax = plt.subplots()

    # Display the image
    img_plot =  ax.imshow(img, cmap= cmap)
    plt.colorbar(img_plot, ax=ax, orientation='vertical', label='Pixel Intensity')

    # Extract the X and Y coordinates from the points array
    x_coords = [float(point[0]) for point in points]
    y_coords = [float(point[1]) for point in points]

    # Plot the points on the image
    ax.scatter(x_coords, y_coords, c='black', s=30, marker='o')

    # Optionally, you can add titles, labels, or a grid
    ax.set_title(title)


    # Show the image with points
    plt.show()

import csv
def extract_columns_b_c(csv_file_path):
    data = []

    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)

        # Skip the header if your CSV has one (uncomment if needed)
        # next(reader)

        for row in reader:
            # Assuming columns B and C are the 2nd and 3rd columns (index 1 and 2)
            data.append([row[1], row[2]])

    # Convert to a numpy array (optional, if you want the result as a NumPy array)
    result = np.array(data)
    print('result shape:', result.shape)
    return result

def compute_normals(K, points):
    # Ensure K is a float type for calculations
    K = np.array(K, dtype=np.float64)


    # Check that K is a 3x3 matrix
    if K.shape != (3, 3):
        raise ValueError("K must be a 3x3 matrix.")

    # Calculate the inverse of the matrix K
    K_inv = np.linalg.inv(K)

    # Ensure points is a float type and check its shape
    points = np.array(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an N x 2 array.")

    # Initialize an array to store the transformed points (N x 3)
    N = points.shape[0]  # Number of points
    transformed_points = np.zeros((N, 3))  # Preallocate an N x 3 array

    # Process each point
    for i in range(N):
        # Convert (x, y) to homogeneous coordinates (x, y, 1)
        homogeneous_point = np.array([points[i, 0], points[i, 1], 1.], dtype=np.float64) #{{{{{{{{{{{
        print("homogeneous point:", homogeneous_point)

        # Multiply the inverse of K by the homogeneous point
        transformed_point = np.dot(K_inv, homogeneous_point)

        # Normalize the resulting vector by dividing by its norm
        norm = np.linalg.norm(transformed_point)
        if norm != 0:
            transformed_points[i] = transformed_point / norm  # Normalize only if norm is not zero
        else:
            transformed_points[i] = transformed_point  # Handle the zero vector case
        print('normal:', transformed_points[i])

    return transformed_points

def compute_normals_with_flip(K, points, image_height=1080):
    # Ensure K is a float type for calculations
    points = np.array(points, dtype=np.float64)

    if K.shape != (3, 3):
        raise ValueError("K must be a 3x3 matrix.")

    # Calculate the inverse of the matrix K
    K_inv = np.linalg.inv(K)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an N x 2 array.")

    # Flip the y-coordinates to match origin in bottom-left
    flipped_points = np.copy(points)
    flipped_points[:, 1] = image_height - points[:, 1]

    # Initialize an array to store the transformed points (N x 3)
    N = flipped_points.shape[0]  # Number of points
    transformed_points = np.zeros((N, 3))  # Preallocate an N x 3 array

    # Process each point
    for i in range(N):
        # Convert (x, y) to homogeneous coordinates (x, y, 1)
        homogeneous_point = np.array([flipped_points[i, 0], flipped_points[i, 1], 1], dtype=np.float64)

        # Multiply the inverse of K by the homogeneous point
        transformed_point = np.dot(K_inv, homogeneous_point)

        # Normalize the resulting vector by dividing by its norm
        norm = np.linalg.norm(transformed_point)
        if norm != 0:
            transformed_points[i] = transformed_point / norm  # Normalize only if norm is not zero
        else:
            transformed_points[i] = transformed_point  # Handle the zero vector case

    return transformed_points


from scipy.ndimage import map_coordinates
def get_interpolated_points_and_intensities(image_array, points):
    # Ensure points is a NumPy array of floats
    points = np.array(points, dtype=np.float64)

    # Check if points shape is N x 2
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an N x 2 array.")

    # Prepare for bilinear interpolation
    x_coords = points[:, 0]  # X coordinates
    y_coords = points[:, 1]  # Y coordinates

    # Use map_coordinates for bilinear interpolation to get pixel intensities
    pixel_intensities = map_coordinates(image_array, [y_coords, x_coords], order=1)

    # Create an N x 2 array of interpolated points
    interpolated_points = np.column_stack((x_coords, y_coords))

    # Return interpolated points and pixel intensities as an N x 1 array
    return interpolated_points, pixel_intensities.reshape(-1, 1)


def concatenate_BPs_depths_and_normals(BPs_2D, depth, Normals, scale=1):
    # Ensure BPs_2D is an N x 2 array and depth is an N x 1 array
    BPs_2D = np.array(BPs_2D)  # Ensure it's a NumPy array
    depth = scale*np.array(depth)  # Ensure it's a NumPy array
    Normals = np.array(Normals)  # Ensure it's a NumPy array

    # Validate shapes
    if BPs_2D.ndim != 2 or BPs_2D.shape[1] != 2:
        raise ValueError("BPs_2D must be an N x 2 array.")
    if depth.ndim != 2 or depth.shape[1] != 1 or depth.shape[0] != BPs_2D.shape[0]:
        raise ValueError("depth must be an N x 1 array matching the number of points.")
    if Normals.ndim != 2 or Normals.shape[1] != 3 or Normals.shape[0] != BPs_2D.shape[0]:
        raise ValueError("Normals must be an N x 3 array matching the number of points.")

    # Concatenate BPs_2D and depth to create an N x 3 array
    points_vector = np.hstack((BPs_2D, depth))

    # Concatenate the points_vector with the Normals to create an N x 6 array
    concatenated_array = np.hstack((points_vector, Normals))

    return concatenated_array


# Fonction pour créer un U-Net modifié pour la prédiction de profondeur
def build_unet_depth_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)

    # Encodeur (chemin contractant)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Couche bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Décodeur (chemin expansif)
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # Sortie : Carte de profondeur
    outputs = layers.Conv2D(1, (1, 1), activation='linear')(c9)

    model = Model(inputs, outputs)
    return model
model = build_unet_depth_model()
model.compile(optimizer='adam', loss='mean_squared_error')

# Charger l'image et la redimensionner
#img_path = '/home/karim/Bureau/karim/Bureau/reconstruction/Unet_depth_uncertainty/Image100.png'

#img_path =  '/home/karim/Bureau/karim/Bureau/colonData/images/30.png'

img_path =  '/home/karim/Bureau/Seq_023/Seq_023_02/Images/Image82.png'

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (256, 256))

# Normaliser l'image
img_normalized = img_resized / 255.0
img_input = np.expand_dims(img_normalized, axis=0)

# Faire plusieurs prédictions pour estimer la carte de profondeur et l'incertitude
n_passes = 20  # Nombre de prédictions à effectuer
predictions = []

for i in range(n_passes):
    prediction = model.predict(img_input)
    predictions.append(prediction)

# Convertir les prédictions en tableau numpy
predictions = np.array(predictions)

# Calculer la carte de profondeur moyenne et l'incertitude (écart-type)
mean_depth = np.mean(predictions, axis=0).squeeze()  # Carte de profondeur moyenne
uncertainty_map = np.std(predictions, axis=0).squeeze()  # Carte d'incertitude (écart-type)

from scipy.ndimage import gaussian_filter

smooth_depth_map = gaussian_filter(mean_depth, sigma=2)
smooth_uncertainty_map = gaussian_filter(uncertainty_map, sigma=2)

# Afficher la carte de profondeur moyenne
plt.figure(figsize=(10, 5))
plt.imshow(smooth_depth_map, cmap='inferno')
plt.colorbar()
plt.title('Mean depth map')
plt.show()

# Afficher la carte d'incertitude
plt.figure(figsize=(10, 5))
plt.imshow(smooth_uncertainty_map, cmap='plasma')
plt.colorbar()
plt.title('Uncertainty map')
plt.show()


################# Get original pixels of brightest points captured in the resized (downsampled image) #################


def get_original_coordinates(x_new, y_new, orig_shape, new_shape):
    """
    Given a pixel coordinate in the resized image, calculate the backward coordinates in the original image.

    Args:
    - x_new, y_new: Coordinates of the pixel in the resized image.
    - orig_shape: Tuple (H_orig, W_orig) representing the original image shape.
    - new_shape: Tuple (H_new, W_new) representing the resized image shape.

    Returns:
    - (x_orig, y_orig): Coordinates of the corresponding pixel in the original image.
    """
    H_orig, W_orig = orig_shape
    H_new, W_new = new_shape

    # Calculate the backward coordinates in the original image
    x_orig = (x_new / W_new) * W_orig
    y_orig = (y_new / H_new) * H_orig

    return x_orig, y_orig

new_shape = img_resized.shape[:2]
original_shape = img.shape[:2]

print("shape of original image:", original_shape)
print("shape of resized (downsampled) image:", new_shape)


# Pixel in resized image (x_new, y_new)
x_new, y_new = 128, 128  # Example: center pixel in resized image

# Get the corresponding coordinates in the original image
x_orig, y_orig = get_original_coordinates(x_new, y_new, original_shape, new_shape)
print(f'Original coordinates: ({x_orig}, {y_orig})')



######################Compute normals, extract coordinates of brightest points manually selected using Makesense AI #####

'''
K = np.array([[1506.46, 0, 624.0],
              [0, 1506.46, 540.0],
              [0, 0, 1]])
'''


# Define the camera intrinsic matrix and distortion coefficients
K = np.array([[785.2521, 0, 741.1905],
              [0, 786.6768, 542.8783],
              [0, 0, 1]])

# Distortion coefficients
D = np.array([-0.1234686, -0.008368, 0.001964611, -9.119877e-05])



#BPs = extract_columns_b_c('/home/karim/Bureau/karim/Bureau/colonData/BPs/30.csv')

BPs = np.load('/home/karim/Bureau/Seq_023/Seq_023_02/Brightest_points/Image82.npy')
print("BPs:", BPs)

############ Undistort Bps (optional)
#BPs = undistort_and_plot(img_path, BPs, K, D)


Normals = compute_normals(K, BPs)
#Normals= compute_normals_with_flip(K, BPs)
print("Normals", Normals)

display_image_with_points(img_path, BPs)
resized_img, resized_BPs = resize_image_and_points(img_path, BPs, new_size=(256, 256))
print("resized_BPs:", resized_BPs)


display_image_with_points2(resized_img, resized_BPs)
#display_image_with_points2(smooth_depth_map, resized_BPs, cmap = 'plasma', title = 'Depth Image with Brightest Points Overlay')




interpolated_downsampled_BPs , depths = get_interpolated_points_and_intensities(smooth_depth_map, resized_BPs)
display_image_with_points2(smooth_depth_map, interpolated_downsampled_BPs, cmap = 'plasma', title = 'Depth Image with interpolated Brightest Points Overlay')


Reconstruction_result = concatenate_BPs_depths_and_normals(BPs, depths, Normals, scale=100)

import matplotlib.cm as cm

def plot_vector_field_with_colormap0(vector_field_array, scale_factor=0.1):
    # Convert input to a NumPy array with float type
    vector_field_array = np.array(vector_field_array, dtype=float)

    # Validate shape
    if vector_field_array.ndim != 2 or vector_field_array.shape[1] != 6:
        raise ValueError("Input must be an N x 6 array.")

    # Extract origins and vector components
    origins = vector_field_array[:, :3]  # First three columns (x, y, z)
    vectors = vector_field_array[:, 3:]  # Last three columns (u, v, w)
    color_values = vector_field_array[:, 2]  # Third column for color mapping

    # Scale the colors based on mean and max values
    norm = plt.Normalize(vmin=np.min(color_values), vmax=np.max(color_values))
    colors = cm.plasma(norm(color_values))  # Get plasma colormap

    # Scale the vectors to adjust their length
    scaled_vectors = vectors * scale_factor

    # Plot the vector field
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the origins as points colored by the third column
    scatter = ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2],
                         c=colors, s=50, label='Reconstructed Brightest Points')

    # Plot the vectors in black with thicker heads
    ax.quiver(origins[:, 0], origins[:, 1], origins[:, 2],
              scaled_vectors[:, 0], scaled_vectors[:, 1], scaled_vectors[:, 2],
              length=1.0, normalize=True, color='k', label='Normals',
              arrow_length_ratio=0.4, linewidth=2)  # Increase linewidth for thicker arrows

    # Set axis limits dynamically based on the data
    ax.set_xlim([np.min(origins[:, 0]) - 1, np.max(origins[:, 0]) + 1])
    ax.set_ylim([np.min(origins[:, 1]) - 1, np.max(origins[:, 1]) + 1])
    ax.set_zlim([np.min(origins[:, 2]) - 1, np.max(origins[:, 2]) + 1])

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Sightline-based Specular Normals')

    # Add colorbar
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='plasma'), ax=ax)
    cbar.set_label('Depth')

    plt.show()





def plot_vector_field_manual(vector_field_array, scale_factor=3.0):
    # Convert input to a NumPy array with float type
    vector_field_array = np.array(vector_field_array, dtype=float)

    # Validate shape
    if vector_field_array.ndim != 2 or vector_field_array.shape[1] != 6:
        raise ValueError("Input must be an N x 6 array.")

    # Extract origins and vector components
    origins = vector_field_array[:, :3]  # First three columns (x, y, z)
    vectors = vector_field_array[:, 3:]  # Last three columns (u, v, w)
    depth_values = vector_field_array[:, 2]  # Third column for depth mapping

    # Reference vector (0, 0, 1) for calculating orientation
    reference_vector = np.array([0, 0, 1])

    # Compute angles in radians between each vector and the reference (0, 0, 1)
    dot_product = np.dot(vectors, reference_vector)
    vector_lengths = np.linalg.norm(vectors, axis=1)  # Should be 1 for unit vectors
    cos_theta = dot_product / (vector_lengths + 1e-8)  # Avoid division by zero
    angles_in_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip values to avoid numerical errors
    angles_in_degrees = np.degrees(angles_in_radians)  # Convert to degrees

    # Normalize angles for color mapping (orientation)
    norm_orientation = plt.Normalize(vmin=np.min(angles_in_degrees), vmax=np.max(angles_in_degrees))
    orientation_colors = cm.coolwarm(norm_orientation(angles_in_degrees))  # Use the coolwarm colormap for orientation

    # Normalize color values for depth
    norm_depth = plt.Normalize(vmin=np.min(depth_values), vmax=np.max(depth_values))
    depth_colors = cm.plasma(norm_depth(depth_values))  # Depth color from plasma colormap

    # Plot the vector field manually
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the origins as points colored by depth
    scatter = ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2],
                         c=depth_colors, s=50, label='Reconstructed Brightest Points')

    # Manually plot vectors using `ax.plot`
    for i in range(origins.shape[0]):
        origin = origins[i]
        vector = vectors[i]
        color = orientation_colors[i]

        # Plot the line for the vector (origin to origin + vector scaled)
        ax.plot([origin[0], origin[0] + vector[0] * scale_factor],
                [origin[1], origin[1] + vector[1] * scale_factor],
                [origin[2], origin[2] + vector[2] * scale_factor],
                color=color, linewidth=2)

        # Optionally add a small arrowhead at the end
        ax.quiver(origin[0], origin[1], origin[2],
                  vector[0] * scale_factor, vector[1] * scale_factor, vector[2] * scale_factor,
                  color=color, arrow_length_ratio=0.2, linewidth=3)

    # Dynamically set axis limits with padding
    padding = 2.0  # Change this value to adjust padding
    ax.set_xlim([np.min(origins[:, 0]) - padding, np.max(origins[:, 0]) + padding])
    ax.set_ylim([np.min(origins[:, 1]) - padding, np.max(origins[:, 1]) + padding])
    ax.set_zlim([np.min(origins[:, 2]) - padding, np.max(origins[:, 2]) + padding])

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Sightline-based Specular Normals (Colored by Orientation and Depth)')

    # Set view angle for better visualization
    ax.view_init(elev=30, azim=45)  # Adjust elevation and azimuth angles as needed

    # Add grid
    ax.grid(True)

    # Add colorbar for depth (right side) using the plasma colormap
    cbar_depth = plt.colorbar(cm.ScalarMappable(norm=norm_depth, cmap='plasma'), ax=ax, location='right')
    cbar_depth.set_label('Depth')

    # Add colorbar for orientation (left side, in degrees) using the coolwarm colormap
    cbar_orientation = plt.colorbar(cm.ScalarMappable(norm=norm_orientation, cmap='coolwarm'), ax=ax, location='left')
    cbar_orientation.set_label('Orientation (Degrees from Z-axis)')

    plt.show()

from mayavi import mlab

def plot_vector_field_mayavi(vector_field_array, scale_factor=3.0):
    # Convert input to a NumPy array with float type
    vector_field_array = np.array(vector_field_array, dtype=float)

    # Validate shape
    if vector_field_array.ndim != 2 or vector_field_array.shape[1] != 6:
        raise ValueError("Input must be an N x 6 array.")

    # Extract origins and vector components
    origins = vector_field_array[:, :3]  # First three columns (x, y, z)
    vectors = vector_field_array[:, 3:]  # Last three columns (u, v, w)

    # Create a figure
    mlab.figure(size=(800, 600))

    # Plot the vector field using quiver3d
    mlab.quiver3d(origins[:, 0], origins[:, 1], origins[:, 2],
                   vectors[:, 0], vectors[:, 1], vectors[:, 2],
                   scale_factor=scale_factor, mode='arrow', color=(0, 0, 1))

    # Set axes labels
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')

    # Show the visualization
    mlab.show()

import plotly.graph_objects as go


from plotly.subplots import make_subplots



def plot_vector_field_manual_plotly2(vector_field_array, scale_factor=3.0):
    # Convert input to a NumPy array with float type
    vector_field_array = np.array(vector_field_array, dtype=float)

    # Validate shape
    if vector_field_array.ndim != 2 or vector_field_array.shape[1] != 6:
        raise ValueError("Input must be an N x 6 array.")

    # Extract origins and vector components
    origins = vector_field_array[:, :3]  # First three columns (x, y, z)
    vectors = vector_field_array[:, 3:]  # Last three columns (u, v, w)
    depth_values = vector_field_array[:, 2]  # Third column for depth mapping

    # Normalize depth values for color mapping
    norm_depth = (depth_values - np.min(depth_values)) / (np.max(depth_values) - np.min(depth_values))

    # Prepare the figure
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scatter3d'}]],
    )

    # Scatter plot for the origins, colored by depth
    scatter_origins = go.Scatter3d(
        x=origins[:, 0], y=origins[:, 1], z=origins[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=norm_depth,  # Color by depth
            colorscale='Plasma',
            colorbar=dict(
                title='Depth',
                tickvals=np.linspace(0, 1, 5),
                ticktext=np.linspace(np.min(depth_values), np.max(depth_values), 5),
                len=0.5,
                x=0.85  # Position colorbar on the right
            ),
        ),
        name='Reconstructed Brightest Points'
    )

    # Add lines to represent the vectors, all in the same color
    quiver_lines = []

    for i in range(origins.shape[0]):
        origin = origins[i]
        vector = vectors[i] * scale_factor

        # Create a line segment for each vector with a fixed color
        quiver_lines.append(go.Scatter3d(
            x=[origin[0], origin[0] + vector[0]],
            y=[origin[1], origin[1] + vector[1]],
            z=[origin[2], origin[2] + vector[2]],
            mode='lines',
            line=dict(
                width=5,
                color='blue'  # Fixed color for all vectors
            ),
            showlegend=False
        ))

    # Add all traces to the figure
    fig.add_trace(scatter_origins)
    for line in quiver_lines:
        fig.add_trace(line)

    # Set axis labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis',
            aspectmode='auto'
        ),
        title='Sightline-based Specular Normals (Colored by Depth)',
        showlegend=False
    )

    # Show the figure
    fig.show()





plot_vector_field_manual(Reconstruction_result, scale_factor=3.0)

plot_vector_field_manual_plotly2(Reconstruction_result, scale_factor=5.0)


print(Reconstruction_result)

#########################################



