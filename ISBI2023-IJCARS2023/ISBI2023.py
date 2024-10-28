from mylib import*

##################### Scene parameters #################################
########################################################################

L = [-210.,0.,210.]
V = [100.,0.,100.]

#L = [-100.,100.,201.]
#V = [150.,140.,100.]

#L = [0.,0.,100.]
#V = [0.,0.,100.]
n = 5
size = [500, 500]

########################################################################
### Compute specular image using Phong's model

image = compute_specular_image(L, V, n, size, scale=1, sigma=0, diffuse=True)
image /= np.max(image)

plot_many_isocontours(image)
## sigma is the smoothing kernel, if sigma is zero, no smoothing is performed

fig = plt.figure(figsize=(10, 10))
plt.imshow(image.T, cmap="gray")
plt.axis([0, size[0], 0, size[1]])
plt.colorbar(fraction=0.046, pad=0.04)
#plt.title('Image plane S')
plt.axis('off')
plt.savefig('/home/karim/Bureau/Results/specularity_gt.png')
plt.show()
#########################################################################

# Compute brightest point in 3D world coordinates (theoretical)

P0, lamda = determine_brightest_point(V, L)

print('P0 in 3D world coordinates:', P0)
print('lamda:', lamda)

# Express brightest point coordinates in the image coordinates

p0x = change_coordinate_range(P0[0], -size[0]/2, size[0]/2, 0, size[0])
p0y = change_coordinate_range(P0[1], -size[1]/2, size[1]/2, 0, size[1])

P0_im = [p0x, p0y, 0]
print('P0 in 3D image coordinates:', P0_im)

fig = plt.figure(figsize=(10, 10))
plt.plot(p0x, p0y, marker="o", markersize=7, markeredgecolor="red", markerfacecolor="red")
plt.imshow(image.T, cmap='gray')  # display
plt.axis([0, size[0], 0, size[1]])
plt.title('Brightest point\'s real world coordinates : P0= {}'.format(P0)+'\n Brightest point\'s image coordinates : P0_im= {}'.format(P0_im))
plt.savefig('/home/karim/Bureau/Results/BP.png')
plt.show()


#########################################################################
########################## plot 3D scene ################################

#plot_3Dscene1(V, L, image, P0)

#########################################################################

## Show orthogonal projection of R and V on the plane S by simply setting Rz = Vz = 0


# Express the coordinates of  r and v in the image coordinate system

rx = change_coordinate_range(L[0], -size[0]/2, size[0]/2, 0, size[0])
ry = change_coordinate_range(L[1], -size[1]/2, size[1]/2, 0, size[1])

vx = change_coordinate_range(V[0], -size[0]/2, size[0]/2, 0, size[0])
vy = change_coordinate_range(V[1], -size[1]/2, size[1]/2, 0, size[1])

print('r =', [rx,ry])
print('v =', [vx,vy])

# Display

fig = plt.figure(figsize=(10, 10))
plt.plot(rx, ry, marker="o", markersize=7, markeredgecolor="magenta", markerfacecolor="magenta", label='r', linestyle = 'None', zorder=2)
plt.plot(vx, vy, marker="o", markersize=7, markeredgecolor="cyan", markerfacecolor="cyan", label='v', linestyle = 'None', zorder=2)
plt.plot(p0x, p0y, marker="o", markersize=7, markeredgecolor="red", markerfacecolor="red", label='p0', linestyle = 'None', zorder=2)
plt. plot([rx,vx], [ry,vy], '--', color='chartreuse',  label='axis of symmetry', zorder=1)
plt.legend()
plt.imshow(image.T, cmap='gray') #display
plt.axis([0, size[0], 0, size[1]])
plt.title('Scene parameters projected on image plane (S)')
plt.savefig('/home/karim/Bureau/Results/projection.png')
plt.show()

p0x, p0y = determine_brightest_point_from_image(image)
print('P0 in 3D image coordinates:', [p0x, p0y, 0])

############## Get coordinates from the iso-contour (from the image)

c = np.max(image)
t = 0.5*np.max(image)

print("c=" , c)
print("t=" , t)

isocontour = exctract_isocontour(image, t)


fig = plt.figure(figsize=(10, 10))
plt.plot(isocontour[:,0],isocontour[:,1], color='r', linewidth=2)
plt.imshow(image.T, cmap='gray')
plt.title('Isocurve: I =  %1.2f' % t)
plt.axis([0, size[0], 0, size[1]])
plt.savefig('/home/karim/Bureau/Results/isocontour.png')
plt.show()


print("n=" , n)


k = determine_k(t, c, n)
print("k=" , k)

a_exact = real_coefficients(k, lamda, V)

print("exact a=", a_exact)

#Q_xy = apply_quatric_fitting(x, y, a)

iso_x = change_coordinate_range(isocontour[:,0], 0, size[0], -size[0]/2, size[0]/2)
iso_y = change_coordinate_range(isocontour[:,1], 0, size[1], -size[1]/2, size[1]/2)

#iso_x = change_coordinate_range(isocontour[:,0], 0, size[0], -0.5, 0.5)
#iso_y = change_coordinate_range(isocontour[:,1], 0, size[1], -0.5, 0.5)

print(isocontour[:,0]-iso_x)


fig = plt.figure(figsize=(10, 10))
plt.plot(p0x, p0y, marker="o", markersize=7, markeredgecolor="red", markerfacecolor="red", label='p0: image coordinates', linestyle = 'None')
plt.plot(P0[0], P0[1], marker="o", markersize=7, markeredgecolor="cyan", markerfacecolor="cyan", label='p0: real coordinates', linestyle = 'None')

plt.scatter(isocontour[:,0], isocontour[:,1], s=3, label='isocontour: image coordinates')
plt.scatter(iso_x, iso_y, s=3, label='isocontour: real coordinates')
plt.imshow(image.T, cmap='gray')
plt.axis([-150, size[0], -150, size[1]])
plt.legend()
plt.show()


Q_xy = apply_quartic_fitting(iso_x, iso_y, a_exact)
#print(Q_xy)

print("V=", V)

### check theoretical model for one isocontour


cm = plt.cm.get_cmap('jet')
fig = plt.figure(figsize=(10, 10))

plt.imshow(image.T, cmap='gray')

sc = plt.scatter(isocontour[:, 0], isocontour[:, 1], c=Q_xy, s=2, cmap=cm)
#plt.colorbar(sc)
plt.colorbar(sc,fraction=0.046, pad=0.04)
plt.show()

lamda1, k1, tau1 = recover_scene_parameters(a_exact, V)
#lamda1, k1, tau1 = recover_scene_parameters(a_exact, vector_norm(V))
n1 = retrieve_n(t,c,tau1)

print("Before:\n")

print("n=", n)
print("k=", k)
print("lambda=", lamda)

print("After:\n")

print("n=", n1)
print("k=", k1)
print("lambda=", lamda1)

#plot_quartic(a_exact, image, iso_x, iso_y, P0)