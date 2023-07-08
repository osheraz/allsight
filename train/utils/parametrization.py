import numpy as np
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
import random

# Finger properties -->  x,  y, z,  cyl height , cyl radius
finger_props = [0, 0, 0, 0.016, 0.0125]

h = finger_props[3]
r = finger_props[4]
H = h + r


def radius(z):
    if z < h:
        return r
    else:
        return np.sqrt(r ** 2 - (z - h) ** 2)


def radius_dz(z):
    if z < h:
        return 0
    else:
        return h - z


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

z = np.linspace(0, H, 30)
q = np.linspace(0, 2 * np.pi, 30)

f = np.where(z < h, r,
             np.sqrt(r ** 2 - (z - h) ** 2)
             )

Z, Q = np.meshgrid(z, q)

X = f * np.cos(Q)
Y = f * np.sin(Q)

Nx = - f * np.cos(Q)
Ny = - f * np.sin(Q)

f_df = np.where(Z <= h,
                0,
                1 * (h - Z))
Nz = f_df

N = np.linalg.norm((Nx, Ny, Nz), axis=0) * 500
# ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)  #
# ax.plot_surface(X, Y, Z)
ax.scatter(X, Y, Z, c='r', marker='o')

# Plot normal.
ax.quiver(X, Y, Z, Nx / N, Ny / N, Nz / N, arrow_length_ratio=0.15, linewidth=1, color='red')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

A = 200
z = np.random.uniform(low=0, high=H, size=A)
phi = np.random.uniform(low=0, high=2 * np.pi, size=A)

for i, j in zip(z, phi):
    x = radius(i) * np.cos(j)
    y = radius(i) * np.sin(j)
    z = i
    nx = -radius(i) * np.cos(j)
    ny = -radius(i) * np.sin(j)
    nz = radius_dz(z)
    n = np.linalg.norm((nx, ny, nz), axis=0) * 500
    ax.quiver(x, y, z, nx / n, ny / n, nz / n,
              arrow_length_ratio=0.15, linewidth=1, color='blue')

fig.tight_layout()

plt.show()
