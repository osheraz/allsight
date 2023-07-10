import numpy as np
import pandas as pd
import os
from glob import glob
from random import randrange
import cv2
import matplotlib.pyplot as plt
from scipy import spatial
from src.allsight.train.utils.vis_utils import data_for_finger_parametrized

# Set the backend for matplotlib
plt.switch_backend('TkAgg')

# Get the username of the current user
pc_name = os.getlogin()

# TODO Define the LED and gel colors
leds = 'rrrgggbbb'
gel = 'clear'

# TODO Define the indenters
indenter = ['sphere4', 'sphere5', 'sphere3']

# TODO Define the paths for simulated and real data
sim_prefix = f'/home/{pc_name}/PycharmProjects/allsight_sim/experiments'
real_prefix = f'/home/{pc_name}/catkin_ws/src/allsight'

sim_paths = [f"{sim_prefix}/dataset/{leds}/data/{ind}" for ind in indenter]
real_paths = [f"{real_prefix}/dataset/{gel}/{leds}/data/{ind}" for ind in indenter]

buffer_sim_paths, buffer_real_paths = [], []

# Get file paths for simulated data
for p in sim_paths:
    buffer_sim_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*.json'))]

# Get file paths for real data
for p in real_paths:
    buffer_real_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*.json'))]
buffer_real_paths = [p for p in buffer_real_paths if ('transformed_annotated' in p)]

# Load real data into a DataFrame
df_data_real = pd.concat([pd.read_json(p).transpose() for p in buffer_real_paths], axis=0)

# Load simulated data into a DataFrame
df_data_sim = pd.concat([pd.read_json(p).transpose() for p in buffer_sim_paths], axis=0)

# Filter non-touching samples from real data
df_data_real = df_data_real[df_data_real.time > 2.0]
df_data_real = df_data_real.sample(n=2000)

# Convert pose data to numpy arrays
pose_real = np.array([df_data_real.iloc[idx].pose_transformed[0][:3] for idx in range(df_data_real.shape[0])])
pose_sim = np.array([df_data_sim.iloc[idx].pose[0] for idx in range(df_data_sim.shape[0])])

# Create a 3D plot
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

# Plot a cylinder and a sphere
Xc, Yc, Zc = data_for_finger_parametrized(h=0.016, r=0.0128)
ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')

# Create a KDTree for real poses
tree = spatial.KDTree(pose_real)

# Perform visualization loop
for i in range(100):
    sample = randrange(len(pose_sim))

    sim_xyz = pose_sim[sample]
    sim_image = (cv2.imread(sim_prefix + df_data_sim['frame'][sample])).astype(np.uint8)

    cv2.imshow('sim', sim_image)
    ax.scatter(sim_xyz[0], sim_xyz[1], sim_xyz[2], c='black', label='sim')

    _, ind = tree.query(sim_xyz)
    real_xyz = pose_real[ind]

    real_image = (cv2.imread(df_data_real['frame'][ind])).astype(np.uint8)
    cp = df_data_real['contact_px'][ind]
    real_image = cv2.circle(np.array(real_image), (int(cp[0]), int(cp[1])), int(cp[2]), (255, 255, 255), 1)

    cv2.imshow('real', real_image)
    ax.scatter(real_xyz[0], real_xyz[1], real_xyz[2], c='red', label='true')

    print(f'real {real_xyz}\nsim {sim_xyz}\nnorm {np.linalg.norm((real_xyz, sim_xyz)) * 1000}')

    ax.set_xlim((-0.014, 0.014))
    ax.set_ylim((-0.014, 0.014))
    ax.set_zlim((0.0, 0.03))

    plt.pause(0.001)
    cv2.waitKey(1) & 0xff
