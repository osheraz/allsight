import numpy as np
import pandas as pd
import os
from glob import glob
from random import randrange
from matplotlib import pyplot as plt
import cv2
from src.allsight.train.utils.vis_utils import data_for_cylinder_along_z, data_for_sphere_along_z
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

from scipy import spatial

# change depend on the data you use
pc_name = os.getlogin()
leds = 'rrrgggbbb'
gel = 'clear'

indenter = ['sphere4', 'sphere5', 'sphere3']

sim_prefix = f'/home/{pc_name}/PycharmProjects/allsight_sim/experiments/'
sim_paths = [f"{sim_prefix}/dataset/{leds}/data/{ind}" for ind in indenter]
real_paths = [f"/home/{pc_name}/catkin_ws/src/allsight/dataset/{gel}/{leds}/data/{ind}" for ind in indenter]

buffer_sim_paths, buffer_real_paths = [], []

for p in sim_paths:
    buffer_sim_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*.json'))]
for p in real_paths:
    buffer_real_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*.json'))]
buffer_real_paths = [p for p in buffer_real_paths if ('transformed_annotated' in p)]

for idx, p in enumerate(buffer_real_paths):
    if idx == 0:
        df_data_real = pd.read_json(p).transpose()
    else:
        df_data_real = pd.concat([df_data_real, pd.read_json(p).transpose()], axis=0)

for idx, p in enumerate(buffer_sim_paths):
    if idx == 0:
        df_data_sim = pd.read_json(p).transpose()
    else:
        df_data_sim = pd.concat([df_data_sim, pd.read_json(p).transpose()], axis=0)


df_data_real = df_data_real[df_data_real.time > 2.0]  # only over touching samples!
df_data_real = df_data_real.sample(n=2000)

pose_real = np.array([df_data_real.iloc[idx].pose_transformed[0][:3] for idx in range(df_data_real.shape[0])])
# df_data_sim = df_data_sim[np.linalg.norm(df_data_sim.pose[0] > pose_real.max(axis=0)[-1]]

pose_sim = np.array([df_data_sim.iloc[idx].pose[0] for idx in range(df_data_sim.shape[0])])
# pose_sim = pose_sim[pose_sim[:, -1] < pose_real.max(axis=0)[-1]]

fig = plt.figure(figsize=(6, 5))

ax = fig.add_subplot(111, projection='3d')

Xc, Yc, Zc = data_for_cylinder_along_z(0., 0., 0.012, 0.016)
ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')
Xc, Yc, Zc = data_for_sphere_along_z(0., 0., 0.012, 0.016)
ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')

tree = spatial.KDTree(pose_real)

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
    # real_image = cv2.circle(np.array(real_image), (int(cp[0]), int(cp[1])), int(cp[2]), (255, 255, 255), 1)

    cv2.imshow('real', real_image)
    ax.scatter(real_xyz[0], real_xyz[1], real_xyz[2], c='red', label='true')

    print(f'real {real_xyz}\nsim {sim_xyz}\nnorm {np.linalg.norm((real_xyz, sim_xyz)) * 1000}')

    ax.set_xlim((-0.014, 0.014))
    ax.set_ylim((-0.014, 0.014))
    ax.set_zlim((0.0, 0.03))

    plt.pause(0.001)
    wait=1000 if i == 0  else 1000
    cv2.waitKey(wait) & 0xff
