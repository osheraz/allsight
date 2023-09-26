<h1 align="center"><img src=".github/logo.svg"
  width=45px>
  AllSight
</h1>
<h2 align="center">
  AllSight: A Low-Cost and High-Resolution Round Tactile Sensor with Zero-Shot Learning Capability
</h2>


<h4 align="center">
  <a href="https://arxiv.org/abs/2307.02928"><b>Paper</b></a> &nbsp;•&nbsp;
  <a href="https://github.com/osheraz/allsight_sim"><b>AllSight-Sim</b></a> &nbsp;•&nbsp; 
  <a href="https://github.com/osheraz/allsight_design"><b>AllSight-Design</b></a> &nbsp;•&nbsp;
  <a href="https://github.com/osheraz/allsight_dataset"><b>AllSight-Dataset</b></a>
</h4>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

This page provides instructions, datasets, and source code for working with AllSight tactile sensor. 
AllSight, is an optical tactile sensor with a round 3D structure,
potentially designed for robotic in-hand manipulation tasks. 

<div align="center">
  <img src=".github/allsight.gif"
  width="80%">
</div>

## Installation

### 1. Clone repository

```bash
git clone https://github.com/osheraz/allsight
cd allsight
```
### 2. Download AllSight-dataset (Optional)
```bash
git clone https://github.com/osheraz/allsight_dataset
cd ..
```
### 3. Download TACTO sim package (Optional)
```bash
cd simulation
git clone https://github.com/osheraz/allsight_sim
cd ..
```
Follow the instruction inside to setup the simulated environment
### 4. Install the dependencies
```bash
pip install -r requirements.txt
```

## Folder structure
```bash
├── train                  # training scripts 
    ├── utils              # helper functions and classes
├── tactile_finger         # ROS1 package for streaming feedback from AllSight
    ├── config             # config files
    ├── launch             # envrionment launch files
    ├── src                # data collection scripts
        ├── envs           # helper functions and classes
├── simulation             # training simulation scripts 
    ├── allsight_sim       # AllSight TACTO sim package
├── allsight_dataset       # dataset and preprocessing scripts
```

## Sensor Setup
For easy plug-and-play support, burn the following [image](https://github.com/showmewebcam/showmewebcam/releases).
to the Raspberry Pi. This firmware transforms the Raspberry Pi into a high-quality webcam. It works reliably, boots quickly, and gets out of your way.
(Credits to [show-me-the-webcam](https://github.com/showmewebcam/showmewebcam)).

#### Next, it is suggested to adjust camera settings to constant values:
Open a new terminal and enter to the debugging interface:
```
$ ls -l /dev/ttyACM*
crw-rw---- 1 root dialout 166, 0 sep 25 14:03 /dev/ttyACM0
$ sudo screen /dev/ttyACM0 115200
```
Within that interface execute:
```bash
/usr/bin/camera-ctl
```
Save the following values:
<div align="center">
  <img src=".github/vals.png"
  width="60%">
</div>

## Scripts
- [tactile_finger/src/envs/finger.py](tactile_finger/src/envs/finger.py): AllSight interface.
- [inference.py](inference.py): live inference script.
- [offline_inference.py](offline_inference.py): offline inference via dataset.
- [hand_inference.py](hand_inference.py): live inference script.
- [train](train/): training scripts.

## Bibtex

```
@misc{azulay2023allsight,
      title={AllSight: A Low-Cost and High-Resolution Round Tactile Sensor with Zero-Shot Learning Capability}, 
      author={Osher Azulay and Nimrod Curtis and Rotem Sokolovsky and Guy Levitski and Daniel Slomovik and Guy Lilling and Avishai Sintov},
      year={2023},
      eprint={2307.02928},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```