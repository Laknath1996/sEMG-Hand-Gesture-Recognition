# Real-Time Hand Gesture Recognition with Temporal Muscle Activation Maps

<p align="center">
  <img src="https://github.com/Laknath1996/Real-Time-Hand-Gesture-Recognition-with-TMA-Maps/blob/master/figures/Amap_Final.png" width="512" height="362">
</p>

This repository containes the source code for the real-time hand gesture recognition algorithm based on Temporal Muscle Activation (TMA) maps of multi-channel surface electromyography (sEMG) signals, which was published in the ICASSP 2020 paper [Real-Time Hand Gesture Recognition Using Temporal Muscle Activation Maps of Multi-Channel Semg Signals](10.1109/ICASSP40776.2020.9054227). An application of this work -- wearable wireless dry contact sEMG sensor system for controlling digital technologies -- received an honorable mention (ranked among top 15 projects in the world) at the IEEE Communication Society Student Competition 2019.

Currently, the following video demonstrations of the work are available. Please note that as of 2020/04/03, the demonstration and evaluations were performed using only the **MyoArmband** device by Thalamic Labs, Canada. We would update demonstrations as we extend the software to other sEMG acquisition devices.

| Device | Real-Time Difference Signal d(n) | Real-Time Hand Gesture Recognition |
|---|:---:|:---:|
| MyoArm Band |[<img src="https://i.imgur.com/TClPwXT.png" width="100%">](https://drive.google.com/file/d/15mN4JwVRRL3TTHlOhGMS2ip6zGMr1LJJ/view?usp=sharing "Video Title") | [<img src="https://i.imgur.com/gxEQ8Sp.png" width="100%">](https://drive.google.com/file/d/1Yd8oEFTagi1tzy1vNjLxUetrtBJKHM6B/view?usp=sharing "Video Title")|
| sEMG Dry Contact Electrodes proposed by Naim et al. [1]| [<img src="https://i.imgur.com/htv288P.png" width="100%">](https://drive.google.com/open?id=1kvWtg2dxUsKVR4OwtHWpW691aTnte2RE "Video Title") | [<img src="https://i.imgur.com/htv288P.png" width="100%">](https://drive.google.com/open?id=1kvWtg2dxUsKVR4OwtHWpW691aTnte2RE "Video Title") |

Please note that robotic hand demonstrated in the above videos are build up using the code from [this](https://github.com/OpenBionics/Prosthetic-Hands/tree/master/Kinematics/MATLAB) repository. 

If you use this code/paper for your research, please cite the following paper:

```
@INPROCEEDINGS{9054227,  
author={A. D. {Silva} and M. V. {Perera} and K. {Wickramasinghe} and A. M. {Naim} and T. {Dulantha Lalitharatne} and S. L. {Kappel}},  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
title={Real-Time Hand Gesture Recognition Using Temporal Muscle Activation Maps of Multi-Channel Semg Signals},   
year={2020},  
volume={},  
number={},  
pages={1299-1303},}
```

## Installation Guide

### Step 1 : Clone the Repository

Clone the repository using the following command.

````
$ git clone https://github.com/Laknath1996/Real-Time-Hand-Gesture-Recognition-with-TMA-Maps.git
````

### Step 2 : Install Dependencies

**Note** : It is recommended to use a conda environment with Python 3.6 with this code. Before running the commands in this guide, make sure you activate the environment using `$ source activate <name of the env>`

The use the `requirements.txt` file given in the repository to install the dependencies via `pip`.

````
$ cd Real-Time\ Hand\ Gesture\ Recognition\ with\ TMA\ Maps/
$ pip install -r requirements.txt 
````
Note that `myo-python` provides classes, methods, etc. to access the MyoArm band device. Also, download the [myo-sdk](https://support.getmyo.com/hc/en-us/articles/360018409792-Myo-Connect-SDK-and-firmware-downloads) according to your relevant OS and keep it inside the the project root folder. 

### Step 3 : Verify the installation of dependencies

To verify whether `tensorflow`, `keras` and `myo-python` were installed properly, run the following.
````
$ python
>>> import tensorflow
>>> import keras
>>> import myo
````
If there are no error messages upon importing the above dependencies, it would indicate that the they are correctly installed. 

Now you can start using the code to recognize hand gestures in real-time using the TMA maps of the mulit-channel sEMG signal from the MyoArm band.

## Execution

`notebooks/gesture_recog_tma.ipynb` describes the execution steps of the real-time hand gesture recognition pipeline in detail.

## Notes

As of 2020/04/03, the above installation and execution steps are only tested on MacOS 10.14.6. We will update as soon as we test the installation and execution steps on Linux and Windows.

## Authors

* **Ashwin De Silva**
* **Malsha Perera** 

In addition, Asma Naim and Kithmin Wickremasinghe participated this project. Dr. Thilina Lalitharatne and Dr. Simon Kappel supervised the work.  

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Bionics Laboratory, Dept. of Mechanical Eng., University of Moratuwa, Sri Lanka.
* Dept. of Electronic and Telecommunication Eng., University of Moratuwa, Sri Lanka.

## References

[1] A. M. Naim, K. Wickramasinghe, A. De Silva, M. V. Perera, T. Dulantha Lalitharatne, and S. L. Kappel, ”Low-Cost Active Dry-Contact Surface EMG Sensor for Bionic Arms,” submitted to 2020 IEEE International Conference on Systems, Man and Cybernetics (SMC)
