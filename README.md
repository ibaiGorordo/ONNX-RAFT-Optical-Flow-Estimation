# ONNX-RAFT-Optical-Flow-Estimation
 Python scripts for performing optical flow estimation using the RAFT model in ONNX

![RAFT Optical flow estimation ONNX eagle](https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/blob/main/doc/img/eagle.gif)

*Original video:https://youtu.be/3wdsE1UgP6k*

# Requirements

 * Check the **requirements.txt** file. Additionally, **pafy** and **youtube-dl** are required for youtube video inference.
 
# Installation
```
pip install -r requirements.txt
pip install pafy youtube_dl>=2021.12.17
```

# ONNX model
The original models were converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309), download the models from [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/252_RAFT) and save them into the **[models](https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/tree/main/models)** folder. 

# Original Pytorch model
The Pytorch pretrained model was taken from the [original repository](https://github.com/princeton-vl/RAFT).
 
# Examples

 * **Image inference**:
 
 ```
 python image_flow_estimation.py
 ```
 
  * **Video inference**:
 
 ```
 python video_flow_estimation.py
 ```
 
 * **Webcam inference**:
 
 ```
 python webcam_flow_estimation.py
 ```
 
# Inference video Examples: https://youtu.be/GNwyuhYu7ZI

## Cheetah
![RAFT Optical flow estimation ONNX cheetah](https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/blob/main/doc/img/cheetah.gif)

*Original video: https://youtu.be/-KheqfpUpr0*

## Water drop
![RAFT Optical flow estimation ONNX water drop](https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/blob/main/doc/img/water_drop.gif)

*Original video: https://youtu.be/gS_tU6chC4A*

## Plant
![RAFT Optical flow estimation ONNX plant](https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/blob/main/doc/img/plant.gif)

*Original video: https://youtu.be/cNr_cttSf4U*

## Blink
![RAFT Optical flow estimation ONNX blink](https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/blob/main/doc/img/eye_blink.gif)

*Original video: https://youtu.be/lkJ4p__ZByU*

## Fall
![RAFT Optical flow estimation ONNX fall](https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/blob/main/doc/img/fall.gif)

*Original video: https://youtu.be/RKTXn_c2tyQ*

## Baseball player
![RAFT Optical flow estimation ONNX baseball player](https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/blob/main/doc/img/baseball.gif)

*Original video: https://youtu.be/t_vEJu3jmpw*

## Dance
![RAFT Optical flow estimation ONNX dance](https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/blob/main/doc/img/dance.gif)

*Original video: https://youtu.be/1WIA6Yvj8Yg*

# References:
* RAFT model: https://github.com/princeton-vl/RAFT
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* OpticalFlowToolkit toolkit: https://github.com/liruoteng/OpticalFlowToolkit
* Original paper: https://arxiv.org/abs/2003.12039
 
