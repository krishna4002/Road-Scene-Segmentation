#!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="NvPiCMXIqZ6Uvw8244vQ")
project = rf.workspace("jay-ngdms").project("pothole-detection-system-rnoh4-xjw6w")
version = project.version(1)
dataset = version.download("coco-segmentation")
                