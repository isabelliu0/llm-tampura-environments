from __future__ import annotations

import os

import pybullet as p

if __name__ == "__main__":
    directory = "./tampura/models/srl/ycb"
    for foldername in os.listdir(directory):
        path = os.path.join(directory, foldername)
        if "chips" in path:
            if os.path.isdir(path):
                print("Running VHACD on " + path)
                in_obj_file = path + "/google_16k/textured.obj"
                out_obj_file = path + "/google_16k/textured_vhacd.obj"
                log_file = path + "/google_16k/vhacd_log.txt"

                # Set up the options
                options = {
                    "resolution": 100000,
                    "depth": 20,
                    "concavity": 0.0025,
                    "planeDownsampling": 4,
                    "convexhullDownsampling": 4,
                    "alpha": 0.05,
                    "beta": 0.05,
                    "gamma": 0.0005,
                    "pca": 0,
                    "mode": 0,
                    "maxNumVerticesPerCH": 64,
                    "minVolumePerCH": 0.0001,
                }

                # Perform the decomposition
                p.vhacd(in_obj_file, out_obj_file, log_file, **options)
