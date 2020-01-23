import math
import numpy as np
import json
import cv2


class Target:

    def __init__(self, target_dict):
        self.name = target_dict["name"]
        self.img_path =  target_dict["img_path"]
        self.img = cv2.imread(target_dict["img_path"])
        self.conts = np.load(target_dict["conts_path"], allow_pickle=True)
        self.feats = np.load(target_dict["feats_path"], allow_pickle=True)
        self.center_coords = target_dict["center_coords"]
        self.real_size =  target_dict["real_size"]
        self.pixel_size = target_dict["pixel_size"]
        self.calibration_offset = [0, 0]
        self.set_score_calculator()


    def set_score_calculator(self, func=None):
        if func:
            self.calc_score = func
        else:
            from oop.Model.ScoreCalculator import ScoreCalculator
            method_list = [func for func in dir(ScoreCalculator) if callable(getattr(ScoreCalculator, func)) and not func.startswith("__")]
            for method in method_list:
                if self.name in method:
                    self.calc_score = getattr(ScoreCalculator, method)
                    break
        

    def calc_distance(self, pts):
        dist = 0
        start = False
        start_time = None
        prev_coords = None
        for pt in reversed(pts):
            coords, is_shot, time = pt.coords, pt.is_shot, pt.time
            if is_shot and not start:
                start = True
                prev_coords = coords
                start_time = time
            if start and (time - start_time < 3) and prev_coords is not None and coords is not None:
                dist += math.sqrt(math.pow((coords[0] - prev_coords[0]),
                                        2) + math.pow((coords[1] - prev_coords[1]), 2))
            prev_coords = coords

        return pixel2mm(dist)


    def pixel2mm(self, pixel_distance):
        real_w, real_h = self.real_size
        pixel_w, pixel_h = self.pixel_size
        real_area = real_w * real_h
        pixel_area = pixel_h * pixel_w
        scale = math.sqrt(real_area / pixel_area)
        real_dist = pixel_distance * scale
        return real_dist

    
    def recalibrate(self, x, y):
        self.calibration_offset = [i - j for i, j in zip([x, y], self.center_coords)]

    
    def find_center(self, size):
        # finds the center of the resized target
        center_x, center_y = self.center_coords
        org_w, org_h = self.img.shape[1], self.img.shape[0]
        new_w, new_h = size
        scale = min(new_w/org_w, new_h/org_h)
        nw = int(org_w*scale)
        nh = int(org_h*scale)
        new_center_x = int(center_x*scale + (new_w-nw)/2)
        new_center_y = int(center_y*scale + (new_h - nh)/2)

        return [new_center_x, new_center_y]