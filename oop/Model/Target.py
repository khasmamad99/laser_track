import numpy as np
import json
import cv2


def calc_score_circlular_1(target, x, y):
    for i, cont in enumerate(target.conts):
        test = cv2.pointPolygonTest(cont, (x, y), True)
        if test >= 0:
            return 10 - i
    return 0


def calc_score_human_1(target, x, y):
    for i, cont in enumerate(target.conts):
        test = cv2.pointPolygonTest(cont, (x, y), True)
        if test >= 0:
            return "Success"
    return "Fail"



class Target:

    def __init__(self, json_file, pixel_size):
        name, img_path, conts_path, feats_path, center_coords, real_size, pixel_size
        attr_dict = json.load(json_file)
        self.name = attr_dict["name"]
        self.img_path =  attr_dict["img_path"]
        self.img = cv2.imread(attr_dict["img_path"])
        self.conts = np.load(attr_dict["conts_path"], allow_pickle=True)
        self.feats = np.load(attr_dict["feats_path"], allow_pickle=True)
        self.center_coords = attr_dict["center_coords"]
        self.real_size =  attr_dict["real_size"]
        self.pixel_size = pixel_size
        self.calibration_offset = [0, 0]
        self.set_score_calculator()


    def set_score_calculator(self, x, y, func=None):
        if func:
            self.calc_score = func
        else:
            if self.name == "circular_1":
                self.calc_score = calc_score_circlular1(self, x, y)
            elif self.name == "human_1":
                self.calc_score = calc_score_human_1(self, x, y)

    
    def calc_score(self, x, y):
        return self.calc_score(self, x, y)


    def calc_distance(self, pts):
        dist = 0
        start = False
        start_time = None
        prev_coords = None
        for pt in reversed(pts):
            coords, is_shot, time = [pt.x, pt.y], pt.is_shot, pt.time
            if is_circle and not start:
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
        scale = real_area / pixel_area
        real_dist = dist * scale
        return real_dist

    
    def recalibrate(self, x, y):
        self.calibration_offset = [i - j for i, j in zip([x, y], self.center_coords)]