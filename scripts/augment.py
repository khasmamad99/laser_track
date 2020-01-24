import numpy as np
import random
import cv2
import glob

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

overlays_paths = ["red_small.png", "green_white.png", "green_black.png"]
bgs = glob.glob("downloads/shooting target/*.jpg")

overlays = [cv2.imread(overlay, -1) for overlay in overlays_paths]

bg_id = 100001

for bg in bgs:
    bg = cv2.imread(bg)
    h, w, _ = bg.shape
    text = ""
    for overlay in overlays:
        overlay_h, overlay_w, _ = overlay.shape
        for i in range(3):
            scale = min(random.uniform(20/1920, 120/1920), random.uniform(20/1920, 120/1920), random.uniform(20/1920, 120/1920))
            overlay_size = int(max(h, w) * scale)
            print(overlay_size)
            overlay = cv2.resize(overlay, (overlay_size, overlay_size))
            w_offset = random.randrange(0, w - overlay_size)
            h_offset = random.randrange(0, h - overlay_size)
            x1, x2 = w_offset, w_offset + overlay_size
            y1, y2 = h_offset, h_offset + overlay_size

            alpha_s = overlay[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                bg[y1:y2, x1:x2, c] = (alpha_s * overlay[:, :, c] +
                                        alpha_l * bg[y1:y2, x1:x2, c])

            box = convert((w, h), (x1, x2, y1, y2))
            text = text + "1 " + " ".join([str(a) for a in box]) + "\n"

    cv2.imwrite("aug_data/" + str(bg_id) + ".jpg", bg)
    open("aug_data/" + str(bg_id) + ".txt", "w").write(text)
    bg_id += 1







    


