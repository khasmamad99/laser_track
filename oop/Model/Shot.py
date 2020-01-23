class Shot:
    def __init__(self, target_img_path, coords=None, score=None):
        self.score = score
        self.coords = coords
        self.target_img_path = target_img_path



class TrackShot(Shot):
    def __init__(self, target_img_path, coords=None, pts=[], score=None, distance=None):
        Shot.__init__(self, score, coords)
        self.pts = pts
        self.distance = distance

    
    def __str__(self):
        return "Score: {}  Distance: {}".format(self.score, self.distance)



class SingleShot(Shot):
    def __str__(self):
        return "Score: {}".format(self.score)



class Point:
    def __init__(self, coords, time, is_shot=False):
        self.coords = coords
        self.time = time
        self.is_shot = is_shot