class ScoreCalculator:

    def calc_score_circlular_1(self, target, x, y):
        for i, cont in enumerate(target.conts):
            test = cv2.pointPolygonTest(cont, (x, y), True)
            if test >= 0:
                return 10 - i
        return 0

    def calc_score_human_1(self, target, x, y):
        for i, cont in enumerate(target.conts):
            test = cv2.pointPolygonTest(cont, (x, y), True)
            if test >= 0:
                return "Success"
        return "Fail"