class Tumor:
    def __init__(self, rectangle, regionArea, frameArea, centerx, centery):
        super().__init__()
        self.rectangles = []
        self.len = 1
        self.area = []
        self.frame_area = []
        self.centerx = centerx
        self.centery = centery

        self.rectangles.append(rectangle)
        self.area.append(regionArea)
        self.frame_area.append(frameArea)
        print("Tumor created!")
        print(self.len)

    def addSlice(self, rect, regionArea, frameArea, centerx, centery):
        self.len += 1
        self.rectangles.append(rect)
        self.area.append(regionArea)
        self.frame_area.append(frameArea)
        self.centerx = centerx
        self.centery = centery

    # TODO delete disappeared slices

    def getLenght(self):
        return self.len

    def getcenterx(self):
        return self.centerx

    def getcentery(self):
        return self.centery

    def isIdenticalTumor(self, newcenterx, newcentery, TRESHOLD):
        if abs(newcenterx - self.centerx) < TRESHOLD and abs(newcentery - self.centery) < TRESHOLD:
            print("These tumors are identical!")
            return True
        else:
            return False


def findTumor(all_tumors, new_tumor):
    if len(all_tumors) == 0:
        print("len is 0!")
        all_tumors.append(new_tumor)
    else:

        for tumor in all_tumors:
            print("getcent:{}".format(new_tumor.centerx))
            if tumor.isIdenticalTumor(new_tumor.centerx, new_tumor.centery, 100):
                print("meh")
                # tumor.addSlice(rect, regionArea, frameArea, newcenterx, newcentery)
                break
            else:
                all_tumors.append(new_tumor)
                break
