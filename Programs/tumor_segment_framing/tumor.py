import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import copy


class Tumor:
    def __init__(self, rectangle, mask, regionArea, frameArea, centerx, centery):
        super().__init__()
        self.rectangles = []
        self.len = 1
        self.area = []
        self.frame_area = []
        self.centerx = centerx
        self.centery = centery
        self.masks = []
        self.id = 1
        self.imgnum = 0

        self.rectangles.append(rectangle)
        self.masks.append(mask)
        self.area.append(regionArea)
        self.frame_area.append(frameArea)

    def addSlice(self, rect, mask, regionArea, frameArea, centerx: int, centery: int):
        self.len += 1
        self.rectangles.append(rect)
        self.masks.append(mask)
        self.area.append(regionArea)
        self.frame_area.append(frameArea)
        self.centerx = centerx
        self.centery = centery
        print("Slice added!")
        return self.len

    # TODO delete disappeared slices

    def getLenght(self):
        return self.len

    def setId(self, id):
        self.id = id

    def getId(self):
        return self.id

    def setStartimg(self, startImg):
        self.imgnum = startImg

    def getStartIMG(self):
        return self.imgnum

    def getcenterx(self):
        return self.centerx

    def getcentery(self):
        return self.centery

    def getfirstRect(self):
        return self.rectangles[0]

    def getfirstMask(self):
        return self.masks[0]

    def getMask(self, num):
        return self.masks[num]

    def getRect(self, num):
        return self.rectangles[num]

    def getRectArea(self):
        return self.frame_area

    def getArea(self):
        return self.area

    def isIdenticalTumor(self, newcenterx, newcentery, TRESHOLD):
        if abs(newcenterx - self.centerx) < TRESHOLD and abs(newcentery - self.centery) < TRESHOLD:
            print("These tumors are identical!\n Centers are: x:{}, y:{}\n"
                  "Other centers are: x:{}, y:{}".format(self.centerx, self.centery, newcenterx, newcentery))
            return True
        else:
            #print("x:{}, y:{}  !!==  x2:{}, y2:{}".format(self.centerx, self.centery, newcenterx, newcentery))
            return False

    def plot_Tumor(self, num):
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.imshow(self.masks[num])

        dot = mpatches.Circle((self.centerx, self.centery), 40, fill=None, edgecolor='red', linewidth=1)
        ax2.add_patch(dot)
        ax2.set_axis_off()
        plt.tight_layout()
        plt.show()

    def plot_Tumor_orig(self, orig):
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.imshow(orig)

        dot = mpatches.Circle((self.centerx, self.centery), 40, fill=None, edgecolor='red', linewidth=1)
        ax2.add_patch(dot)
        ax2.set_axis_off()
        plt.tight_layout()
        plt.show()

    def plot_onlyTumor(self, num):
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        coords = self.rectangles[num].get_xy()
        width = self.rectangles[num].get_width()

        x = coords[0]
        y = coords[1]
        ax3.imshow(self.masks[num][y:y + width, x:x + width], cmap="bwr")  # crop_img = img[y:y+h, x:x+w
        # ax3.imshow(self.masks[0][340:419, 280:353])
        plt.show()


def findTumor(all_tumors, new_tumor: Tumor):
    length = 0
    if len(all_tumors) == 0:
        all_tumors.append(new_tumor)
        return length
    else:
        for tumor in all_tumors:
            if tumor.getStartIMG() != new_tumor.getStartIMG():
                #print("imgnum1:{},  imgnum2:{}".format(tumor.getStartIMG(), new_tumor.getStartIMG()))
                if tumor.isIdenticalTumor(new_tumor.centerx, new_tumor.centery, 50):
                    # tmptum = copy.deepcopy(Tumor(new_tumor))
                    # def addSlice(self, rect,mask, regionArea, frameArea, centerx, centery):
                    length = tumor.addSlice(new_tumor.getfirstRect(), new_tumor.getfirstMask(), new_tumor.getArea(),
                                   new_tumor.getRectArea(), new_tumor.centerx, new_tumor.centery)
                    # if we found it, we can break the for loop
                    return length
        #the coordinates shows us no matches, so, it is a new tumor:
        new_tumor.setId(len(all_tumors)+1)
        all_tumors.append(new_tumor)
        print("No match, added a new tumor! id:{}, imgnum:{}".format(new_tumor.getId(),new_tumor.getStartIMG()))
        return length


def plot_all(all_tumors, MASK, orig):
    for tumor in all_tumors:
        tumor.plot_Tumor()


def plot_all_orig(all_tumors, orig):
    for tumor in all_tumors:
        tumor.plot_Tumor_orig(orig)


def plot_all_sus(all_tumors, all):
    for tumor in all_tumors:
        if (all):
            for num in range(0, tumor.getLenght()):
                tumor.plot_onlyTumor(num)

        elif tumor.getLenght() > 2:
            for num in range(0, tumor.getLenght()):
                #tumor.plot_onlyTumor(num)
                tumor.plot_Tumor(num)

def plot_sus(all_tumors):
    for tumor in all_tumors:
        if(tumor.getLenght()>2):
            fig = plt.figure(figsize=(50, 50))
            for num in range(0, tumor.getLenght()):
                ax = fig.add_subplot(2, 5, num+1)
                plt.imshow(tumor.getMask(num), cmap="bone")
                ax.title.set_text(num)
                ax.set_axis_off()
                #dot = mpatches.Circle((tumor.getcenterx(), tumor.getcentery()), 40, fill=None, edgecolor='red', linewidth=1)
                rt = tumor.getRect(num)
                ax.add_patch(rt)
            plt.show()
