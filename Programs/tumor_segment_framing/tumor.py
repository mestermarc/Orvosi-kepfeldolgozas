import math
from statistics import median, mode

import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import copy

from numpy import average


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
        self.description =" "
        self.probability = 0

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

    def get_desc(self):
        return self.description

    def add_desc(self, text):
        self.description += text

    def add_proba(self, weight):
        self.probability += weight

    def get_proba(self):
        return self.probability

    def getArea(self, num):
        return self.area[num]

    def getArea_array(self):
        return self.area

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

    def getArea(self, num):
        return self.area[num]

    def isIdenticalTumor(self, newcenterx, newcentery, TRESHOLD):
        if abs(newcenterx - self.centerx) < TRESHOLD and abs(newcentery - self.centery) < TRESHOLD:
            print("These tumors are identical!\n Centers are: x:{}, y:{}\n"
                  "Other centers are: x:{}, y:{}".format(self.centerx, self.centery, newcenterx, newcentery))
            return True
        else:
            # print("x:{}, y:{}  !!==  x2:{}, y2:{}".format(self.centerx, self.centery, newcenterx, newcentery))
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

    def calc_lenght(self):
        sus = False
        # calculated for mininmum 3 long slices
        areas = self.getArea_array()
        middle_element = self.getArea(int(len(areas) / 2))
        str = "id: {}#, areas len:{};\n" \
              "avg area:{}, firstarea:{}, lastarea:{}, middle element:{}\n" \
              "maxarea:{}, radius:{}\n" \
              " {}  <  {}" \
            .format(self.getId(), len(areas),
                    round(average(areas), 2), areas[0], areas[self.getLenght() - 1], middle_element,
                    max(areas), round(2 * getRadius(max(areas)), 2),
                    round(2 * getRadius(max(areas)), 2)/2, self.getLenght()+1)
        #TODO: implement condition: round(2 * getRadius(max(areas)), 2)/2 < self.getLenght()+1)
        if areas[0] < average(areas) and areas[self.getLenght() - 1] < average(areas) and average(areas) < max(areas):
            result = "Suspicious!"
        else:
            result = "Not suspicious!"
        print(str + result)
        return str + "\n" + result

    def tumor_lookalike(self):
        # calculated for mininmum 3 long slices
        areas = self.getArea_array()
        if areas[0] < average(areas) and areas[self.getLenght() - 1] < average(areas) and average(areas) < max(areas) and round(2 * getRadius(max(areas)), 2)/2 < self.getLenght()+1 and round(2 * getRadius(max(areas)), 2)/2 > self.getLenght()-2:
            return True
        else:
            return False

    def calculate_proba(self):
        #1
        if self.getLenght()>=3:
            self.add_desc("#1 This form is long enough{}\n".format(self.getLenght()))
            self.add_proba(0.1)

            # 2
            area_rate = self.calc_area_rate()
            self.add_proba(area_rate)
            self.add_desc("#2 Area / Circle drawn around {}%\n".format(area_rate))

            # 3
            areas = self.getArea_array()
            first_area = areas[0]
            last_area = areas[self.getLenght()-1]
            avg_area = average(areas)

            if first_area < avg_area and last_area < avg_area:
                self.add_proba(0.3)
                self.add_desc("#3 Avg is greater than first and last \n".format(area_rate))
            elif first_area < avg_area or last_area < avg_area:
                self.add_proba(0.15)
                self.add_desc("#3 Avg is greater than first OR last \n".format(area_rate))
            else:
                self.add_proba(-0.4)
                self.add_desc("#3 Avg is smaller than first and last \n".format(area_rate))

            #4
            diameter = 2 * getRadius(max(areas))
            if round( diameter/ 2, 2) < self.getLenght() + 1:
                self.add_proba(0.4)
                self.add_desc("Shape's diameter/2({}) < {} \n".format(diameter/2,self.getLenght() + 1))
            else:
                self.add_proba(-0.2)

            #5
            if round(diameter/ 2 ,2) > (self.getLenght()+1)/2:
                self.add_proba(0.4)
                self.add_desc("Shape's diameter/2({}) > {} \n".format(diameter/2,(self.getLenght()+1)/2))
            else:
                self.add_proba(-0.2)

            #6
            if(areas_growing(areas)):
                self.add_desc("#6 Areas growing to middle\n")
                self.add_proba(0.2)
            else:
                self.add_proba(-0.2)

            #6
            if (areas_decreasing(areas)):
                self.add_desc("#6 Areas decreasing from middle\n")
                self.add_proba(0.2)
            else:
                self.add_proba(-0.2)
            #7
            if (largest_in_middle(areas)):
                self.add_desc("#7 Largest area is in the middle!\n")
                self.add_proba(0.2)
            else:
                self.add_proba(-0.2)

            #8
            if (middle_ok(areas)):
                self.add_desc("#8 Middle areas are close enough!\n")
                self.add_proba(0.2)

        else:
            self.add_desc("This form isnt long enough{}\n".format(self.getLenght()))

        self.add_desc("This form's probability is:{}".format(self.get_proba()))

        if(self.get_proba()>1.5):
            return True
        else:
            return False


    def calc_area_rate(self):
        sum = 0
        for slice_num in range(0,self.getLenght()-1):
            sum += (self.area[slice_num] / self.frame_area[slice_num])
        return sum/self.getLenght()


def largest_in_middle(areas):
    if len(areas) % 2 == 0:
        element1 = areas[int(len(areas) / 2) - 1]      # middle element (smaller)
        element2 = areas[int(len(areas)/2)]
        if(max(areas)==max(element1,element2)):
            return True
    elif(max(areas) == areas[int(len(areas)/2)]):
        return True
    return False

def areas_growing(areas):
    if len(areas) % 2 == 0:
        togo = int(len(areas) / 2) - 1      # middle element (smaller)
    else:
        togo = int(len(areas) / 2)
    for elsok in range(0, togo):
        if(areas[elsok]>=areas[togo]):
            return False
    return True

def areas_decreasing(areas):
    togo = int(len(areas) / 2)       # middle element (greater)
    for utolsok in range(togo+1, len(areas)):
        print(areas[utolsok],"<=",areas[togo])
        if(areas[togo]<= areas[utolsok]):
            return False
    return True

def middle_ok(areas):
    if len(areas) % 2 == 0:
        element1 = areas[int(len(areas) / 2) - 1]      # middle element (smaller)
        element2 = areas[int(len(areas)/2)]
        if 0.65 < (element1/element2) < 1.5:
            return True
        else:
            return False
    else:
        return True

def findTumor(all_tumors, new_tumor: Tumor):
    length = 0
    if len(all_tumors) == 0:
        all_tumors.append(new_tumor)
        return length
    else:
        for tumor in all_tumors:
            if tumor.getStartIMG() != new_tumor.getStartIMG():
                # print("imgnum1:{},  imgnum2:{}".format(tumor.getStartIMG(), new_tumor.getStartIMG()))
                if tumor.isIdenticalTumor(new_tumor.centerx, new_tumor.centery, 5):
                    # tmptum = copy.deepcopy(Tumor(new_tumor))
                    # def addSlice(self, rect,mask, regionArea, frameArea, centerx, centery):
                    length = tumor.addSlice(new_tumor.getfirstRect(), new_tumor.getfirstMask(), new_tumor.getArea(0),
                                            new_tumor.getRectArea(), new_tumor.centerx, new_tumor.centery)
                    # if we found it, we can break the for loop
                    return length
        # the coordinates shows us no matches, so, it is a new tumor:
        new_tumor.setId(len(all_tumors) + 1)
        all_tumors.append(new_tumor)
        print("No match, added a new tumor! id:{}, imgnum:{}".format(new_tumor.getId(), new_tumor.getStartIMG()))
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
                # tumor.plot_onlyTumor(num)
                tumor.plot_Tumor(num)


def plot_sus(all_tumors):
    LOGGING_ENABLED = True
    for tumor in all_tumors:

        res = tumor.calc_lenght()

        fig = plt.figure(figsize=(50, 10))
        title = "Suspicious form: ID:{}, lenght:{}".format(tumor.getId(), tumor.getLenght()) + "\n" + res
        if LOGGING_ENABLED:
            print(title)
        fig.suptitle(title, fontsize=16)
        for num in range(0, tumor.getLenght()):
            ax = fig.add_subplot(1, 5, num + 1)
            plt.imshow(tumor.getMask(num), cmap='coolwarm')
            ax.title.set_text("#{}, area = {}".format(num, tumor.getArea(num)))
            ax.set_axis_off()
            # dot = mpatches.Circle((tumor.getcenterx(), tumor.getcentery()), 40, fill=None, edgecolor='red', linewidth=1)
            rt = tumor.getRect(num)
            ax.add_patch(rt)
        plt.show()

def plot_sus_proba(all_tumors):
    LOGGING_ENABLED = True
    for tumor in all_tumors:

        #res = tumor.calc_lenght()
        res = tumor.calculate_proba()

        if(res):
            fig = plt.figure(figsize=(50, 10))
            title = "Suspicious form: ID:{}, lenght:{}".format(tumor.getId(), tumor.getLenght()) + "\n" + tumor.get_desc()
            if LOGGING_ENABLED:
                print(title)
            fig.suptitle(title, fontsize=16)
            for num in range(0, tumor.getLenght()):
                ax = fig.add_subplot(1, 5, num + 1)
                plt.imshow(tumor.getMask(num), cmap='coolwarm')
                ax.title.set_text("#{}, area = {}".format(num, tumor.getArea(num)))
                ax.set_axis_off()
                # dot = mpatches.Circle((tumor.getcenterx(), tumor.getcentery()), 40, fill=None, edgecolor='red', linewidth=1)
                rt = tumor.getRect(num)
                ax.add_patch(rt)
            plt.show()

def getRadius(area):
    return math.sqrt(area / math.pi)
