from PIL import Image as img
from PIL import ImageFilter

wide = 1920
high = 1080

zoom = 3e8
naX = 1920
naY = 1080
Zedge = 1920*zoom
XTo = (naX*2)*zoom#3.2e-5#
YTo = (naY*2)*zoom#1.8e-5#

BlackHoles = 8

Num = 0
jump = 1

def Draw(Num, Locations):
    temp = img.new("RGB", (wide, high), (0, 0, 0))
    for Point in Locations.keys():
        [x, y] = eval(Point)
        color = Locations[Point]
        for i in range(3):
            color[i] = int(color[i])
        temp.putpixel([x, y], tuple(color))
    temp = temp.filter(ImageFilter.SMOOTH_MORE)
    temp.save("Pics/{}.png".format(Num))
    
def Rendering(Num, PointGird):
    [Xs, Ys, Zs] = PointGird
    Locations = dict({})
    for i in range(len(Xs)):
        if abs(Zs[i]) > Zedge:
            continue
        P_x = Xs[i]
        P_y = Ys[i]
        display_X = int(P_x * (wide / XTo) + (wide / 2))
        display_Y = int(P_y * (high / YTo) + (high / 2))
        if (display_X >= wide) or (display_Y >= high) or (display_X < 0) or (display_Y < 0):
            continue
        else:
            try:
                if Locations["[{}, {}]".format(display_X, display_Y)][0]*2 <= 256:
                    Locations["[{}, {}]".format(display_X, display_Y)][0] *= 2
                    
                elif Locations["[{}, {}]".format(display_X, display_Y)][1]*2 <= 256:
                    Locations["[{}, {}]".format(display_X, display_Y)][1] *= 2
                    Locations["[{}, {}]".format(display_X, display_Y)][0] = 255
                    
                elif Locations["[{}, {}]".format(display_X, display8)][2]*2 <= 256:
                    Locations["[{}, {}]".format(display_X, display_Y)][2] *= 2
                    Locations["[{}, {}]".format(display_X, display_Y)][0] = 255
                    Locations["[{}, {}]".format(display_X, display_Y)][1] = 255
                else:
                    Locations["[{}, {}]".format(display_X, display_Y)][0] = 255
                    Locations["[{}, {}]".format(display_X, display_Y)][1] = 255
                    Locations["[{}, {}]".format(display_X, display_Y)][2] = 255
            except:
                Locations["[{}, {}]".format(display_X, display_Y)] = [16, 16, 128]
    Draw(Num, Locations)

for Pg in open("data.data"):
    Num += 1
    if Num % jump == 0:
        Gird = eval(Pg) 
        print("{} : Drawing.".format(Num/jump), end="")
        Rendering(int(Num/jump), Gird)
        print("\t...Done!")
    else:
        continue
