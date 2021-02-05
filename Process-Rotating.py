from PIL import Image as img
from PIL import ImageFilter
from math import *

wide = 1920
high = 1080

LY = 9460730472580800

zoom = 300 * LY
naX = 1.920
naY = 1.080
Zedge = 1.920*zoom
XTo = (naX*2)*zoom#3.2e-5#
YTo = (naY*2)*zoom#1.8e-5#

BlackHoles = 0

Num = 0
jump = 1

def Matconv(A, B):
    res = [0, 0, 0]
    for a in range(3):
        for b in range(3):
            res[a] += B[a][b] * A[b]
    return res
        
def Rotating(Location, x, y, z):
    Rx = [[1, 0, 0],
          [0, cos(y), -sin(y)],
          [0, sin(y), cos(y)]]
    Ry = [[cos(x), 0, sin(x)],
          [0, 1, 0],
          [-sin(x), 0, cos(x)]]
    Rz = [[cos(z), -sin(z), 0],
          [sin(z), cos(z), 0],
          [0, 0, 1]]
    return Matconv(Matconv(Matconv(Location, Rx), Ry), Rz)
    
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
    rotatingDegree = Num / 1000
    [Xs, Ys, Zs] = PointGird
    for i in range(len(Xs)):
        xyz = [Xs[i], Ys[i], Zs[i]]
        xyz = Rotating(xyz, rotatingDegree, rotatingDegree/2, 0)
        [Xs[i], Ys[i], Zs[i]] = xyz
        
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
                Locations["[{}, {}]".format(display_X, display_Y)] = [128, 16, 16]
    return Locations

for Pg in open("data.data"):
    Num += 1
    if Num % jump == 0:
        Gird = eval(Pg) 
        print("{} : Drawing.".format(Num/jump), end="")
        Locations = Rendering(int(Num/jump), Gird)
        Draw(int(Num/jump), Locations)
        print("\t...Done!")
    else:
        continue
