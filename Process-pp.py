from PIL import Image as img
from PIL import ImageFilter
import pp

ppservers = ()
job_server = pp.Server(4, ppservers=ppservers, secret="1222",)

LY = 9460730472580800

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
JobPerRange = 64
JobPerCPU = 16

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
    
def Rendering(Num, Pgs, Zedge, wide, high, XTo, YTo, JobPerCPU):
    Locations = []
    for ADDindex in range(JobPerCPU):
        Pg = Pgs[ADDindex]
        PointGird = eval(Pg)
        [Xs, Ys, Zs] = PointGird
        Location = dict({})
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
                    if Location["[{}, {}]".format(display_X, display_Y)][0]*2 <= 256:
                        Location["[{}, {}]".format(display_X, display_Y)][0] *= 2
                    
                    elif Location["[{}, {}]".format(display_X, display_Y)][1]*2 <= 256:
                        Location["[{}, {}]".format(display_X, display_Y)][1] *= 2
                        Location["[{}, {}]".format(display_X, display_Y)][0] = 255
                    
                    elif Location["[{}, {}]".format(display_X, display8)][2]*2 <= 256:
                        Location["[{}, {}]".format(display_X, display_Y)][2] *= 2
                        Location["[{}, {}]".format(display_X, display_Y)][0] = 255
                        Location["[{}, {}]".format(display_X, display_Y)][1] = 255
                    else:
                        Location["[{}, {}]".format(display_X, display_Y)][0] = 255
                        Location["[{}, {}]".format(display_X, display_Y)][1] = 255
                        Location["[{}, {}]".format(display_X, display_Y)][2] = 255
                except:
                    Location["[{}, {}]".format(display_X, display_Y)] = [16, 16, 128]
        Locations += [(Location, Num+ADDindex, )]
    return Locations

Pgs = []
jobs = []
for Pg in open("data.data"):
    Num += 1
    if Num % jump == 0:
        Pgs += [Pg]
        if Num % JobPerCPU == 0:
            #print(len(Pgs))
            jobs += [(job_server.submit(Rendering, (int(Num/jump)-JobPerCPU, Pgs, Zedge, wide, high, XTo, YTo, JobPerCPU, ), (), ()), int(Num/jump)-JobPerCPU,)]
            print("Preparing tasks : {} to {}\r".format(int(Num/jump)-JobPerCPU, int(Num/jump)-1), end="")
            Pgs = []
        if Num % JobPerRange == 0:
            print("\nStart drawing with {} tasks".format(JobPerRange))
            Locations = []
            for job, StartIndex in jobs:
                print("Rendering {} to {}...".format(StartIndex+1, StartIndex+JobPerCPU))
                Locations += job()
                #print(len(Locations))
            for Location, number in Locations:
                print("\t{} : Drawing.".format(number+1), end="")
                Draw(number+1, Location)
                print("\t...Done!")
            Locations = []
            jobs = []
