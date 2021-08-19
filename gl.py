import struct
import random
from model import Obj
from collections import namedtuple


V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])

def sum(v0, v1):
    """
    Input: 2 size 3 vectors
    Output: Size 3 vector with the per element sum
    """
    return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

def sub(v0, v1):
    """
    Input: 2 size 3 vectors
    Output: Size 3 vector with the per element substraction
    """
    return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def mul(v0, k):
    """
    Input: 2 size 3 vectors
    Output: Size 3 vector with the per element multiplication
    """  
    return V3(v0.x * k, v0.y * k, v0.z *k)

def dot(v0, v1):
    """
    Input: 2 size 3 vectors
    Output: Scalar with the dot product
    """
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

def cross(v0, v1):
    """
    Input: 2 size 3 vectors
    Output: Size 3 vector with the cross product
    """  
    return V3(
    v0.y * v1.z - v0.z * v1.y,
    v0.z * v1.x - v0.x * v1.z,
    v0.x * v1.y - v0.y * v1.x,
    )

def length(v0):
    """
    Input: 1 size 3 vector
    Output: Scalar with the length of the vector
    """  
    return (v0.x**2 + v0.y**2 + v0.z**2)**0.5

def norm(v0):
    """
    Input: 1 size 3 vector
    Output: Size 3 vector with the normal of the vector
    """  
    v0length = length(v0)

    if not v0length:
        return V3(0, 0, 0)

    return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)

def bbox(*vertices):
    """
    Input: n size 2 vectors
    Output: 2 size 2 vectors defining the smallest bounding rectangle possible
    """  
    xs = [ vertex.x for vertex in vertices ]
    ys = [ vertex.y for vertex in vertices ]
    xs.sort()
    ys.sort()

    return V2(xs[0], ys[0]), V2(xs[-1], ys[-1])

def barycentric(A, B, C, P):
    """
    Input: 3 size 2 vectors and a point
    Output: 3 barycentric coordinates of the point in relation to the triangle formed
            * returns -1, -1, -1 for degenerate triangles
    """  
    bary = cross(
    V3(C.x - A.x, B.x - A.x, A.x - P.x), 
    V3(C.y - A.y, B.y - A.y, A.y - P.y)
    )

    if abs(bary[2]) < 1:
        return -1, -1, -1   # this triangle is degenerate, return anything outside

    return (
    1 - (bary[0] + bary[1]) / bary[2], 
    bary[1] / bary[2], 
    bary[0] / bary[2]
    )

def char(c):
    return struct.pack('=c', c.encode('ascii'))

def word(w):
    return struct.pack('=h', w)

def dword(w):
    return struct.pack('=l', w)

class Render(object):
    def color(self, r, g, b):
        return bytes([int(b*self.color_range), int(g*self.color_range), int(r*self.color_range)])

    def glInit(self):
        self.color_range = 255
        self.current_color_clear = self.color(0,0,0)
        self.current_color = self.color(1, 1, 1)

    def glCreateWindow(self, width, height):
        self.width = width
        self.height = height
        self.glClear()

    def glClear(self):
        self.framebuffer = [
            [self.current_color_clear for x in range(self.width)]
            for y in range(self.height)
        ]
        self.zbuffer = [
            [-float('inf') for x in range(self.width)]
            for y in range(self.height)
        ]

    def glClearColor(self, r, g, b):
        self.current_color_clear = self.color(r, g, b)

    def glViewPort(self, x, y, width, height):
        if x >= 0 and y >= 0 and width >= 0 and height >= 0 and x + width <= self.width and y + height <= self.height:
            self.xvp = x
            self.yvp = y
            self.wvp = width
            self.hvp = height

    def glFinish(self, filename):
        f = open(filename, 'bw')
        f.write(char('B'))
        f.write(char('M'))
        f.write(dword(14+40+3*(self.width*self.height)))
        f.write(dword(0))
        f.write(dword(14+40))

        f.write(dword(40))
        f.write(dword(self.width))
        f.write(dword(self.height))
        f.write(word(1))
        f.write(word(24))
        f.write(dword(0))
        f.write(dword(self.width*self.height*3))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))

        for y in range(self.height):
            for x in range(self.width):
                f.write(self.framebuffer[y][x])

        f.close()

    def glColor(self, r, g, b):
        self.current_color = self.color(r, g, b)
    def glPoint(self, x, y, color = None):
        self.framebuffer[y+self.yvp][x+self.xvp] = color or self.current_color
    def glVertex(self, x, y, color = None):
        if x >= -1 and x <= 1 and y >= -1 and y <= 1:
            self.framebuffer[int(self.yvp + y * (self.hvp / 2) + self.hvp / 2)][int(self.xvp + x * (self.wvp / 2) + self.wvp / 2)] = color or self.current_color
            
    def glLine(self, x0, y0, x1, y1):
        x0 = round(x0*self.wvp)
        y0 = round(y0*self.hvp)
        x1 = round(x1*self.wvp)
        y1 = round(y1*self.hvp)
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        steep = dy > dx
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

            dy = abs(y1 - y0)
            dx = abs(x1 - x0)

        offset = 0 * 2 * dx
        threshold = 0.5 * 2 * dx
        y = y0
        
        # y = mx + b
        points = []
        for x in range(x0, x1):
            if steep:
                points.append([y/self.wvp, x/self.hvp])
            else:
                points.append([x/self.wvp, y/self.hvp])

            offset += (dy) * 2
            if offset >= threshold:
                y += 1 if y0 < y1 else -1
                threshold += 1 * 2 * dx
        for point in points:
            self.glVertex(*point)
    def line(self, x0, y0, x1, y1):
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        steep = dy > dx
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

            dy = abs(y1 - y0)
            dx = abs(x1 - x0)

        offset = 0 * 2 * dx
        threshold = 0.5 * 2 * dx
        y = y0
        # y = mx + b
        points = []
        x = x0
        cont = 1
        if x0 > x1:
            cont = -1
        while x != x1:
            if steep:
                points.append([y, x])
            else:
                points.append([x, y])

            offset += (dy/dx) * 2 * dx
            if offset >= threshold:
                y += 1 if y0 < y1 else -1
                threshold += 1 * 2 * dx
            x = x + cont
        for point in points:
            self.glPoint(*point)
    """ def load(self, filename, translate, scale):
        model = Obj(filename)
        for face in model.faces:
            vcount = len(face)
            for j in range(vcount):
                f1 = face[j][0]
                f2 = face[(j + 1) % vcount][0]

                v1 = model.vertices[f1 - 1]
                v2 = model.vertices[f2 - 1]

                x1 = round((v1[0] + translate[0]) * scale[0])/self.wvp
                y1 = round((v1[1] + translate[1]) * scale[1])/self.hvp
                x2 = round((v2[0] + translate[0]) * scale[0])/self.wvp
                y2 = round((v2[1] + translate[1]) * scale[1])/self.hvp
                self.glLine(x1, y1, x2, y2) """
    def triangle(self, A, B, C, color=None):
        bbox_min, bbox_max = bbox(A, B, C)

        for x in range(bbox_min.x, bbox_max.x + 1):
            for y in range(bbox_min.y, bbox_max.y + 1):
                w, v, u = barycentric(A, B, C, V2(x, y))
                if w < 0 or v < 0 or u < 0:  # 0 is actually a valid value! (it is on the edge)
                    continue
                
                z = A.z * w + B.z * v + C.z * u
                if z > self.zbuffer[y][x]:
                    self.glPoint(x, y, color)
                    self.zbuffer[y][x] = z

    def transform(self, vertex, translate=(0, 0, 0), scale=(1, 1, 1)):
    # returns a vertex 3, translated and transformed
        return V3(
            round((vertex[0] + translate[0]) * scale[0]),
            round((vertex[1] + translate[1]) * scale[1]),
            round((vertex[2] + translate[2]) * scale[2])
        )
    def load(self, filename, translate=(0, 0, 0), scale=(1, 1, 1)):
        model = Obj(filename)

        light = V3(0,0,1)

        for face in model.vfaces:
            vcount = len(face)

            if vcount == 3:
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1

                a = self.transform(model.vertices[f1], translate, scale)
                b = self.transform(model.vertices[f2], translate, scale)
                c = self.transform(model.vertices[f3], translate, scale)

                normal = norm(cross(sub(b, a), sub(c, a)))
                intensity = dot(normal, light)
                grey = intensity
                if grey < 0:
                    continue  
                
                self.triangle(a, b, c, self.color(grey, grey, grey))
            else:
                # assuming 4
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1
                f4 = face[3][0] - 1   

                vertices = [
                    self.transform(model.vertices[f1], translate, scale),
                    self.transform(model.vertices[f2], translate, scale),
                    self.transform(model.vertices[f3], translate, scale),
                    self.transform(model.vertices[f4], translate, scale)
                ]

                normal = norm(cross(sub(vertices[0], vertices[1]), sub(vertices[1], vertices[2])))  # no necesitamos dos normales!!
                intensity = dot(normal, light)
                grey = intensity
                if grey < 0:
                    continue # dont paint this face

                # vertices are ordered, no need to sort!
                # vertices.sort(key=lambda v: v.x + v.y)
        
                A, B, C, D = vertices 
                
                self.triangle(A, B, C, self.color(grey, grey, grey))
                self.triangle(A, C, D, self.color(grey, grey, grey))

    def fillPolygon(self, texto, traslado):
        puntos = texto[:-1].split(') ')
        separado = [punto[1:].split(', ') for punto in puntos]
        lista = []
        for punto in separado:
            lista.append([str(int(punto[0])+traslado[0]),str(int(punto[1])+traslado[1])])
        cont = 0
        minx = 1000000
        miny = 1000000
        maxx = 0
        maxy = 0
        while cont < len(lista):
            self.line(int(lista[cont][0]), int(lista[cont][1]), int(lista[(cont+1) % len(lista)][0]), int(lista[(cont+1) % len(lista)][1]))
            if minx>int(lista[cont][0]):
                minx = int(lista[cont][0])
            if maxx<int(lista[cont][0]):
                maxx = int(lista[cont][0])
            if miny>int(lista[cont][1]):
                miny = int(lista[cont][1])
            if maxy<int(lista[cont][1]):
                maxy = int(lista[cont][1])
            #self.glLine(int(lista[cont][0])/self.wvp, int(lista[cont][1])/self.hvp, int(lista[(cont+1) % len(lista)][0])/self.wvp, int(lista[(cont+1) % len(lista)][1])/self.hvp)
            cont = cont + 1
        bandera = False
        for x in range(minx,maxx+1):
            for y in range(miny,maxy+1):
                if self.framebuffer[y][x] == self.current_color and not ([str(x),str(y)] in lista) and self.framebuffer[y+1][x] != self.current_color :
                    valor = False
                    for i in range(y+1, maxy+1):
                        if self.framebuffer[i][x] == self.current_color:
                            valor = True 
                    bandera = valor
                if bandera:
                    self.framebuffer[y][x] = self.current_color

r = Render()
r.glInit()
r.glCreateWindow(1920, 1080)
r.glViewPort(0, 0, 1920, 1080)
r.load('./models/dragon.obj',[990,450,100],[1.3,1.3,1])
#r.load('./models/s.obj',[2.7,1,0],[350,350,400])
#r.load('./models/face.obj',[50,20,0],[10,10,30])
""" r.glLine(-1,-1,1,0)
r.glLine(-1,-1,1,1)
r.glLine(-1,0,1,1)
r.glLine(0,-1,0,1)   """
""" r.current_color = color(255, 255, 255)
r.point(10, 10)
r.point(11, 10)
r.point(10, 11)
r.point(11, 11)
for x in range(1024):
    for y in range(768):
        if random.random() > 0.5:
            r.point(x, y) """
""" r.glVertex(0,0)
r.glVertex(1,1)
r.glVertex(0,1)
r.glVertex(0.2,1)
r.glVertex(0.4,1)
r.glVertex(0.6,1)
r.glVertex(0.8,1)
r.glVertex(1,0.2)
r.glVertex(1,0.4)
r.glVertex(1,0.6)
r.glVertex(1,0.8)
r.glColor(0, 0, 1)
r.glVertex(-1,1)
r.glVertex(-1,0)
r.glVertex(-0.2,1)
r.glVertex(-0.4,1)
r.glVertex(-0.6,1)
r.glVertex(-0.8,1)
r.glVertex(-1,0.2)
r.glVertex(-1,0.4)
r.glVertex(-1,0.6)
r.glVertex(-1,0.8)
r.glColor(0, 1, 0)
r.glVertex(1,-1)
r.glVertex(0,-1)
r.glVertex(1,-0.2)
r.glVertex(1,-0.4)
r.glVertex(1,-0.6)
r.glVertex(1,-0.8)
r.glVertex(0.2,-1)
r.glVertex(0.4,-1)
r.glVertex(0.6,-1)
r.glVertex(0.8,-1)
r.glColor(1, 0, 0)
r.glVertex(-1,-1)
r.glVertex(1,0)
r.glVertex(-0.2,-1)
r.glVertex(-0.4,-1)
r.glVertex(-0.6,-1)
r.glVertex(-0.8,-1)
r.glVertex(-1,-0.2)
r.glVertex(-1,-0.4)
r.glVertex(-1,-0.6)
r.glVertex(-1,-0.8) """
""" r.glClearColor(1,0,0)
r.glClear() """
r.glFinish('a.bmp')