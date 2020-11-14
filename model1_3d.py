from numpy import cos,sin
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


G = 9.81 # gravity
R = 0.33 # radius 
M1 = 1.3 # mass of the wheel
M2 = 0.490 # mass of the bird
D1 = 0.5 # friction theta 
D2 = 0.5 # friction phi
D3 = 100 # friction nu
INIT_STATE = [0.1, 0, 0, 0, 1, 0]


class Wheel:
    def __init__(self, init_state=INIT_STATE):
        self.state = init_state
        self.time_elapsed = 0

    def a(self):
        theta, x1, phi, x2, z, x3 = self.state
        a1 = 0
        a1 -= 2 * M2 * z * x3 * (x1 + x2)
        a1 -= M2 * R * x3 * cos(phi) * x2
        a1 -= M2 * R * x3 * (2 * cos(phi) * x1 + x2 * cos(phi))
        a1 -= D1 * x1
        a1 += 2 * M2 * R * z * sin(phi) * x1 * x2
        a1 += M2 * R * z * sin(phi) * (x2 ** 2)
        a1 += M2 * G * R * sin(theta)
        a1 += M2 * G * z * sin(theta + phi)
        return a1

    def b(self):
        theta, x1 , phi, x2, z, x3 = self.state
        b1 = 0
        b1 -= 2 * M2 * z * x3 * (x1 + x2)
        b1 -= M2 * R * x1 * (x3 * cos(phi) - z * sin(phi) * x2)
        b1 += M2 * G * z * sin(theta + phi)
        b1 += M2 * R * x3 * x1 * cos(phi)
        b1 -= M2 * R * x1 * z * (x1 + x2) * sin(phi)
        b1 -= D2 * x2 
        return b1

    def c(self):
        theta, x1 , phi, x2, z, x3 = self.state
        c1 = 0
        c1 -= R * x1 * cos(phi) * x2
        c1 += M2 * z * (x1 + x2) **2
        c1 += M2 * R * x1 *(x1 + x2) * cos(phi)
        c1 -= M2 * G * cos(theta + phi)
        c1 -= D3 * x3
        return c1

    def A(self):
        theta, x1, phi, x2, z, x3 = self.state
        A1 = ((M2 * z) **2) * self.a()
        A1 -= ((M2 * z) **2 + (M2 **2) * R * z * cos(phi)) * self.b()
        A1 -= ((M2 * z) **2) * R * sin(phi) * self.c()
    
        A2 = -( (M2 * z) **2 + (M2 **2) * R * z * cos(phi)) * self.a()
        A2 += (M1 * M2 * (R **2) + ((M2 * R) **2) + ((M2 * z) **2) + 2 * (M2 **2) * R * z * cos(phi) - ((M2 * R * sin(phi)) **2)) * self.b()
        A2 += (((M2 * z) **2)* R * sin(phi) + ((M2 * R) **2) * z * sin(phi) * cos(phi)) * self.c()
    
        A3 = -((M2 * z) **2) * R * sin(phi) * self.a()
        A3 += (((M2 * z) **2) * R * sin(phi) + ((M2 * R) **2) * z * sin(phi) * cos(phi)) * self.b()
        A3 += (M1 * M2 * ((R * z) **2) + (M2 * R * z) **2 - (M2 * R * z * cos(phi)) ** 2) * self.c()
        alfa = lambda z: (M1 + M2 - 1) * ((M2 * R * z) ** 2) 
        alfa1 = 1/alfa(z)
        return alfa1 * A1 ,alfa1 * A2, alfa1 * A3
       
    def model(self,state,t):
        theta, x1, phi, x2, z, x3 = state
        dtheta = x1
        dphi = x2
        dz = x3
        matriz = self.A()
        dx1 = matriz[0]
        dx2 = matriz[1]
        dx3 = matriz[2]
        return dtheta, dx1, dphi, dx2, dz, dx3

    def step(self,dt):
        self.state = odeint(self.model, self.state, [0, dt])[1]
        self.time_elapsed += dt
        
    def position(self):
        theta,_,phi,_,z,_ = self.state
        x = 0, R * sin(theta), R * sin(theta) + z * sin(theta + phi)    
        y = 0, R * cos(theta), R * cos(theta) + z * cos(theta + phi)
        return (x,y)
        


Modelo = Wheel()
dt = 1./30

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-.75, .75), ylim=(-.75, .75))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    return line, time_text, energy_text

def animate(i):
    """perform animation step"""
    global Modelo, dt
    Modelo.step(dt)
    line.set_data(Modelo.position())
    time_text.set_text('time = %.1f' % Modelo.time_elapsed)
    return line, time_text

# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300,
                              interval=interval, blit=True, init_func=init)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('2grados.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
def circle(l):
    s = np.linspace(0,2*np.pi,100)
    return [l*cos(s),l*sin(s)]
x1,y1 = circle(R)
plt.plot(x1,y1)
#ani.save('3grados.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
