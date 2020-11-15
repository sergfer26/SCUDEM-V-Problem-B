from numpy import cos,sin
from scipy.integrate import odeint
from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from time import sleep

G = 9.81 # gravity
R = 0.33 # radius 
M1 = 1.3 # mass of the wheel
M2 = 0.490 # mass of the bird
D1 = 0.0022 # friction theta 
D2 = 0.5 # friction phi
D3 = 20 #0.1 # friction eta
INIT_STATE = [0.1* np.pi, .2 * np.pi, .05, .5, 0, 0]
L0 = 0.1
K = 75


class Wheel:

    def __init__(self, init_state=INIT_STATE):
        self.state = init_state
        self.time_elapsed = 0

    def invM(self):
        '''
            returns Inverse Matrix LHS
        '''
        theta, phi, nu, _, _, _= self.state
        l = (L0 + nu)
        M = np.zeros((3, 3))
        M[0, :] = [(M1 + M2) * R **2, M2 * R * l * cos(phi - theta), M2 * R * sin(phi - theta)]
        M[1, :] = [M2 * R * l, M2 * l ** 2, 0]
        M[2, :] = [M2 * R, 0, M2]
        return inv(M)

    def A(self):
        '''
            returns RHS 1
        '''
        theta, phi, nu, theta_d, phi_d, nu_d = self.state
        a0 = (L0 + nu) * (phi_d ** 2) * sin(phi - theta) - 2 * nu_d * phi_d  * cos(phi - theta)
        a1 = -(L0 + nu) * (theta_d ** 2) * sin(phi - theta)
        a2 = (theta_d ** 2) * cos(phi - theta)
        A = np.array([a0, a1, a2])
        A *= M2 * R
        return A

    def B(self):
        '''
            returns RHS 2
        '''
        theta, phi, nu, _, phi_d, nu_d = self.state
        b0 = (M2) * G * R * sin(theta)
        b1 = - 2 * M2 * (nu + L0) * nu_d * phi_d + M2 * G * (nu + L0) * sin(phi)
        b2 = M2 * (nu + L0) * phi_d ** 2 - K * nu-M2*G*cos(phi)
        B = np.array([b0, b1, b2])
        return B

    def model(self, state, t):
        '''
            returns RHS F
        '''
        _, _, _, theta_d, phi_d, nu_d = state
        theta = theta_d
        phi = phi_d
        nu = nu_d
        x = self.B() + self.A() - np.array([D1 * theta_d, D2 * phi_d, D3 * nu_d])
        theta_d, phi_d, nu_d = np.matmul(self.invM(), x)
        return theta, phi, nu, theta_d, phi_d, nu_d

    def step(self,dt):
        #sleep(0.5)
        self.state = odeint(self.model, self.state, [0, dt])[1]
        self.time_elapsed += dt

    def position(self):
        theta, phi, nu, _, _, _ = self.state
        x = 0, R * sin(theta), R * sin(theta) + (L0 + nu) * sin(phi)    
        y = 0, R * cos(theta), R * cos(theta) + (L0 + nu) * cos(phi)
        return (x,y)

    
Model = Wheel()
dt = 1./30

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=True,
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
    global Model, dt
    Model.step(dt)
    line.set_data(Model.position())
    time_text.set_text('time = %.1f' % Model.time_elapsed)
    return line, time_text

# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)
print(t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=550,
                              interval=interval, blit=True, init_func=init)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('2grados.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
def circle(l):
    s = np.linspace(0, 2 * np.pi, 100)
    return [l * cos(s),l * sin(s)]
x1, y1 = circle(R)
plt.plot(x1, y1)
# ani.save('3grados.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()



        


         