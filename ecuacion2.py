from numpy import cos,sin
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
R=1
l=1
k1 = 1
k2 = 1
g = -9.81
m1 = 1
m2 = 1
d2 = 1
d1 = 1
d3 = 1

def a(theta,x1,phi,x2,z,x3):
    a1 = 0
    a1 -= 2*m2*z*x3*(x1+x2)
    a1 -= m2*R*x3*cos(phi)*x2
    a1 -= m2*R*x3*(2*cos(phi)*x1 + x2*cos(phi))
    a1 -= d1*x1
    a1 += 2*m2*R*z*sin(phi)*x1*x2
    a1 += m2*R*z*sin(phi)*(x2**2)
    a1 += m2*g*R*sin(theta)
    a1 += m2*g*z*sin(theta+phi)
    return a1

def b(theta,x1,phi,x2,z,x3):
    b1 = 0
    b1 -= 2*m2*z*x3*(x1+x2)
    b1 -= m2*R*x1*(x3*cos(phi)-z*sin(phi)*x2)
    b1 += m2*g*z*sin(theta+phi)
    b1 += m2*R*x3*x1*cos(phi)
    b1 -= m2*R*x1*z*(x1+x2)*sin(phi)
    b1 -= d2*x2 
    return b1

def c(theta,x1,phi,x2,z,x3):
    c1 = 0
    c1 -= R*x1*cos(phi)*x2
    c1 += m2*z*(x1*x2)**2
    c1 += m2*R*x1*(x1+x2)*cos(phi)
    c1 -= m2*g*cos(theta + phi)
    c1 -= d3*x3
    return c1


alfa = lambda z : (m1+m2-1)*((m2*R*z)**2)

def A(theta,x1,phi,x2,z,x3):
    A1 = (m2*z)**2
    A1 -= ((m2*z)**2 + (m2**2)*R*z*cos(phi))*b(theta,x1,phi,x2,z,x3)
    A1 -= ((m2*z)**2)*R*sin(phi)*c(theta,x1,phi,x2,z,x3)

    A2 = -( (m2*z)**2 + (m2**2)*R*z*cos(phi))*a(theta,x1,phi,x2,z,x3)
    A2 += (m1*m2*(R**2) + ((m2*R)**2) + ((m2*z)**2) + 2*(m2**2)*R*z*cos(phi) - ((m2*R*sin(phi))**2))*b(theta,x1,phi,x2,z,x3)
    A2 += (((m2*R)**2)*sin(phi) + ((m2*R)**2)*z *sin(phi)*cos(phi))*c(theta,x1,phi,x2,z,x3)

    A3 = -((m2*z)**2)*R*sin(phi)*a(theta,x1,phi,x2,z,x3)
    A3 += (((m2*z)**2)*R*sin(phi) + m2*(R**2)*z*sin(phi)*cos(phi))*b(theta,x1,phi,x2,z,x3)
    A3 += (m1*m2*((R*z)**2) + (m2*R*z)**2 - (m2*R*z*cos(phi))**2)*c(theta,x1,phi,x2,z,x3)

    alfa1 = 1/alfa(z)
    return alfa1*A1 ,alfa1*A2,alfa1*A3

def model(y,t):
    theta,x1,phi,x2,z,x3 = y
    dtheta = x1
    dphi = x2
    dz = x3
    matriz = A(theta,x1,phi,x2,z,x3)
    dx1 = matriz[0]
    dx2 = matriz[1]
    dx3 = matriz[2]
    return dtheta,dx1,dphi,dx2,dz,dx3

y0 = [1,0,1,0,1,0]
t = np.linspace(0,10,2000)
y = odeint(model,y0,t)
print(y)

theta = y[0]
phi = y[2]
def circle(l):
    s = np.linspace(0,2*np.pi,1000)
    return [l*cos(s),l*sin(s)]

x1,y1 = circle(1)
plt.plot(0,0,'x')
plt.plot(x1,y1,alpha = 0.2)
plt.plot(cos(theta),sin(theta),'.r')
plt.plot(cos(phi),sin(phi),'.b')
plt.show() 



    
