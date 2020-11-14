from numpy import cos,sin
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

R=3
g = 9.81
m1 = 10
m2 = 4
d2 = 0.5
d1 = 0.5
z=1

def a(theta,x1,phi,x2):
    a1 = 0
    a1 -= d1*x1
    a1 += 2*m2*R*z*sin(phi)*x1*x2
    a1 += m2*R*z*sin(phi)*(x2**2)
    a1 += m2*g*R*sin(theta)
    a1 += m2*g*z*sin(theta+phi)
    return a1

def b(theta,x1,phi,x2):
    b1 = 0
    b1 += m2*g*z*sin(theta+phi)
    b1 += m2*R*z*x1*sin(phi)*x2
    b1 -= m2*R*x1*z*(x1+x2)*sin(phi)
    b1 -= d2*x2 
    return b1

alfa = lambda phi : m2*((R*z)**2)*(m1+m2-m2*(cos(phi)**2))

def A(theta,x1,phi,x2):
    A1 = (m2*((z)**2))*a(theta,x1,phi,x2)
    A1 -= (m2*(z**2) + (m2*R*z*cos(phi)))*b(theta,x1,phi,x2)
     
           
    A2 = -(m2*(z**2) + (m2*R*z*cos(phi)))*a(theta,x1,phi,x2)
    A2 += (m1*(R**2) + m2*(R**2) + (m2*(z**2)) + 2*(m2*R*z*cos(phi)))*b(theta,x1,phi,x2)
 
    alfa1 = 1/alfa(z)
    return alfa1*A1 ,alfa1*A2

def model(y,t):
    theta,x1,phi,x2 = y
    dtheta = x1
    dphi = x2
    matriz = A(theta,x1,phi,x2)
    dx1 = matriz[0]
    dx2 = matriz[1]
    return dtheta,dx1,dphi,dx2


y0 = [1,0,0,0]
t = np.linspace(0,10,2000)
y = odeint(model,y0,t)
print(y)

theta = y[:,0]
phi = y[:,2]
def circle(l):
    s = np.linspace(0,2*np.pi,1000)
    return [R*cos(s),R*sin(s)]

x1,y1 = circle(1)
plt.plot(0,0,'x')
plt.plot(x1,y1,alpha = 0.2)
plt.plot(R*sin(theta),R*cos(theta),'.r')
plt.plot(R*sin(theta)+z*sin(theta+phi),R*cos(theta)+z*cos(theta+phi),'.b')
plt.show() 