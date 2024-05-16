import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap as lsc
from scipy.spatial.transform import Rotation as R

# 生成花朵数据
t1 = np.array(range(25))/24
t2 = np.arange(0, 575.5, 0.5)/575*20*np.pi + 4*np.pi
[xr, tr] = np.meshgrid(t1, t2)
pr = (np.pi/2)*np.exp(-tr/(8*np.pi))
ur = 1 - (1 - np.mod(3.6*tr, 2*np.pi)/np.pi)**4/2 + np.sin(15*tr)/150 + np.sin(15*tr)/150
yr = 2*(xr**2 - xr)**2*np.sin(pr)
rr = ur*(xr*np.sin(pr) + yr*np.cos(pr))
hr = ur*(xr*np.cos(pr) - yr*np.sin(pr))

tb = np.resize(np.linspace(0, 2, 151), (1,151))
rb = np.resize(np.linspace(0, 1, 101), (101,1)) @ ((abs((1-np.mod(tb*5,2))))/2 + .3)/2.5
xb = rb*np.cos(tb*np.pi)
yb = rb*np.sin(tb*np.pi)
hb = np.power(-np.cos(rb*1.2*np.pi)+1, .2)

cL = np.array([[.33,.33,.69], [.68,.42,.63], [.78,.42,.57], [.96,.73,.44]])
cL = np.array([[.02,.04,.39], [.02,.06,.69], [.01,.26,.99], [.17,.69,1]])
cMpr = lsc.from_list('slandarer', cL)
cMpb = lsc.from_list('slandarer', cL*.4 + .6)

# 绕轴旋转数据点
def rT(X, Y, Z, T):
    SZ = X.shape
    XYZ = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)))
    RMat = R.from_euler('xyz', T, degrees = True); XYZ = RMat.apply(XYZ)
    return XYZ[:,0].reshape(SZ), XYZ[:,1].reshape(SZ), XYZ[:,2].reshape(SZ)

# 贝塞尔函数插值生成花杆并绘制
def dS(X, Y, Z):
    MN = np.where(Z == np.min(Z)); M = MN[0][0]; N = MN[1][0]
    x1 = X[M, N]; y1 = Y[M, N]; z1 = Z[M, N] + .03
    x = np.array([x1, 0, (x1*np.cos(np.pi/3) - y1*np.sin(np.pi/3))/3]).reshape((3,1))
    y = np.array([y1, 0, (y1*np.cos(np.pi/3) + x1*np.sin(np.pi/3))/3]).reshape((3,1))
    z = np.array([z1, -.7, -1.5]).reshape((3,1))
    P = np.hstack((x,y,z)).T
    t = (np.array(range(50)) + 1)/50
    c1 = np.array([1, 2, 1]).reshape(3,1)
    c2 = np.power(t, np.array(range(3)).reshape(3,1))
    c3 = np.power(1 - t, np.array(range(2, -1, -1)).reshape(3,1))
    P = (P @ (c1*c2*c3))
    ax.plot(P[0], P[1], P[2], color = '#58827E')


# 创建figure窗口及axis坐标区域
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 绘制花束
ax.plot_surface(rr*np.cos(tr), rr*np.sin(tr), hr + .35, rstride = 1, cstride = 1,
                    facecolors = cMpr(hr), antialiased = True, shade = False)
U, V, W = rT(rr*np.cos(tr), rr*np.sin(tr), hr + .35, [180/8, 0, 0]); V = V - .4
for i in range(5):
    U, V, W = rT(U, V, W, [0, 0, 72])
    ax.plot_surface(U, V, W - .1, rstride = 1, cstride = 1,
                        facecolors = cMpr(hr), antialiased = True, shade = False)
    dS(U, V, W - .1)

u1, v1, w1=rT(xb, yb, hb/2.5 + .32, [180/9, 0, 0])
v1 = v1 - 1.35
u2, v2, w2 = rT(u1, v1, w1, [0, 0, 36])
u3, v3, w3 = rT(u1, v1, w1, [0, 0, 24])
u4, v4, w4 = rT(u3, v3, w3, [0, 0, 24])
for i in range(5):
    u1, v1, w1 = rT(u1, v1, w1, [0, 0, 72])
    u2, v2, w2 = rT(u2, v2, w2, [0, 0, 72])
    u3, v3, w3 = rT(u3, v3, w3, [0, 0, 72])
    u4, v4, w4 = rT(u4, v4, w4, [0, 0, 72])
    ax.plot_surface(u1, v1, w1, rstride = 1, cstride = 1,
                        facecolors = cMpb(hb), antialiased = True, shade = False)
    ax.plot_surface(u2, v2, w2, rstride = 1, cstride = 1,
                        facecolors = cMpb(hb), antialiased = True, shade = False)
    ax.plot_surface(u3, v3, w3, rstride = 1, cstride = 1,
                        facecolors = cMpb(hb), antialiased = True, shade = False)
    ax.plot_surface(u4, v4, w4, rstride = 1, cstride = 1,
                        facecolors = cMpb(hb), antialiased = True, shade = False)
    dS(u1, v1, w1)
    dS(u2, v2, w2)
    dS(u3, v3, w3)
    dS(u4, v4, w4)

ax.set_position((-.215, -.3, 1.43, 1.43))
ax.set_box_aspect((1, 1, .8))
ax.view_init(elev = 50, azim = 2)
ax.axis('off')
plt.show()