
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np

"""w = 0.1
x = np.linspace(0,10,100)
y = np.cos(w*x)
ax = plt.subplot()
ax.plot(x, y)

def onselect(verts):
    print(verts)
lasso = mwidgets.LassoSelector(ax, onselect)

plt.show()"""


fig, ax = plt.subplots()
ax.plot([1, 2, 3], [10, 50, 100])
click = [None,None]
release = [None,None]
def onselect(eclick, erelease):
    click[:] = eclick.xdata, eclick.ydata
    release[:] = erelease.xdata, erelease.ydata
    print(eclick.xdata, eclick.ydata)
    print(erelease.xdata, erelease.ydata)
props = dict(facecolor='blue', alpha=0.5)
rect = mwidgets.RectangleSelector(ax, onselect, interactive=True,props=props)
rect.add_state('rotate')
plt.show()

print("Center ",rect.center)
print(rect.corners)
print(rect.rotation)