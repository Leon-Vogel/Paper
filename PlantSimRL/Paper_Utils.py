import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

fsize = 15
tsize = 18

tdir = 'in'

major = 5.0
minor = 3.0

style = 'default'

plt.style.use(style)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor

r = 15
theta = 5
rw = 12
gamma = 0.1

xsize = 8
ysize = 5

# fig, ax = plt.subplots(figsize=(xsize, ysize))

plt.figure(figsize=(xsize, ysize))

err = np.arange(0., r, .05)
z = np.where(err < rw, 0, gamma * (err - rw) ** 2 * np.sin(np.deg2rad(theta)))

plt.scatter(err, z, s=7, label=r'$\Sigma(x) = \gamma x^2 \sin(\theta)$')
plt.title('Title size = ' + str(tsize))

plt.annotate('Plotting style = ' + style, xy=(1, 0.05), ha='left', va='center')
plt.annotate('Figure size = ' + str(xsize) + ' x ' + str(ysize) + ' (in inches)', xy=(1, 0.045), ha='left', va='center')
plt.annotate('Font size = ' + str(fsize) + ' (in pts)', xy=(1, 0.04), ha='left', va='center')
plt.annotate('Tick direction = ' + tdir, xy=(1, 0.035), ha='left', va='center')
plt.annotate('Tick major, minor size = ' + str(major) + ' and ' + str(minor), xy=(1, 0.03), ha='left', va='center')

ax = plt.gca()

ax.xaxis.set_minor_locator(MultipleLocator(.5))
ax.yaxis.set_minor_locator(MultipleLocator(.005))
plt.legend()
plt.xlabel('$x$', labelpad=10)
plt.ylabel('$\phi$', labelpad=10);
plt.savefig('professional_plot.png', dpi=300, pad_inches=.1, bbox_inches='tight')