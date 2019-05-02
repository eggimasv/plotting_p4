import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colorbar, colors
from matplotlib.colors import Normalize


cmap = plt.cm.get_cmap('Reds')
norm = Normalize(vmin=1, vmax=100)

# Add legend to figure
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
plt.colorbar(sm, ax=ax)
                        # draw a new figure and replot the colorbar there
fig_cb, ax_cb = plt.subplots(figsize=cm2inch(10, 10))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
plt.colorbar(sm, ax=ax_cb)
                # Reset figure size
#fig = plt.gcf()
#fig.set_size_inches(cm2inch(0.5, 3))
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(path_out_folder_fig6, "{}__legend.pdf".format(filname)))
plt.close()