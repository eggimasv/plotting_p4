import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''A = [45]
B = [91]

fig = plt.figure()
ax = fig.subplots()

bar_width = 0.5
bar_l = [1]
tick_pos = [i + (bar_width / 2) for i in bar_l]

ax1 = ax.bar(bar_l, A, width=bar_width, label="A", color="green")
ax2 = ax.bar(bar_l, B, bottom=A, width=bar_width, label="B", color="blue")

for r1, r2 in zip(ax1, ax2):
    h1 = r1.get_height()
    h2 = r2.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="center", color="white", fontsize=16, fontweight="bold")
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="center", color="white", fontsize=16, fontweight="bold")

plt.show()'''
'''
fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('title')
plt.xlabel('x')
plt.ylabel('y')      
#plt.show()
'''
'''def autolabel(rects):
    # Attach some text labels.
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width() / 2.,
                rect.get_y() + rect.get_height() / 2.,
                '%f'%rect.get_height(),
                ha = 'center',
                va = 'center')

fig, ax = plt.subplots()

df = pd.DataFrame([[80, 40]], columns=['a', 'b'])

ax.barh(df.values, df.columns width, stacked=True,)

autolabel(r)

plt.show()
'''
df = pd.DataFrame([[80.234234234, 20.234234234]], columns=['a', 'b'])
ax = df.plot(kind='barh', stacked=True, width=1)


def autolabel(rects):
    for rect in rects:
        value = rect.get_width()
        ax.text(rect.get_x() + rect.get_width() / 2.,
                rect.get_y() + rect.get_height() / 2.,
                s=value,
                ha = 'center',
                va = 'center')

# create a list to collect the plt.patches data
'''totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)'''
#http://robertmitchellv.com/blog-bar-chart-annotations-pandas-mpl.html
# set individual bar lables using above list
autolabel(ax.patches)
for container in ax.containers:
    plt.setp(container, height=10)

'''for i in ax.patches:

    value = i.get_width()
    value = i.get_x() #Return the left coord of the rectangle.
    height = i.get_y() + 0.4
    print("{}  {}".format(value, height))
    # get_width pulls left or right; get_y pushes up or down
    ##ax.text(i.get_width()+.3, i.get_y()+.38, str(round((i.get_width()/total)*100, 2))+'%', fontsize=15, color='dimgrey')
    ax.text(x=(100 - value), y=height, s=" {}%".format(value), ha="left", va="center", color="black", fontsize=8, fontweight="bold")
    #ax.text(x=value, y=height, s=value, ha="center", va="center", color="white", fontsize=8, fontweight="bold")'''
plt.tight_layout()
plt.show()