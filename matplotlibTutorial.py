#
# author: F. Coppo
# description: visualizing data with python tutorial
#

"""
Matplotlib

- back-end layer: it has 3 built-in interface classes (FigureCanvas define  the figure's area; Render class define how to draw; Event class manage input like mouse or keyboard)

- artist layer: Artist object is able to take the Render and put ink on canvas. The title, the lines, the tick labels on matplotlib are all Artist instance
				There are two types of Artist objects. The first type is the primitive type, such as a line, a rectangle, a circle, or text;
				the second type is the composite type, such as the figure
				
- scripting layer: user layer
"""

# example 1
# generate histo with random number using artist layer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas # import figure canvas from back end layer (agg -> anty grain geometry that is high performance lib)
from matplotlib.figure import Figure # import figure artist
fig = Figure()
canvas = FigureCanvas(fig) # attach the Figure artist to Figure canvas

import numpy as np 	# to generate random number
x = np.random.randn(10000)
ax = fig.add_subplot(111) # it create an axis artist obj (the axes artist is added automatically to the figure axes container)
ax.hist(x, 100) # generate the histogram of 10000 points
ax.set_title('Histo Example')
fig.savefig('matplotlib_histogram.png')

# example 2
# generate histo with random number using scripting layer
import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(10000)
plt.hist(x, 100)
plt.title('Histo example using script layer')
plt.show()