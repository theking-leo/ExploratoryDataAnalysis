Exploratory Data Analysis(EDA)
We Use EDA to get the basic understanding of the data set i.e we explore the data.

First thing we do after getting the data set is EDA.

Iris Flower DataSet
Toy Dataset: Iris Dataset: [https://en.wikipedia.org/wiki/Iris_flower_data_set]

A simple dataset to learn the basics.
3 flowers of Iris species. [see images on wikipedia link above]
1936 by Ronald Fisher.
Petal and Sepal: http://terpconnect.umd.edu/~petersd/666/html/iris_with_labels.jpg
* Objective: Classify a new flower as belonging to one of the 3 classes given the 4 features.
Importance of domain knowledge.
Why use petal and sepal dimensions as features?
Why do we not use 'color' as a feature?
The four features of these 3 types of flowers
(Iris setosa, Iris versicolor, Iris virginica) are:

1)Sepal Length
2)Sepal Width
3)Petal Length
4)petal Width

We will classify the given flower by using this Four Features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
iris = pd.read_csv('iris.csv');
# Q) How many data points and Features in dataset?
iris.shape
(150, 5)
observation: 150 rows/data points and 5 cols/features

#cols present in dataset

iris.columns
Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
       'species'],
      dtype='object')
iris.head()
sepal_length	sepal_width	petal_length	petal_width	species
0	5.1	3.5	1.4	0.2	setosa
1	4.9	3.0	1.4	0.2	setosa
2	4.7	3.2	1.3	0.2	setosa
3	4.6	3.1	1.5	0.2	setosa
4	5.0	3.6	1.4	0.2	setosa
#Q) how many datapts for each type of flower?

#types = iris.groupby('species');
#types.count()

#or

iris['species'].value_counts()
virginica     50
versicolor    50
setosa        50
Name: species, dtype: int64
Observation: its a balanced data set

2-D ScatterPlot
# label, color are optional parameters
iris.plot(x='sepal_length',y='sepal_width',kind='scatter',label='datapoints');
plt.show()

# using seaborn lib to stylize the graph
# sns is seaborn as imported above
sns.set_style("whitegrid");

# hue = 'species' -> color encoding based on 
# distinct species here 3 types so 3 colors

#size -> size of the graph
sns.FacetGrid(iris, hue='species',size=5) \
    .map(plt.scatter, 'sepal_length','sepal_width') \
    .add_legend();
plt.show();

Obeservation:
1)Setosa Flowers can be distinguished from this as they dont overlap with other types of flowers.

2)Versicolor and and Virginica flowers cant be distinguished with this features(sepal lenght and width) as they overlapped.

If we draw a line we can classify setosa from these 3 types as they lie left side of the line.

3-D Scatter Plots
https://plot.ly/pandas/3d-scatter-plots/

Needs a lot to mouse interaction to interpret data.

What about 4-D, 5-D or n-D scatter plot? Ans) as Human can visualize only 3d. n-D can be visualized through Maths oe of the way is 'Pair-Plot'

Pair Plots
As we cant visualize N-D we will divide features into all possible pairs and visualize the plot of pairs gives the total idea of the data.

In Our Case We have 4 features:

1)SL - sepal length
2)SW - sepal width
3)PL - petal length
4)PW - petal width

No.of pair plots with these 4 features is: 4C2
i.e 6

(SL,SW),(SL,PL),(PL,SW).....

Note: We use Seaborn Lib to PairPlot

sns.set_style("whitegrid");
sns.pairplot(iris,hue='species',size=3);
plt.show()

Observation: From the above graphs PW-PL graph is perfect to distinguish between 3 flowers with some compramise, below is the final graph to look clearly

sns.set_style("whitegrid");

# hue = 'species' -> color encoding based on 
# distinct species here 3 types so 3 colors

#size -> size of the graph
sns.FacetGrid(iris, hue='species',size=5) \
    .map(plt.scatter, 'petal_width','petal_length') \
    .add_legend();
plt.show();

Disadvantage of Pair Plot
Here in the above case for 4d or 4 features we have 6 plots, Consider 100d or 100 features no of plots will be 100C2 plots.

Visualizing Covariance Matrix
iris.corr()
sepal_length	sepal_width	petal_length	petal_width
sepal_length	1.000000	-0.109369	0.871754	0.817954
sepal_width	-0.109369	1.000000	-0.420516	-0.356544
petal_length	0.871754	-0.420516	1.000000	0.962757
petal_width	0.817954	-0.356544	0.962757	1.000000
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Compute the correlation matrix
corr = iris.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
<matplotlib.axes._subplots.AxesSubplot at 0x2299667fe10>

import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
uniform_data = iris.corr()
ax = sns.heatmap(uniform_data)
