
# coding: utf-8

# # Linear Regression

# In[3]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin

# Set some Pandas options
#pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 25)


# In[16]:

x = np.array([2.2, 4.3, 5.1, 5.8, 6.4, 8.0])
y = np.array([0.4, 10.1, 14.0, 10.9, 15.4, 18.5])
plt.plot(x,y,'ro')
plt.xlim(2,9), plt.ylim(0,20)


# We can build a model to characterize the relationship between $X$ and $Y$, recognizing that additional factors other than $X$ (the ones we have measured or are interested in) may influence the response variable $Y$.
# 
# <div style="font-size: 150%;">  
# $y_i = f(x_i) + \epsilon_i$
# </div>

# where $f$ is some function, for example a linear function:
# 
# <div style="font-size: 150%;">  
# $y_i = \beta_0 + \beta_1 x_i + \epsilon_i$
# </div>

# and $\epsilon_i$ accounts for the difference between the observed response $y_i$ and its prediction from the model $\hat{y_i} = \beta_0 + \beta_1 x_i$. This is sometimes referred to as **process uncertainty**.

# We would like to select $\beta_0, \beta_1$ so that the difference between the predictions and the observations is zero, but this is not usually possible. Instead, we choose a reasonable criterion: ***the smallest sum of the squared differences between $\hat{y}$ and $y$***.
# 
# <div style="font-size: 120%;">  
# $$R^2 = \sum_i (y_i - [\beta_0 + \beta_1 x_i])^2 = \sum_i \epsilon_i^2 $$  
# </div>
# 
# Squaring serves two purposes: (1) to prevent positive and negative values from cancelling each other out and (2) to strongly penalize large deviations. Whether the latter is a good thing or not depends on the goals of the analysis.
# 
# In other words, we will select the parameters that minimize the squared error of the model.

# In[22]:

ss = lambda beta, x, y: np.sum((y - beta[0] - beta[1]*x) ** 2)


# In[23]:

ss([np.mean(y),0],x,y)


# In[24]:

b0,b1 = fmin(ss, [np.mean(y),0], args=(x,y))
b0,b1


# In[25]:

plt.plot(x, y, 'ro')
plt.plot([0,10], [b0, b0+b1*10])
plt.xlim(2,9);plt.ylim(0,20)


# In[17]:

plt.plot(x, y, 'ro')
plt.plot([0,10], [b0, b0+b1*10])
for xi, yi in zip(x,y):
    plt.plot([xi]*2, [yi, b0+b1*xi], 'k:')
plt.xlim(2,9); plt.ylim(0, 20)


# Minimizing the sum of squares is not the only criterion we can use; it is just a very popular (and successful) one. For example, we can try to minimize the sum of absolute differences:

# In[28]:

sabs = lambda theta, x, y: np.sum(np.abs(y - theta[0] - theta[1]*x))
b0,b1 = fmin(sabs, [0,1], args=(x,y))
print b0,b1
plt.plot(x, y, 'ro')
plt.plot([0,10], [b0, b0+b1*10])
plt.xlim(2,9);plt.ylim(0,20)


# We are not restricted to a straight-line regression model; we can represent a curved relationship between our variables by introducing **polynomial** terms. For example, a cubic model:
# 
# <div style="font-size: 150%;">  
# $y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \epsilon_i$
# </div>

# In[29]:

ss2 = lambda theta, x, y: np.sum((y - theta[0] - theta[1]*x - theta[2]*(x**2)) ** 2)
b0,b1,b2 = fmin(ss2, [1,1,-1], args=(x,y))
print b0,b1,b2
plt.plot(x, y, 'ro')
xvals = np.linspace(0, 10, 100)
plt.plot(xvals, b0 + b1*xvals + b2*(xvals**2))


# Although polynomial model characterizes a nonlinear relationship, it is a linear problem in terms of estimation. That is, the regression model $f(y | x)$ is linear in the parameters.
# 
# For some data, it may be reasonable to consider polynomials of order>2. For example, consider the relationship between the number of home runs a baseball player hits and the number of runs batted in (RBI) they accumulate; clearly, the relationship is positive, but we may not expect a linear relationship.

# In[40]:

ss3 = lambda theta, x, y: np.sum((y - theta[0] - theta[1]*x - theta[2]*(x**2)) ** 2)

bb = pd.read_csv("C:/Users/Sean Najera/OneDrive/lvpythonds/meetings/Meetup_3/baseball.csv", index_col=0)
x=bb.hr
y=bb.rbi
plt.plot(x, y, 'r.')
b0,b1,b2 = fmin(ss3, [np.mean(y),0,0], args=(x, y))
xvals = np.arange(37)
plt.plot(xvals, b0 + b1*xvals + b2*(xvals**2))
'Intercept: ' + str(b0), 'beta1: ' + str(b1), 'beta2: ' + str(b2)


# Of course, we need not fit least squares models by hand. The `statsmodels` package implements least squares models that allow for model fitting in a single line:

# In[41]:

from statsmodels.formula.api import ols 

data = pd.DataFrame(dict(x=x, y=y))
cubic_fit = ols('y ~ x + I(x**2)', data).fit()

cubic_fit.summary()


# # K- Nearest Neighbor

# In[6]:

import pylab as pl
import pyreadline
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from sklearn import neighbors


# In[4]:

iris = pd.read_excel('C:/Users/Sean Najera/OneDrive/lvpythonds/meetings/Meetup_3/iris.xlsx')
X = iris[['sepal_L','sepal_W']]  
y = iris['type'] 


# In pattern recognition, the ** k-Nearest Neighbors ** (or ** k-NN ** for short) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:
# 
# * In k-NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
# 
# * In k-NN regression, the output is the property value for the object. This value is the average of the values of its k nearest neighbors.
# 
# k-NN is a type of supervised learning and is among the simplest of all machine learning algorithms.
# 
# Both for classification and regression, it can be useful to weight the contributions of the neighbors, so that the nearer neighbors contribute more to the average than the more distant ones. For example, a common weighting scheme consists in giving each neighbor a weight of 1/d, where d is the distance to the neighbor.
# 
# The neighbors are taken from a set of objects for which the class (for k-NN classification) or the object property value (for k-NN regression) is known. This can be thought of as the training set for the algorithm, though no explicit training step is required.
# 
# A shortcoming of the k-NN algorithm is that it is sensitive to the local structure of the data.

# In[25]:

with mpl.rc_context(rc={'font.family': 'serif', 'font.weight': 'bold', 'font.size': 8}):
    plt.plot(iris.sepal_L[:50],iris.sepal_W[:50] , 'ro', label = 'Iris Setosa')
    plt.plot(iris.sepal_L[50:100],iris.sepal_W[50:100] , 'go', label = 'Iris Versicolor')
    plt.plot(iris.sepal_L[101:150],iris.sepal_W[101:150] , 'bo', label = 'Iris Virginica')
    plt.ylabel('Sepal Width');plt.xlabel('Sepal Length');plt.title('Iris Type by Sepal Dimensions');plt.legend(loc = 1, fontsize = 'medium')


# Let us attempt to create a simple classifer of Iris species based on two features
# 
# * Sepal Length
# 
# * Sepal Width

# We create an instance of Neighbors Classifier from the Scikit package and fit the data to a k=8 nearest neighbor model.

# In[40]:

k=8
clf = neighbors.KNeighborsClassifier(k) #establish settings
clf.fit(X, y) #enter the training data


# Now let us classify all other sepal length/width coordinates by predicting their classification with the model

# In[41]:

#### setting up the plot
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X['sepal_L'].min() - 1, X['sepal_L'].max() + 1
y_min, y_max = X['sepal_W'].min() - 1, X['sepal_W'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .1),
                     np.arange(y_min, y_max, .1))

####running the prediction for all the points in the plot
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])


# We will now put the results into a color map along with the training points.

# In[42]:

with mpl.rc_context(rc={'font.family': 'serif', 'font.weight': 'bold', 'font.size': 8}):
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.figure()
    pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot the training points
    pl.scatter(X['sepal_L'], X['sepal_W'], c=y, cmap=cmap_bold)
    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())
    pl.title("3-Class classification (k = %s, Metric = Euclidean Distance)" % (k))


# In[ ]:



