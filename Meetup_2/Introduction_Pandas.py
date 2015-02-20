
# coding: utf-8

# # Introduction to Pandas
# 
# **pandas** is a Python package providing fast, flexible, and expressive data structures designed to work with *relational* or *labeled* data both. It is a fundamental high-level building block for doing practical, real world data analysis in Python. 
# 
# pandas is well suited for:
# 
# - Tabular data with heterogeneously-typed columns, as in an SQL table or Excel spreadsheet
# - Ordered and unordered (not necessarily fixed-frequency) time series data.
# - Arbitrary matrix data (homogeneously typed or heterogeneous) with row and column labels
# - Any other form of observational / statistical data sets. The data actually need not be labeled at all to be placed into a pandas data structure
# 
# 
# Key features:
#     
# - Easy handling of **missing data**
# - **Size mutability**: columns can be inserted and deleted from DataFrame and higher dimensional objects
# - Automatic and explicit **data alignment**: objects can be explicitly aligned to a set of labels, or the data can be aligned automatically
# - Powerful, flexible **group by functionality** to perform split-apply-combine operations on data sets
# - Intelligent label-based **slicing, fancy indexing, and subsetting** of large data sets
# - Intuitive **merging and joining** data sets
# - Flexible **reshaping and pivoting** of data sets
# - **Hierarchical labeling** of axes
# - Robust **IO tools** for loading data from flat files, Excel files, databases, and HDF5
# - **Time series functionality**: date range generation and frequency conversion, moving window statistics, moving window linear regressions, date shifting and lagging, etc.

# In[1]:

from IPython.core.display import HTML
HTML("<iframe src=http://pandas.pydata.org width=800 height=350></iframe>")


# In[3]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import pyreadline

# Set some Pandas options
pd.set_option('html', True)
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)


# ## Pandas Data Structures

# ### Series
# 
# A **Series** is a single vector of data (like a NumPy array) with an *index* that labels each element in the vector.

# In[3]:

counts = pd.Series([632, 1638, 569, 115])
counts


# If an index is not specified, a default sequence of integers is assigned as the index. A NumPy array comprises the values of the `Series`, while the index is a pandas `Index` object.

# In[4]:

counts.values


# In[5]:

counts.index


# We can assign meaningful labels to the index, if they are available:

# In[2]:

bacteria = pd.Series([632, 1638, 569, 115], 
    index=['Firmicutes', 'Proteobacteria', 'Actinobacteria', 'Bacteroidetes'])

bacteria


# These labels can be used to refer to the values in the `Series`.

# In[7]:

bacteria['Actinobacteria']


# In[8]:

bacteria[[name.endswith('bacteria') for name in bacteria.index]]


# In[9]:

[name.endswith('bacteria') for name in bacteria.index]


# Notice that the indexing operation preserved the association between the values and the corresponding indices.
# 
# We can still use positional indexing if we wish.

# In[10]:




# We can give both the array of values and the index meaningful labels themselves:

# In[11]:

bacteria.name = 'counts'
bacteria.index.name = 'phylum'
bacteria


# NumPy's math functions and other operations can be applied to Series without losing the data structure.

# In[12]:




# We can also filter according to the values in the `Series`:

# In[13]:

bacteria[bacteria>1000]


# A `Series` can be thought of as an ordered key-value store. In fact, we can create one from a `dict`:

# In[14]:

bacteria_dict = {'Firmicutes': 632, 'Proteobacteria': 1638, 'Actinobacteria': 569, 'Bacteroidetes': 115}
pd.Series(bacteria_dict)


# Notice that the `Series` is created in key-sorted order.
# 
# If we pass a custom index to `Series`, it will select the corresponding values from the dict, and treat indices without corrsponding values as missing. Pandas uses the `NaN` (not a number) type for missing values.

# In[15]:

bacteria2 = pd.Series(bacteria_dict, index=['Cyanobacteria','Firmicutes','Proteobacteria','Actinobacteria'])
bacteria2


# In[16]:

bacteria2.isnull()


# Critically, the labels are used to **align data** when used in operations with other Series objects:

# In[17]:

bacteria + bacteria2


# Contrast this with NumPy arrays, where arrays of the same length will combine values element-wise; adding Series combined values with the same label in the resulting series. Notice also that the missing values were propogated by addition.

# ### DataFrame
# 
# Inevitably, we want to be able to store, view and manipulate data that is *multivariate*, where for every index there are multiple fields or columns of data, often of varying data type.
# 
# A `DataFrame` is a tabular data structure, encapsulating multiple series like columns in a spreadsheet. Data are stored internally as a 2-dimensional object, but the `DataFrame` allows us to represent and manipulate higher-dimensional data.

# In[4]:

data = pd.DataFrame({'value':[632, 1638, 569, 115, 433, 1130, 754, 555],
                     'patient':[1, 1, 1, 1, 2, 2, 2, 2],
                     'phylum':['Firmicutes', 'Proteobacteria', 'Actinobacteria', 
    'Bacteroidetes', 'Firmicutes', 'Proteobacteria', 'Actinobacteria', 'Bacteroidetes']})
data


# Notice the `DataFrame` is sorted by column name. We can change the order by indexing them in the order we desire:

# In[19]:

data[['phylum','value','patient']]


# A `DataFrame` has a second index, representing the columns:

# In[20]:

data.columns


# If we wish to access columns, we can do so either by dict-like indexing or by attribute:

# In[8]:

data[['value','patient']]


# In[6]:

data.value


# In[23]:

type(data.value)


# In[24]:

type(data[['value']])


# Notice this is different than with `Series`, where dict-like indexing retrieved a particular element (row). If we want access to a row in a `DataFrame`, we index its `ix` attribute.
# 

# In[25]:

data.ix[3]


# Alternatively, we can create a `DataFrame` with a dict of dicts:

# In[10]:

data = pd.DataFrame({0: {'patient': 1, 'phylum': 'Firmicutes', 'value': 632},
                    1: {'patient': 1, 'phylum': 'Proteobacteria', 'value': 1638},
                    2: {'patient': 1, 'phylum': 'Actinobacteria', 'value': 569},
                    3: {'patient': 1, 'phylum': 'Bacteroidetes', 'value': 115},
                    4: {'patient': 2, 'phylum': 'Firmicutes', 'value': 433},
                    5: {'patient': 2, 'phylum': 'Proteobacteria', 'value': 1130},
                    6: {'patient': 2, 'phylum': 'Actinobacteria', 'value': 754},
                    7: {'patient': 2, 'phylum': 'Bacteroidetes', 'value': 555}})


# In[11]:

data


# We probably want this transposed:

# In[9]:

data = data.T
data


# Its important to note that the Series returned when a DataFrame is indexted is merely a **view** on the DataFrame, and not a copy of the data itself. So you must be cautious when manipulating this data:

# In[29]:

vals = data.value
vals


# In[12]:

vals[5] = 0
vals


# In[13]:

data


# In[32]:

vals = data.value.copy()
vals[5] = 1000
data


# We can create or modify columns by assignment:

# In[33]:

data.value[3] = 14
data


# In[34]:

data['year'] = 2013
data


# But note, we cannot use the attribute indexing method to add a new column:

# In[35]:

data.treatment = 1
data


# In[36]:

data.treatment


# Specifying a `Series` as a new columns cause its values to be added according to the `DataFrame`'s index:

# In[37]:

treatment = pd.Series([0]*4 + [1]*2)
treatment


# In[38]:

data['treatment'] = treatment
data


# Other Python data structures (ones without an index) need to be the same length as the `DataFrame`:

# In[39]:

month = ['Jan', 'Feb', 'Mar', 'Apr']
data['month'] = month


# In[40]:

data['month'] = ['Jan']*len(data)
data


# We can use `del` to remove columns, in the same way `dict` entries can be removed:

# In[41]:

del data['month']
data


# We can extract the underlying data as a simple `ndarray` by accessing the `values` attribute:

# In[42]:

data.values


# Notice that because of the mix of string and integer (and `NaN`) values, the dtype of the array is `object`. The dtype will automatically be chosen to be as general as needed to accomodate all the columns.

# In[43]:

df = pd.DataFrame({'foo': [1,2,3], 'bar':[0.4, -1.0, 4.5]})
df.values


# Pandas uses a custom data structure to represent the indices of Series and DataFrames.

# In[44]:

data.index


# Index objects are immutable:

# In[45]:

data.index[0] = 15


# This is so that Index objects can be shared between data structures without fear that they will be changed.

# In[46]:

bacteria2.index = bacteria.index


# In[47]:

bacteria2


# ## Importing data

# A key, but often under-appreciated, step in data analysis is importing the data that we wish to analyze. Though it is easy to load basic data structures into Python using built-in tools or those provided by packages like NumPy, it is non-trivial to import structured data well, and to easily convert this input into a robust data structure:
# 
#     genes = np.loadtxt("genes.csv", delimiter=",", dtype=[('gene', '|S10'), ('value', '<f4')])
# 
# Pandas provides a convenient set of functions for importing tabular data in a number of formats directly into a `DataFrame` object. These functions include a slew of options to perform type inference, indexing, parsing, iterating and cleaning automatically as data are imported.

# Let's start with some more bacteria data, stored in csv format.

# In[48]:

get_ipython().system(u'cat data/microbiome.csv')


# This table can be read into a DataFrame using `read_csv`:

# In[49]:

mb = pd.read_csv("data/microbiome.csv")
mb


# Notice that `read_csv` automatically considered the first row in the file to be a header row.
# 
# We can override default behavior by customizing some the arguments, like `header`, `names` or `index_col`.

# In[50]:

pd.read_csv("data/microbiome.csv", header=None).head()


# `read_csv` is just a convenience function for `read_table`, since csv is such a common format:

# In[51]:

mb = pd.read_table("data/microbiome.csv", sep=',')


# The `sep` argument can be customized as needed to accomodate arbitrary separators. For example, we can use a regular expression to define a variable amount of whitespace, which is unfortunately very common in some data formats: 
#     
#     sep='\s+'

# For a more useful index, we can specify the first two columns, which together provide a unique index to the data.

# In[52]:

mb = pd.read_csv("data/microbiome.csv", index_col=['Taxon','Patient'])
mb.head()


# This is called a *hierarchical* index, which we will revisit later in the tutorial.

# If we have sections of data that we do not wish to import (for example, known bad data), we can populate the `skiprows` argument:

# In[53]:

pd.read_csv("data/microbiome.csv", skiprows=[3,4,6]).head()


# Conversely, if we only want to import a small number of rows from, say, a very large data file we can use `nrows`:

# In[54]:

pd.read_csv("data/microbiome.csv", nrows=4)


# Alternately, if we want to process our data in reasonable chunks, the `chunksize` argument will return an iterable object that can be employed in a data processing loop. For example, our microbiome data are organized by bacterial phylum, with 15 patients represented in each:

# In[55]:

data_chunks = pd.read_csv("data/microbiome.csv", chunksize=15)

mean_tissue = {chunk.Taxon[0]:chunk.Tissue.mean() for chunk in data_chunks}
    
mean_tissue


# Most real-world data is incomplete, with values missing due to incomplete observation, data entry or transcription error, or other reasons. Pandas will automatically recognize and parse common missing data indicators, including `NA` and `NULL`.

# In[56]:

get_ipython().system(u'cat data/microbiome_missing.csv')


# In[57]:

pd.read_csv("data/microbiome_missing.csv").head(20)


# Above, Pandas recognized `NA` and an empty field as missing data.

# In[58]:

pd.isnull(pd.read_csv("data/microbiome_missing.csv")).head(20)


# Unfortunately, there will sometimes be inconsistency with the conventions for missing data. In this example, there is a question mark "?" and a large negative number where there should have been a positive integer. We can specify additional symbols with the `na_values` argument:
#    

# In[59]:

pd.read_csv("data/microbiome_missing.csv", na_values=['?', -99999]).head(20)


# These can be specified on a column-wise basis using an appropriate dict as the argument for `na_values`.

# ### Microsoft Excel
# 
# Since so much financial and scientific data ends up in Excel spreadsheets (regrettably), Pandas' ability to directly import Excel spreadsheets is valuable. This support is contingent on having one or two dependencies (depending on what version of Excel file is being imported) installed: `xlrd` and `openpyxl` (these may be installed with either `pip` or `easy_install`).
# 
# Importing Excel data to Pandas is a two-step process. First, we create an `ExcelFile` object using the path of the file:                                             

# In[60]:

mb_file = pd.ExcelFile('data/microbiome/MID1.xls')
mb_file


# Then, since modern spreadsheets consist of one or more "sheets", we parse the sheet with the data of interest:

# In[61]:

mb1 = mb_file.parse("Sheet 1", header=None)
mb1.columns = ["Taxon", "Count"]
mb1.head()


# There is now a `read_excel` conveneince function in Pandas that combines these steps into a single call:

# In[62]:

mb2 = pd.read_excel('data/microbiome/MID2.xls', sheetname='Sheet 1', header=None)
mb2.head()


# There are several other data formats that can be imported into Python and converted into DataFrames, with the help of buitl-in or third-party libraries. These include JSON, XML, HDF5, relational and non-relational databases, and various web APIs. These are beyond the scope of this tutorial, but are covered in [Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do).

# ## Pandas Fundamentals

# This section introduces the new user to the key functionality of Pandas that is required to use the software effectively.
# 
# For some variety, we will leave our digestive tract bacteria behind and employ some baseball data.

# In[63]:

baseball = pd.read_csv("data/baseball.csv", index_col='id')
baseball.head()


# Notice that we specified the `id` column as the index, since it appears to be a unique identifier. We could try to create a unique index ourselves by combining `player` and `year`:

# In[64]:

player_id = baseball.player + baseball.year.astype(str)
baseball_newind = baseball.copy()
baseball_newind.index = player_id
baseball_newind.head()


# This looks okay, but let's check:

# In[65]:

baseball_newind.index.is_unique


# So, indices need not be unique. Our choice is not unique because some players change teams within years.

# In[66]:

pd.Series(baseball_newind.index).value_counts()


# The most important consequence of a non-unique index is that indexing by label will return multiple values for some labels:

# In[67]:

baseball_newind.ix['wickmbo012007']


# We will learn more about indexing below.

# We can create a truly unique index by combining `player`, `team` and `year`:

# In[68]:

player_unique = baseball.player + baseball.team + baseball.year.astype(str)
baseball_newind = baseball.copy()
baseball_newind.index = player_unique
baseball_newind.head()


# In[69]:

baseball_newind.index.is_unique


# We can create meaningful indices more easily using a hierarchical index; for now, we will stick with the numeric `id` field as our index.

# ### Manipulating indices
# 
# **Reindexing** allows users to manipulate the data labels in a DataFrame. It forces a DataFrame to conform to the new index, and optionally, fill in missing data if requested.
# 
# A simple use of `reindex` is to alter the order of the rows:

# In[70]:

baseball.reindex(baseball.index[::-1]).head()


# Notice that the `id` index is not sequential. Say we wanted to populate the table with every `id` value. We could specify and index that is a sequence from the first to the last `id` numbers in the database, and Pandas would fill in the missing data with `NaN` values:

# In[71]:

id_range = range(baseball.index.values.min(), baseball.index.values.max())
baseball.reindex(id_range).head()


# Missing values can be filled as desired, either with selected values, or by rule:

# In[72]:

baseball.reindex(id_range, method='ffill', columns=['player','year']).head()


# In[73]:

baseball.reindex(id_range, fill_value='mr.nobody', columns=['player']).head()


# Keep in mind that `reindex` does not work if we pass a non-unique index series.

# We can remove rows or columns via the `drop` method:

# In[74]:

baseball.shape


# In[75]:

baseball.drop([89525, 89526])


# In[76]:

baseball.drop(['ibb','hbp'], axis=1)


# ## Indexing and Selection
# 
# Indexing works analogously to indexing in NumPy arrays, except we can use the labels in the `Index` object to extract values in addition to arrays of integers.

# In[77]:

# Sample Series object
hits = baseball_newind.h
hits


# In[78]:

# Numpy-style indexing
hits[:3]


# In[79]:

# Indexing by label
hits[['womacto01CHN2006','schilcu01BOS2006']]


# We can also slice with data labels, since they have an intrinsic order within the Index:

# In[80]:

hits['womacto01CHN2006':'gonzalu01ARI2006']


# In[81]:

hits['womacto01CHN2006':'gonzalu01ARI2006'] = 5
hits


# In a `DataFrame` we can slice along either or both axes:

# In[82]:

baseball_newind[['h','ab']]


# In[83]:

baseball_newind[baseball_newind.ab>500] baseball_newind[ab>500] 


# The indexing field `ix` allows us to select subsets of rows and columns in an intuitive way:

# In[84]:

baseball_newind.ix['gonzalu01ARI2006', ['h','X2b', 'X3b', 'hr']]


# In[85]:

baseball_newind.ix[['gonzalu01ARI2006','finlest01SFN2006'], 5:8]


# In[86]:

baseball_newind.ix[:'myersmi01NYA2006', 'hr']


# Similarly, the cross-section method `xs` (not a field) extracts a single column or row *by label* and returns it as a `Series`:

# In[87]:

baseball_newind.xs('myersmi01NYA2006')


# ## Operations
# 
# `DataFrame` and `Series` objects allow for several operations to take place either on a single object, or between two or more objects.
# 
# For example, we can perform arithmetic on the elements of two objects, such as combining baseball statistics across years:

# In[88]:

hr2006 = baseball[baseball.year==2006].xs('hr', axis=1)
hr2006.index = baseball.player[baseball.year==2006]

hr2007 = baseball[baseball.year==2007].xs('hr', axis=1)
hr2007.index = baseball.player[baseball.year==2007]


# In[89]:

hr2006 = pd.Series(baseball.hr[baseball.year==2006].values, index=baseball.player[baseball.year==2006])
hr2007 = pd.Series(baseball.hr[baseball.year==2007].values, index=baseball.player[baseball.year==2007])


# In[90]:

hr_total = hr2006 + hr2007
hr_total


# Pandas' data alignment places `NaN` values for labels that do not overlap in the two Series. In fact, there are only 6 players that occur in both years.

# In[91]:

hr_total[hr_total.notnull()]


# While we do want the operation to honor the data labels in this way, we probably do not want the missing values to be filled with `NaN`. We can use the `add` method to calculate player home run totals by using the `fill_value` argument to insert a zero for home runs where labels do not overlap:

# In[92]:

hr2007.add(hr2006, fill_value=0)


# Operations can also be **broadcast** between rows or columns.
# 
# For example, if we subtract the maximum number of home runs hit from the `hr` column, we get how many fewer than the maximum were hit by each player:

# In[93]:

baseball.hr - baseball.hr.max()


# Or, looking at things row-wise, we can see how a particular player compares with the rest of the group with respect to important statistics

# In[94]:

baseball.ix[89521]["player"]


# In[95]:

stats = baseball[['h','X2b', 'X3b', 'hr']]


# We can also apply functions to each column or row of a `DataFrame`

# In[96]:

#START HERE
stats.apply(np.median)


# In[97]:

stat_range = lambda x: x.max() - x.min()
stats.apply(stat_range)


# Lets use apply to calculate a meaningful baseball statistics, slugging percentage:
# 
# $$SLG = \frac{1B + (2 \times 2B) + (3 \times 3B) + (4 \times HR)}{AB}$$
# 
# And just for fun, we will format the resulting estimate.

# In[98]:

slg = lambda x: (x['h']-x['X2b']-x['X3b']-x['hr'] + 2*x['X2b'] + 3*x['X3b'] + 4*x['hr'])/(x['ab']+1e-6)
baseball.apply(slg, axis=1).apply(lambda x: '%.3f' % x)


# ## Sorting and Ranking
# 
# Pandas objects include methods for re-ordering data.

# In[99]:

baseball_newind.sort_index().head()


# In[100]:

baseball_newind.sort_index(ascending=False).head()


# In[101]:

baseball_newind.sort_index(axis=1).head()


# We can also use `order` to sort a `Series` by value, rather than by label.

# In[102]:

baseball.hr.order(ascending=False)


# For a `DataFrame`, we can sort according to the values of one or more columns using the `by` argument of `sort_index`:

# In[103]:

baseball[['player','sb','cs']].sort_index(ascending=[False,True], by=['sb', 'cs']).head(10)


# **Ranking** does not re-arrange data, but instead returns an index that ranks each value relative to others in the Series.

# In[104]:

baseball.hr.rank()


# Ties are assigned the mean value of the tied ranks, which may result in decimal values.

# In[105]:

pd.Series([100,100]).rank()


# Alternatively, you can break ties via one of several methods, such as by the order in which they occur in the dataset:

# In[106]:

baseball.hr.rank(method='first')


# Calling the `DataFrame`'s `rank` method results in the ranks of all columns:

# In[107]:

baseball.rank(ascending=False).head()


# In[108]:

baseball[['r','h','hr']].rank(ascending=False).head()


# ### Exercise
# 
# Calculate **on base percentage** for each player, and return the ordered series of estimates.
# 
# $$OBP = \frac{H + BB + HBP}{AB + BB + HBP + SF}$$

# In[108]:

# Write your answer here


# ## Hierarchical indexing
# 
# In the baseball example, I was forced to combine 3 fields to obtain a unique index that was not simply an integer value. A more elegant way to have done this would be to create a hierarchical index from the three fields.

# In[109]:

baseball_h = baseball.set_index(['year', 'team', 'player'])
baseball_h.head(10)


# This index is a `MultiIndex` object that consists of a sequence of tuples, the elements of which is some combination of the three columns used to create the index. Where there are multiple repeated values, Pandas does not print the repeats, making it easy to identify groups of values.

# In[110]:

baseball_h.index[:10]


# In[111]:

baseball_h.index.is_unique


# In[112]:

baseball_h.ix[(2007, 'ATL', 'francju01')]


# Recall earlier we imported some microbiome data using two index columns. This created a 2-level hierarchical index:

# In[113]:

mb = pd.read_csv("data/microbiome.csv", index_col=['Taxon','Patient'])


# In[114]:

mb.head(10)


# In[115]:

mb.index


# With a hierachical index, we can select subsets of the data based on a *partial* index:

# In[116]:

mb.ix['Proteobacteria']


# Hierarchical indices can be created on either or both axes. Here is a trivial example:

# In[117]:

frame = pd.DataFrame(np.arange(12).reshape(( 4, 3)), 
                  index =[['a', 'a', 'b', 'b'], [1, 2, 1, 2]], 
                  columns =[['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])

frame


# If you want to get fancy, both the row and column indices themselves can be given names:

# In[118]:

frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame


# With this, we can do all sorts of custom indexing:

# In[119]:

frame.ix['a']['Ohio']


# In[120]:

frame.ix['b', 2]['Colorado']


# Additionally, the order of the set of indices in a hierarchical `MultiIndex` can be changed by swapping them pairwise:

# In[121]:

mb.swaplevel('Patient', 'Taxon').head()


# Data can also be sorted by any index level, using `sortlevel`:

# In[122]:

mb.sortlevel('Patient', ascending=False).head()


# ## Missing data
# 
# The occurence of missing data is so prevalent that it pays to use tools like Pandas, which seamlessly integrates missing data handling so that it can be dealt with easily, and in the manner required by the analysis at hand.
# 
# Missing data are represented in `Series` and `DataFrame` objects by the `NaN` floating point value. However, `None` is also treated as missing, since it is commonly used as such in other contexts (*e.g.* NumPy).

# In[130]:

foo = pd.Series([np.nan, -3, None, 'foobar'])
foo


# In[131]:

foo.isnull()


# Missing values may be dropped or indexed out:

# In[132]:

bacteria2


# In[133]:

bacteria2.dropna()


# In[134]:

bacteria2[bacteria2.notnull()]


# By default, `dropna` drops entire rows in which one or more values are missing.

# In[135]:

data


# In[136]:

data.dropna()


# This can be overridden by passing the `how='all'` argument, which only drops a row when every field is a missing value.

# In[137]:

data.dropna(how='all')


# This can be customized further by specifying how many values need to be present before a row is dropped via the `thresh` argument.

# In[140]:

data.ix[7, 'year'] = np.nan
data


# In[141]:

data.dropna(thresh=4)


# This is typically used in time series applications, where there are repeated measurements that are incomplete for some subjects.

# If we want to drop missing values column-wise instead of row-wise, we use `axis=1`.

# In[142]:

data.dropna(axis=1)


# Rather than omitting missing data from an analysis, in some cases it may be suitable to fill the missing value in, either with a default value (such as zero) or a value that is either imputed or carried forward/backward from similar data points. We can do this programmatically in Pandas with the `fillna` argument.

# In[143]:

bacteria2.fillna(0)


# In[144]:

data.fillna({'year': 2013, 'treatment':2})


# Notice that `fillna` by default returns a new object with the desired filling behavior, rather than changing the `Series` or  `DataFrame` in place (**in general, we like to do this, by the way!**).

# In[145]:

data


# We can alter values in-place using `inplace=True`.

# In[146]:

data.year.fillna(2013, inplace=True)
data


# Missing values can also be interpolated, using any one of a variety of methods:

# In[147]:

bacteria2.fillna(method='bfill')


# In[148]:

bacteria2.fillna(bacteria2.mean())


# ## Data summarization
# 
# We often wish to summarize data in `Series` or `DataFrame` objects, so that they can more easily be understood or compared with similar data. The NumPy package contains several functions that are useful here, but several summarization or reduction methods are built into Pandas data structures.

# In[149]:

baseball.sum()


# Clearly, `sum` is more meaningful for some columns than others. For methods like `mean` for which application to string variables is not just meaningless, but impossible, these columns are automatically exculded:

# In[150]:

baseball.mean()


# The important difference between NumPy's functions and Pandas' methods is that the latter have built-in support for handling missing data.

# In[151]:

bacteria2


# In[152]:

bacteria2.mean()


# Sometimes we may not want to ignore missing values, and allow the `nan` to propagate.

# In[153]:

bacteria2.mean(skipna=False)


# Passing `axis=1` will summarize over rows instead of columns, which only makes sense in certain situations.

# In[154]:

extra_bases = baseball[['X2b','X3b','hr']].sum(axis=1)
extra_bases.order(ascending=False)


# A useful summarization that gives a quick snapshot of multiple statistics for a `Series` or `DataFrame` is `describe`:

# In[155]:

baseball.describe()


# `describe` can detect non-numeric data and sometimes yield useful information about it.

# In[156]:

baseball.player.describe()


# We can also calculate summary statistics *across* multiple columns, for example, correlation and covariance.
# 
# $$cov(x,y) = \sum_i (x_i - \bar{x})(y_i - \bar{y})$$

# In[157]:

baseball.hr.cov(baseball.X2b)


# $$corr(x,y) = \frac{cov(x,y)}{(n-1)s_x s_y} = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2 \sum_i (y_i - \bar{y})^2}}$$

# In[158]:

baseball.hr.corr(baseball.X2b)


# In[159]:

baseball.ab.corr(baseball.h)


# In[160]:

baseball.corr()


# If we have a `DataFrame` with a hierarchical index (or indices), summary statistics can be applied with respect to any of the index levels:

# In[161]:

mb.head()


# In[162]:

mb.sum(level='Taxon')


# ## Writing Data to Files
# 
# As well as being able to read several data input formats, Pandas can also export data to a variety of storage formats. We will bring your attention to just a couple of these.

# In[163]:

mb.to_csv("mb.csv")


# The `to_csv` method writes a `DataFrame` to a comma-separated values (csv) file. You can specify custom delimiters (via `sep` argument), how missing values are written (via `na_rep` argument), whether the index is writen (via `index` argument), whether the header is included (via `header` argument), among other options.

# An efficient way of storing data to disk is in binary format. Pandas supports this using Python’s built-in pickle serialization.

# In[164]:

baseball.to_pickle("baseball_pickle")


# The complement to `to_pickle` is the `read_pickle` function, which restores the pickle to a `DataFrame` or `Series`:

# In[165]:

pd.read_pickle("baseball_pickle")


# As Wes warns in his book, it is recommended that binary storage of data via pickle only be used as a temporary storage format, in situations where speed is relevant. This is because there is no guarantee that the pickle format will not change with future versions of Python.
