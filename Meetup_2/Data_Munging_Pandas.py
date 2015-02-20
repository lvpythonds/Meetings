
# coding: utf-8

# # Data Wrangling with Pandas
# 
# Now that we have been exposed to the basic functionality of Pandas, lets explore some more advanced features that will be useful when addressing more complex data management tasks.
# 
# As most statisticians/data analysts will admit, often the lion's share of the time spent implementing an analysis is devoted to preparing the data itself, rather than to coding or running a particular model that uses the data. This is where Pandas and  Python's standard library are beneficial, providing high-level, flexible, and efficient tools for manipulating your data as needed.
# 

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set some Pandas options
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 25)


# ## Date/Time data handling
# 
# Date and time data are inherently problematic. There are an unequal number of days in every month, an unequal number of days in a year (due to leap years), and time zones that vary over space. Yet information about time is essential in many analyses, particularly in the case of time series analysis.

# The `datetime` built-in library handles temporal information down to the nanosecond.

# In[2]:

from datetime import datetime


# In[3]:

now = datetime.now()
now


# In[4]:

now.day


# In[5]:

now.weekday()


# In addition to `datetime` there are simpler objects for date and time information only, respectively.

# In[6]:

from datetime import date, time


# In[7]:

time(3, 24)


# In[8]:

date(1970, 9, 3)


# Having a custom data type for dates and times is convenient because we can perform operations on them easily. For example, we may want to calculate the difference between two times:

# In[9]:

my_age = now - datetime(1970, 9, 3)
my_age


# In[10]:

my_age.days/365.


# In this section, we will manipulate data collected from ocean-going vessels on the eastern seaboard. Vessel operations are monitored using the Automatic Identification System (AIS), a safety at sea navigation technology which vessels are required to maintain and that uses transponders to transmit very high frequency (VHF) radio signals containing static information including ship name, call sign, and country of origin, as well as dynamic information unique to a particular voyage such as vessel location, heading, and speed. 
# 
# The International Maritime Organization’s (IMO) International Convention for the Safety of Life at Sea requires functioning AIS capabilities on all vessels 300 gross tons or greater and the US Coast Guard requires AIS on nearly all vessels sailing in U.S. waters. The Coast Guard has established a national network of AIS receivers that provides coverage of nearly all U.S. waters. AIS signals are transmitted several times each minute and the network is capable of handling thousands of reports per minute and updates as often as every two seconds. Therefore, a typical voyage in our study might include the transmission of hundreds or thousands of AIS encoded signals. This provides a rich source of spatial data that includes both spatial and temporal information.
# 
# For our purposes, we will use summarized data that describes the transit of a given vessel through a particular administrative area. The data includes the start and end time of the transit segment, as well as information about the speed of the vessel, how far it travelled, etc.

# In[21]:

segments = pd.read_csv("C:/Users/Sean Najera/OneDrive/lvpythonds/Meetings/Meetup_2/transit_segments.csv")
segments.head()


# For example, we might be interested in the distribution of transit lengths, so we can plot them as a histogram:

# In[12]:

segments.seg_length.hist(bins=500)


# Though most of the transits appear to be short, there are a few longer distances that make the plot difficult to read. This is where a transformation is useful:

# In[13]:

segments.seg_length.apply(np.log).hist(bins=500)


# We can see that although there are date/time fields in the dataset, they are not in any specialized format, such as `datetime`.

# In[14]:

segments.st_time.dtype


# Our first order of business will be to convert these data to `datetime`. The `strptime` method parses a string representation of a date and/or time field, according to the expected format of this information.

# In[15]:

datetime.strptime(segments.st_time.ix[0], '%m/%d/%y %H:%M')


# The `dateutil` package includes a parser that attempts to detect the format of the date strings, and convert them automatically.

# In[16]:

from dateutil.parser import parse


# In[17]:

parse(segments.st_time.ix[0])


# We can convert all the dates in a particular column by using the `apply` method.

# In[18]:

segments.st_time.apply(lambda d: datetime.strptime(d, '%m/%d/%y %H:%M'))


# As a convenience, Pandas has a `to_datetime` method that will parse and convert an entire Series of formatted strings into `datetime` objects.

# In[19]:

pd.to_datetime(segments.st_time)


# Pandas also has a custom NA value for missing datetime objects, `NaT`.

# In[20]:

pd.to_datetime([None])


# Also, if `to_datetime()` has problems parsing any particular date/time format, you can pass the spec in using the `format=` argument.

# ## Merging and joining DataFrame objects

# Now that we have the vessel transit information as we need it, we may want a little more information regarding the vessels themselves. In the `data/AIS` folder there is a second table that contains information about each of the ships that traveled the segments in the `segments` table.

# In[18]:

vessels = pd.read_csv("C:/Users/Sean Najera/OneDrive/lvpythonds/Meetings/Meetup_2/vessel_information.csv", index_col='mmsi')
vessels.head()


# In[22]:

[v for v in vessels.type.unique() if v.find('/')==-1]


# In[23]:

vessels.type.value_counts()


# The challenge, however, is that several ships have travelled multiple segments, so there is not a one-to-one relationship between the rows of the two tables. The table of vessel information has a *one-to-many* relationship with the segments.
# 
# In Pandas, we can combine tables according to the value of one or more *keys* that are used to identify rows, much like an index. Using a trivial example:

# In[10]:

df1 = pd.DataFrame(dict(id=range(4), age=np.random.randint(18, 31, size=4)))
df2 = pd.DataFrame(dict(id=range(3)+range(3), score=np.random.random(size=6)))

df1


# In[11]:

pd.merge(df1, df2)


# Notice that without any information about which column to use as a key, Pandas did the right thing and used the `id` column in both tables. Unless specified otherwise, `merge` will used any common column names as keys for merging the tables. 
# 
# Notice also that `id=3` from `df1` was omitted from the merged table. This is because, by default, `merge` performs an **inner join** on the tables, meaning that the merged table represents an intersection of the two tables.

# In[12]:

pd.merge(df1, df2, how='outer')


# The **outer join** above yields the union of the two tables, so all rows are represented, with missing values inserted as appropriate. One can also perform **right** and **left** joins to include all rows of the right or left table (*i.e.* first or second argument to `merge`), but not necessarily the other.

# Looking at the two datasets that we wish to merge:

# In[22]:

segments.head(1)


# In[19]:

vessels.head(1)


# we see that there is a `mmsi` value (a vessel identifier) in each table, but it is used as an index for the `vessels` table. In this case, we have to specify to join on the index for this table, and on the `mmsi` column for the other.

# In[23]:

segments_merged = pd.merge(vessels, segments, left_index=True, right_on='mmsi')


# In[24]:

segments_merged.head()


# In this case, the default inner join is suitable; we are not interested in observations from either table that do not have corresponding entries in the other. 
# 
# Notice that `mmsi` field that was an index on the `vessels` table is no longer an index on the merged table.

# Here, we used the `merge` function to perform the merge; we could also have used the `merge` method for either of the tables:

# In[25]:

vessels.merge(segments, left_index=True, right_on='mmsi').head()


# Occasionally, there will be fields with the same in both tables that we do not wish to use to join the tables; they may contain different information, despite having the same name. In this case, Pandas will by default append suffixes `_x` and `_y` to the columns to uniquely identify them.

# In[26]:

segments['type'] = 'foo'
pd.merge(vessels, segments, left_index=True, right_on='mmsi').head()


# This behavior can be overridden by specifying a `suffixes` argument, containing a list of the suffixes to be used for the columns of the left and right columns, respectively.

# ## Concatenation
# 
# A common data manipulation is appending rows or columns to a dataset that already conform to the dimensions of the exsiting rows or colums, respectively. In NumPy, this is done either with `concatenate` or the convenience functions `c_` and `r_`:

# In[33]:

np.concatenate([np.random.random(5), np.random.random(5)])


# In[34]:

np.r_[np.random.random(5), np.random.random(5)]


# In[35]:

np.c_[np.random.random(5), np.random.random(5)]


# This operation is also called *binding* or *stacking*.
# 
# With Pandas' indexed data structures, there are additional considerations as the overlap in index values between two data structures affects how they are concatenate.
# 
# Lets import two microbiome datasets, each consisting of counts of microorganiams from a particular patient. We will use the first column of each dataset as the index.

# In[27]:

mb1 = pd.read_excel('C:/Users/Sean Najera/OneDrive/lvpythonds/Meetings/Meetup_2/MID1.xls', 'Sheet 1', index_col=0, header=None)
mb2 = pd.read_excel('C:/Users/Sean Najera/OneDrive/lvpythonds/Meetings/Meetup_2/MID2.xls', 'Sheet 1', index_col=0, header=None)
mb1.shape, mb2.shape


# In[28]:

mb1.head()


# Let's give the index and columns meaningful labels:

# In[29]:

mb1.columns = mb2.columns = ['Count']


# In[30]:

mb1.index.name = mb2.index.name = 'Taxon'


# In[31]:

mb1.head()


# The index of these data is the unique biological classification of each organism, beginning with *domain*, *phylum*, *class*, and for some organisms, going all the way down to the genus level.
# 
# ![classification](http://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Biological_classification_L_Pengo_vflip.svg/150px-Biological_classification_L_Pengo_vflip.svg.png)

# In[41]:

mb1.index[:3]


# In[42]:

mb1.index.is_unique


# If we concatenate along `axis=0` (the default), we will obtain another data frame with the the rows concatenated:

# In[43]:

pd.concat([mb1, mb2], axis=0).shape


# However, the index is no longer unique, due to overlap between the two DataFrames.

# In[44]:

pd.concat([mb1, mb2], axis=0).index.is_unique


# Concatenating along `axis=1` will concatenate column-wise, but respecting the indices of the two DataFrames.

# In[45]:

pd.concat([mb1, mb2], axis=1).shape


# In[32]:

pd.concat([mb1, mb2], axis=1).head()


# In[47]:

pd.concat([mb1, mb2], axis=1).values[:5]


# If we are only interested in taxa that are included in both DataFrames, we can specify a `join=inner` argument.

# In[33]:

pd.concat([mb1, mb2], axis=1, join='inner').head()


# If we wanted to use the second table to fill values absent from the first table, we could use `combine_first`.

# In[36]:

mb1.combine_first(mb2).head()


# We can also create a hierarchical index based on keys identifying the original tables.

# In[37]:

pd.concat([mb1, mb2], keys=['patient1', 'patient2']).head()


# In[51]:

pd.concat([mb1, mb2], keys=['patient1', 'patient2']).index.is_unique


# Alternatively, you can pass keys to the concatenation by supplying the DataFrames (or Series) as a dict.

# In[38]:

pd.concat(dict(patient1=mb1, patient2=mb2), axis=1).head()


# If you want `concat` to work like `numpy.concatanate`, you may provide the `ignore_index=True` argument.

# ## Exercise
# 
# In the *data/microbiome* subdirectory, there are 9 spreadsheets of microbiome data that was acquired from high-throughput RNA sequencing procedures, along with a 10th file that describes the content of each. Write code that imports each of the data spreadsheets and combines them into a single `DataFrame`, adding the identifying information from the metadata spreadsheet as columns in the combined `DataFrame`.

# In[52]:

# Write your answer here


# ## Reshaping DataFrame objects
# 
# In the context of a single DataFrame, we are often interested in re-arranging the layout of our data. 

# This dataset in from Table 6.9 of [Statistical Methods for the Analysis of Repeated Measurements](http://www.amazon.com/Statistical-Methods-Analysis-Repeated-Measurements/dp/0387953701) by Charles S. Davis, pp. 161-163 (Springer, 2002). These data are from a multicenter, randomized controlled trial of botulinum toxin type B (BotB) in patients with cervical dystonia from nine U.S. sites.
# 
# * Randomized to placebo (N=36), 5000 units of BotB (N=36), 10,000 units of BotB (N=37)
# * Response variable: total score on Toronto Western Spasmodic Torticollis Rating Scale (TWSTRS), measuring severity, pain, and disability of cervical dystonia (high scores mean more impairment)
# * TWSTRS measured at baseline (week 0) and weeks 2, 4, 8, 12, 16 after treatment began

# In[39]:

cdystonia = pd.read_csv("C:/Users/Sean Najera/OneDrive/lvpythonds/Meetings/Meetup_2/cdystonia.csv", index_col=None)
cdystonia.head()


# This dataset includes repeated measurements of the same individuals (longitudinal data). Its possible to present such information in (at least) two ways: showing each repeated measurement in their own row, or in multiple columns representing mutliple measurements.
# 

# The `stack` method rotates the data frame so that columns are represented in rows:

# In[40]:

stacked = cdystonia.stack()
stacked


# To complement this, `unstack` pivots from rows back to columns.

# In[41]:

stacked.unstack().head()


# For this dataset, it makes sense to create a hierarchical index based on the patient and observation:

# In[42]:

cdystonia2 = cdystonia.set_index(['patient','obs'])
cdystonia2.head()


# In[57]:

cdystonia2.index.is_unique


# If we want to transform this data so that repeated measurements are in columns, we can `unstack` the `twstrs` measurements according to `obs`.

# In[43]:

twstrs_wide = cdystonia2['twstrs'].unstack('obs')
twstrs_wide.head()


# In[44]:

cdystonia_long = cdystonia[['patient','site','id','treat','age','sex']].drop_duplicates().merge(
                    twstrs_wide, right_index=True, left_on='patient', how='inner').head()
cdystonia_long


# A slightly cleaner way of doing this is to set the patient-level information as an index before unstacking:

# In[45]:

cdystonia.set_index(['patient','site','id','treat','age','sex','week'])['twstrs'].unstack('week').head()


# To convert our "wide" format back to long, we can use the `melt` function, appropriately parameterized:

# In[61]:

pd.melt(cdystonia_long, id_vars=['patient','site','id','treat','age','sex'], 
        var_name='obs', value_name='twsters').head()


# This illustrates the two formats for longitudinal data: **long** and **wide** formats. Its typically better to store data in long format because additional data can be included as additional rows in the database, while wide format requires that the entire database schema be altered by adding columns to every row as data are collected.
# 
# The preferable format for analysis depends entirely on what is planned for the data, so it is imporant to be able to move easily between them.

# ## Pivoting
# 
# The `pivot` method allows a DataFrame to be transformed easily between long and wide formats in the same way as a pivot table is created in a spreadsheet. It takes three arguments: `index`, `columns` and `values`, corresponding to the DataFrame index (the row headers), columns and cell values, respectively.
# 
# For example, we may want the `twstrs` variable (the response variable) in wide format according to patient:

# In[46]:

cdystonia.pivot(index='patient', columns='obs', values='twstrs').head()


# If we omit the `values` argument, we get a `DataFrame` with hierarchical columns, just as when we applied `unstack` to the hierarchically-indexed table:

# In[47]:

cdystonia.pivot('patient', 'obs')


# A related method, `pivot_table`, creates a spreadsheet-like table with a hierarchical index, and allows the values of the table to be populated using an arbitrary aggregation function.

# In[54]:

cdystonia.pivot_table(rows=['site', 'treat'], columns='week', values='twstrs', aggfunc=max).head(20)


# For a simple cross-tabulation of group frequencies, the `crosstab` function (not a method) aggregates counts of data according to factors in rows and columns. The factors may be hierarchical if desired.

# In[55]:

pd.crosstab(cdystonia.sex, cdystonia.site)


# ## Data transformation
# 
# There are a slew of additional operations for DataFrames that we would collectively refer to as "transformations" that include tasks such as removing duplicate values, replacing values, and grouping values.

# ### Dealing with duplicates
# 
# We can easily identify and remove duplicate values from `DataFrame` objects. For example, say we want to remove ships from our `vessels` dataset that have the same name:

# In[57]:

vessels.duplicated(subset='names') #subset == columns; Searches in order from top to bottom


# In[58]:

vessels.drop_duplicates(['names'])


# ### Value replacement
# 
# Frequently, we get data columns that are encoded as strings that we wish to represent numerically for the purposes of including it in a quantitative analysis. For example, consider the treatment variable in the cervical dystonia dataset:

# In[68]:

cdystonia.treat.value_counts()


# A logical way to specify these numerically is to change them to integer values, perhaps using "Placebo" as a baseline value. If we create a dict with the original values as keys and the replacements as values, we can pass it to the `map` method to implement the changes.

# In[69]:

treatment_map = {'Placebo': 0, '5000U': 1, '10000U': 2}


# In[70]:

cdystonia['treatment'] = cdystonia.treat.map(treatment_map)
cdystonia.treatment


# Alternately, if we simply want to replace particular values in a `Series` or `DataFrame`, we can use the `replace` method. 
# 
# An example where replacement is useful is dealing with zeros in certain transformations. For example, if we try to take the log of a set of values:

# In[59]:

vals = pd.Series([float(i)**10 for i in range(10)])
vals


# In[60]:

np.log(vals)


# In such situations, we can replace the zero with a value so small that it makes no difference to the ensuing analysis. We can do this with `replace`.

# In[62]:

vals = vals.replace(0, 1e-13)
np.log(vals)


# We can also perform the same replacement that we used `map` for with `replace`:

# In[74]:

cdystonia2.treat.replace({'Placebo': 0, '5000U': 1, '10000U': 2})


# ### Inidcator variables
# 
# For some statistical analyses (*e.g.* regression models or analyses of variance), categorical or group variables need to be converted into columns of indicators--zeros and ones--to create a so-called **design matrix**. The Pandas function `get_dummies` (indicator variables are also known as *dummy variables*) makes this transformation straightforward.
# 
# Let's consider the DataFrame containing the ships corresponding to the transit segments on the eastern seaboard. The `type` variable denotes the class of vessel; we can create a matrix of indicators for this. For simplicity, lets filter out the 5 most common types of ships:
# 

# In[75]:

top5 = vessels.type.apply(lambda s: s in vessels.type.value_counts().index[:5])
vessels5 = vessels[top5]


# In[76]:

pd.get_dummies(vessels5.type).head(10)


# ### Discretization
# 
# Pandas' `cut` function can be used to group continuous or countable data in to bins. Discretization is generally a very **bad idea** for statistical analysis, so use this function responsibly!
# 
# Lets say we want to bin the ages of the cervical dystonia patients into a smaller number of groups:

# In[67]:

cdystonia.age.describe()


# Let's transform these data into decades, beginnnig with individuals in their 20's and ending with those in their 90's:

# In[68]:

pd.cut(cdystonia.age, [20,30,40,50,60,70,80,90])[:30]


# The parentheses indicate an open interval, meaning that the interval includes values up to but *not including* the endpoint, whereas the square bracket is a closed interval, where the endpoint is included in the interval. We can switch the closure to the left side by setting the `right` flag to `False`:

# In[69]:

pd.cut(cdystonia.age, [20,30,40,50,60,70,80,90], right=False)[:30]


# Since the data are now **ordinal**, rather than numeric, we can give them labels:

# In[80]:

pd.cut(cdystonia.age, [20,40,60,80,90], labels=['young','middle-aged','old','ancient'])[:30]


# A related function `qcut` uses empirical quantiles to divide the data. If, for example, we want the quartiles -- (0-25%], (25-50%], (50-70%], (75-100%] -- we can just specify 4 intervals, which will be equally-spaced by default:

# In[81]:

pd.qcut(cdystonia.age, 4)[:30]


# Alternatively, one can specify custom quantiles to act as cut points:

# In[82]:

quantiles = pd.qcut(segments.seg_length, [0, 0.01, 0.05, 0.95, 0.99, 1])
quantiles[:30]


# Note that you can easily combine discretiztion with the generation of indicator variables shown above:

# In[83]:

pd.get_dummies(quantiles).head(10)


# ### Permutation and sampling
# 
# For some data analysis tasks, such as simulation, we need to be able to randomly reorder our data, or draw random values from it. Calling NumPy's `permutation` function with the length of the sequence you want to permute generates an array with a permuted sequence of integers, which can be used to re-order the sequence.

# In[71]:

new_order = np.random.permutation(len(segments))
new_order[:30]


# Using this sequence as an argument to the **`take`** method results in a reordered DataFrame:

# In[72]:

segments.take(new_order).head()


# Compare this ordering with the original:

# In[73]:

segments.head()


# ## Exercise
# 
# Its easy to see how this permutation approach allows us to draw a random sample **without replacement**. How would you sample **with replacement**? Generate a random sample of 5 ships from the `vessels` DataFrame using this scheme.

# In[86]:

# Write your answer here


# ## Data aggregation and GroupBy operations
# 
# One of the most powerful features of Pandas is its **GroupBy** functionality. On occasion we may want to perform operations on *groups* of observations within a dataset. For exmaple:
# 
# * **aggregation**, such as computing the sum of mean of each group, which involves applying a function to each group and returning the aggregated results
# * **slicing** the DataFrame into groups and then doing something with the resulting slices (*e.g.* plotting)
# * group-wise **transformation**, such as standardization/normalization

# In[74]:

cdystonia_grouped = cdystonia.groupby(cdystonia.patient)


# This *grouped* dataset is hard to visualize
# 
# 

# In[75]:

cdystonia_grouped


# However, the grouping is only an intermediate step; for example, we may want to **iterate** over each of the patient groups:

# In[76]:

for patient, group in cdystonia_grouped:
    print patient
    print group
    print


# A common data analysis procedure is the **split-apply-combine** operation, which groups subsets of data together, applies a function to each of the groups, then recombines them into a new data table.
# 
# For example, we may want to aggregate our data with with some function.
# 
# ![split-apply-combine](http://f.cl.ly/items/0s0Z252j0X0c3k3P1M47/Screen%20Shot%202013-06-02%20at%203.04.04%20PM.png)
# 
# <div align="right">*(figure taken from "Python for Data Analysis", p.251)*</div>

# We can aggregate in Pandas using the `aggregate` (or `agg`, for short) method:

# In[91]:

cdystonia_grouped.agg(np.mean).head()


# Notice that the `treat` and `sex` variables are not included in the aggregation. Since it does not make sense to aggregate non-string variables, these columns are simply ignored by the method.
# 
# Some aggregation functions are so common that Pandas has a convenience method for them, such as `mean`:

# In[92]:

cdystonia_grouped.mean().head()


# The `add_prefix` and `add_suffix` methods can be used to give the columns of the resulting table labels that reflect the transformation:

# In[93]:

cdystonia_grouped.mean().add_suffix('_mean').head()


# In[94]:

# The median of the `twstrs` variable
cdystonia_grouped['twstrs'].quantile(0.5)


# If we wish, we can easily aggregate according to multiple keys:

# In[77]:

cdystonia.groupby(['week','site']).mean().head()


# Alternately, we can **transform** the data, using a function of our choice with the `transform` method:

# In[96]:

normalize = lambda x: (x - x.mean())/x.std()

cdystonia_grouped.transform(normalize).head()


# It is easy to do column selection within `groupby` operations, if we are only interested split-apply-combine operations on a subset of columns:

# In[78]:

cdystonia_grouped['twstrs'].mean().head()


# In[79]:

# This gives the same result as a DataFrame
cdystonia_grouped[['twstrs']].mean().head()


# If you simply want to divide your DataFrame into chunks for later use, its easy to convert them into a dict so that they can be easily indexed out as needed:

# ### Apply
# 
# We can generalize the split-apply-combine methodology by using `apply` function. This allows us to invoke any function we wish on a grouped dataset and recombine them into a DataFrame.

# The function below takes a DataFrame and a column name, sorts by the column, and takes the `n` largest values of that column. We can use this with `apply` to return the largest values from every group in a DataFrame in a single call. 

# In[83]:

def top(df, column, n=5):
    return df.sort_index(by=column, ascending=False)[:n]


# To see this in action, consider the vessel transit segments dataset (which we merged with the vessel information to yield `segments_merged`). Say we wanted to return the 3 longest segments travelled by each ship:

# In[84]:

top3segments = segments_merged.groupby('mmsi').apply(top, column='seg_length', n=3)[['names', 'seg_length']]
top3segments


# Notice that additional arguments for the applied function can be passed via `apply` after the function name. It assumes that the DataFrame is the first argument.

# In[106]:

top3segments.head(20)


# Recall the microbiome data sets that we used previously for the concatenation example. Suppose that we wish to aggregate the data at a higher biological classification than genus. For example, we can identify samples down to *class*, which is the 3rd level of organization in each index.

# In[107]:

mb1.index[:3]


# Using the string methods `split` and `join` we can create an index that just uses the first three classifications: domain, phylum and class.

# In[108]:

class_index = mb1.index.map(lambda x: ' '.join(x.split(' ')[:3]))


# In[109]:

mb_class = mb1.copy()
mb_class.index = class_index


# However, since there are multiple taxonomic units with the same class, our index is no longer unique:

# In[110]:

mb_class.head()


# We can re-establish a unique index by summing all rows with the same class, using `groupby`:

# In[111]:

mb_class.groupby(level=0).sum().head(10)


# ### Exercise
# 
# Load the dataset in `titanic.xls`. It contains data on all the passengers that travelled on the Titanic.

# In[112]:

from IPython.core.display import HTML
HTML(filename='data/titanic.html')


# Women and children first?
# 
# 1. Use the `groupby` method to calculate the proportion of passengers that survived by sex.
# 2. Calculate the same proportion, but by class and sex.
# 3. Create age categories: children (under 14 years), adolescents (14-20), adult (21-64), and senior(65+), and calculate survival proportions by age category, class and sex.

# In[112]:

# Write your answer here


# In[ ]:



