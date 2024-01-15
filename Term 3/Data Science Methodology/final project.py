#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Part I

# # read data

# In[2]:


data=pd.read_csv(r'C:\Users\Verina\Documents\project data\walmart-sales-dataset-of-45stores.csv')
data


# In[3]:


data.head(10)


# In[4]:


data.tail(10)


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.hist(figsize=(10,10))


# In[9]:


x=data['Weekly_Sales'].max()


# In[10]:


data[data.Weekly_Sales==x]


# In[11]:


y=data['Weekly_Sales'].min()


# In[12]:


data[data.Weekly_Sales==y]


# # Part II

# # data cleaning

# In[13]:


col=data.columns
col


# ## check null data

# In[14]:


data[col].isnull().sum()


# ## check duplicated data

# In[15]:


print(data.duplicated().to_string())


# ## date reformat

# In[16]:


data['Date']=pd.to_datetime(data['Date'])
data


# In[17]:


data1=data.groupby('Store')['Weekly_Sales'].sum().reset_index()
data1


# # Part II

# ### a) Which store has maximum sales?

# In[18]:


plt.figure(figsize=(20,7))
plt.bar(data1['Store'],data1['Weekly_Sales'],color='#a5d1e6') # sns.color_palette('pastel')


# In[19]:


z=data1['Weekly_Sales'].max()
data1[data1.Weekly_Sales==z]


# ### b) Which store has maximum standard deviation i.e., the sales vary a lot

# In[20]:


data2 = data.groupby('Store')['Weekly_Sales'].std().reset_index()
data2.columns = ['Store', 'Standard_Deviation']
data2


# In[21]:


p=data2['Standard_Deviation'].max()
data2[data2.Standard_Deviation==p]


# ### c) Some holidays have a negative impact on sales. Find out holidays that have higher sales than the mean sales in the non-holiday season for all stores together.

# In[22]:


data3=data[data.Holiday_Flag==0]
data3


# In[23]:


mean=data3['Weekly_Sales'].mean()
mean


# In[24]:


data4=data[data.Holiday_Flag==1]
data4


# In[25]:


data4['Month']=data4.agg({'Date':lambda date :(date.month)})
data4


# In[26]:


holiday_mean=data4.groupby('Month')['Weekly_Sales'].mean().reset_index()
holiday_mean


# In[27]:


seg_map={r'7':"Independance Day",
         r'9':"Labor Day",
         r'10':"Halloween",
         r'11':"Thanksgiving",
         r'12':"Christmas"}


# In[28]:


holiday_mean['Holiday']=holiday_mean['Month'].astype(str)
holiday_mean['Holiday']=holiday_mean['Holiday'].replace(seg_map,regex=True)
holiday_mean


# In[29]:


holiday_mean[holiday_mean.Weekly_Sales>mean]


# In[30]:


data5=data[data.Holiday_Flag==1]
data5


# In[33]:


from datetime import date

Christmas1 = pd.Timestamp(date(2010,12,31) )
Christmas2 = pd.Timestamp(date(2011,12,30) )
Christmas3 = pd.Timestamp(date(2012,12,28) )

Thanksgiving1=pd.Timestamp(date(2010,11,26) )
Thanksgiving2=pd.Timestamp(date(2011,11,25) )
Thanksgiving3=pd.Timestamp(date(2012,11,23) )

LabourDay1=pd.Timestamp(date(2010,9,10) )
LabourDay2=pd.Timestamp(date(2011,9,9) )
LabourDay3=pd.Timestamp(date(2012,9,7) )

SuperBowl1=pd.Timestamp(date(2010,2,12) )
SuperBowl2=pd.Timestamp(date(2011,2,11) )
SuperBowl3=pd.Timestamp(date(2012,2,10) )


Christmas_mean_sales=data5[(data5['Date'] == Christmas1) | (data5['Date'] == Christmas2) | (data5['Date'] == Christmas3) ]
Thanksgiving_mean_sales=data5[(data5['Date'] == Thanksgiving1) | (data5['Date'] == Thanksgiving2) | (data5['Date'] == Thanksgiving3) ]
LabourDay_mean_sales=data5[(data5['Date'] == LabourDay1) | (data5['Date'] == LabourDay2) | (data5['Date'] == LabourDay3) ]
SuperBowl_mean_sales=data5[(data5['Date'] == SuperBowl1) | (data5['Date'] == SuperBowl2) | (data5['Date'] == SuperBowl3) ]

Christmas_mean_sales


# In[34]:


means_dict = {'Christmas' : Christmas_mean_sales['Weekly_Sales'].mean(),
              'Thanksgiving' : Thanksgiving_mean_sales['Weekly_Sales'].mean(),
              'Labour Day' : LabourDay_mean_sales['Weekly_Sales'].mean(),
              'Super Bowl' : SuperBowl_mean_sales['Weekly_Sales'].mean()}
means_dict


# In[35]:


print('Holidays that have higher sales than the mean sales in the non-holiday season: ')
for key,value in means_dict.items():
    if(value > mean):
        print(key)


# In[79]:


plt.figure(figsize=(5,6))
plt.bar(means_dict.keys(), means_dict.values(), color="pink")
plt.axhline(mean, color="maroon", linestyle='-')
plt.show()


# ### d) Provide a monthly and semester view of sales in units and give insights.

# In[55]:


# Splitting Date and create new columns (Day, Month, and Year)
data["Day"]= pd.DatetimeIndex(data['Date']).day
data['Month'] = pd.DatetimeIndex(data['Date']).month
data['Year'] = pd.DatetimeIndex(data['Date']).year
data


# In[90]:


# Monthly view of sales for each years
c = 'plum'

fig, ax = plt.subplots(3,2 ,figsize = (12,18))


ax[0,0].bar(data[data.Year==2010]["Month"],data[data.Year==2010]["Weekly_Sales"],color=c)
ax[0,0].set_xlabel("months")
ax[0,0].set_ylabel("Weekly Sales")
ax[0,0].set_title("Monthly  sales in 2010")


ax[1,0].bar(data[data.Year==2011]["Month"],data[data.Year==2011]["Weekly_Sales"],color='teal')
ax[1,0].set_xlabel("months")
ax[1,0].set_ylabel("Weekly Sales")
ax[1,0].set_title("Monthly sales in 2011")


ax[2,0].bar(data[data.Year==2012]["Month"],data[data.Year==2012]["Weekly_Sales"],color='lightblue')
ax[2,0].set_xlabel("months")
ax[2,0].set_ylabel("Weekly Sales")
ax[2,0].set_title("Monthly sales in 2012")



ax[0,1].scatter(data[data.Year==2010]["Month"],data[data.Year==2010]["Weekly_Sales"],color=c)
ax[0,1].set_xlabel("months")
ax[0,1].set_ylabel("Weekly Sales")
ax[0,1].set_title("Monthly  sales in 2010")


ax[1,1].scatter(data[data.Year==2011]["Month"],data[data.Year==2011]["Weekly_Sales"],color='teal')
ax[1,1].set_xlabel("months")
ax[1,1].set_ylabel("Weekly Sales")
ax[1,1].set_title("Monthly sales in 2011")


ax[2,1].scatter(data[data.Year==2012]["Month"],data[data.Year==2012]["Weekly_Sales"],color='lightblue')
ax[2,1].set_xlabel("months")
ax[2,1].set_ylabel("Weekly Sales")
ax[2,1].set_title("Monthly sales in 2012")


# In[91]:


# Monthly view of sales for all years
plt.figure(figsize=(10,6))
c = 'teal'
plt.bar(data["Month"],data["Weekly_Sales"],color=c)
plt.xlabel("months")
plt.ylabel("Weekly Sales")
plt.title("Monthly sales")
plt.show()


# In[181]:


# Yearly view of sales
plt.figure(figsize=(9,6))


d = data.groupby("Year")[["Weekly_Sales"]].sum().reset_index()
sns.barplot(d['Weekly_Sales'], d['Year'], orient='h', color='lightblue')
plt.title("Yearly sales");


# ## Quarter 1 & 2 as Semester 1
# 
# ## the rest, Quarter 3 & 4 as Semester 2

# In[105]:


data["quarter"] =data["Date"].dt.quarter
data["semester"] = np.where(data["quarter"].isin([1,2]),1,2)
data


# In[184]:


c = 'plum'

fig, ax = plt.subplots(1,3, figsize=(20,6))

ax[0].bar(data[data.Year==2010]["semester"],data[data.Year==2010]["Weekly_Sales"],color='teal')
ax[0].set_xlabel("semesters")
ax[0].set_ylabel("Weekly Sales")
ax[0].set_title("semester  sales in 2010")

ax[1].bar(data[data.Year==2011]["semester"],data[data.Year==2011]["Weekly_Sales"],color='lightblue')
ax[1].set_xlabel("semesters")
ax[1].set_ylabel("Weekly Sales")
ax[1].set_title("semester  sales in 2011")

ax[2].bar(data[data.Year==2012]["semester"],data[data.Year==2012]["Weekly_Sales"],color=c)
ax[2].set_xlabel("semesters")
ax[2].set_ylabel("Weekly Sales")
ax[2].set_title("semester  sales in 2012")


# ### e) Plot the relations between weekly sales vs. other numeric features and give insights.

# In[186]:


data6=data.groupby("Store").agg({"Weekly_Sales":lambda sale :sale.mean(),
                                    "Temperature":lambda temp : temp.mean(),
                                    "Fuel_Price":lambda price :price.mean(),
                                    "CPI":lambda cpi :cpi.mean(),
                                    "Unemployment":lambda unemp :unemp.mean()})
data6.reset_index()
data6


# In[107]:


x=data.groupby('Month')[['Weekly_Sales']].sum()
x


# In[108]:


y=data.groupby('Month')[['Temperature']].mean()
y


# In[109]:


x=data.groupby('Month')[['Weekly_Sales']].sum()
y=data.groupby('Month')[['Temperature']].mean()
plt.scatter(x,y,color='deeppink')


# In[115]:


data_vis=data.head(45)
plt.figure(figsize=(18,6))
plt.bar(data_vis["Temperature"],data_vis["Weekly_Sales"],color=c)


# In[110]:


x=data.groupby('Month')[['Weekly_Sales']].sum()
y=data.groupby('Month')[['Fuel_Price']].mean()
plt.scatter(x,y,color='maroon')


# In[111]:


x=data.groupby('Month')[['Weekly_Sales']].sum()
y=data.groupby('Month')[['CPI']].mean()
plt.scatter(x,y,color='midnightblue')


# In[112]:


x=data.groupby('Month')[['Weekly_Sales']].sum()
y=data.groupby('Month')[['CPI']].sum()
plt.scatter(x,y,color='darkslategray')


# In[214]:


x=data.groupby('Month')[['Weekly_Sales']].sum()
y=data.groupby('Month')[['Unemployment']].sum()
plt.scatter(y['Unemployment'],x['Weekly_Sales'],color='darkgoldenrod')
plt.show()


# In[213]:


plt.figure(figsize=(16,6))
x=data.groupby('Month')[['Weekly_Sales']].sum()
y=data.groupby('Month')[['Unemployment']].mean()
g = sns.barplot(y['Unemployment'], x['Weekly_Sales'], palette="Blues")
g.set_xticklabels(g.get_xticklabels(), rotation = 30)


# In[116]:


data["quarter"] =data["Date"].dt.quarter
data["semester"] = np.where(data["quarter"].isin([1,2]),1,2)
data


# In[187]:


c = 'plum'

plt.bar(data[data.Year==2010]["semester"],data[data.Year==2010]["Weekly_Sales"],color=c)
plt.xlabel("semesters")
plt.ylabel("Weekly Sales")
plt.title("semester  sales in 2010")
plt.show()


# In[ ]:




