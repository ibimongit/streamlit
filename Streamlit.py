#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np


# In[3]:


df_crop = pd.read_csv("Crop Yield.csv")


# In[4]:


df_crop = df_crop[~df_crop.Statistic.str.contains("Area under Crops")]
df_crop = df_crop[~df_crop.Statistic.str.contains("Crop Yield per Hectare")]
df_crop = df_crop[~df_crop.Statistic.str.contains("Unit")]


# In[5]:


df_crop.rename(columns = {'VALUE':'Yield_000_Tonnes'}, inplace = True)


# In[6]:


df_crop = df_crop.dropna()


# In[7]:


df_temp = pd.read_csv("Temperature Data.csv")


# In[8]:


df_temp = df_temp.dropna()


# In[9]:


df_temp['Month'] = df_temp['Month'].str.replace('M','/')
df_temp['Month'] = pd.to_datetime(df_temp['Month'])
df_temp['Month'] = df_temp['Month'].dt.strftime('%Y')


# In[10]:


df_grp_temp = df_temp.groupby(['Month'])['VALUE'].mean().reset_index()


# In[11]:


df_grp_temp.rename(columns = {'Month':'Year','VALUE':'mean_temp'}, inplace = True)


# In[12]:


df_grp_temp['Year'] = df_grp_temp['Year'].astype(np.int64)


# In[13]:


df_crop_temp=df_crop.merge(df_grp_temp, on='Year', how='left')


# In[14]:


df_rain = pd.read_csv("Rain Data.csv")


# In[15]:


df_rain = df_rain[~df_rain.Statistic.str.contains("Most Rainfall in a Day")]
df_rain = df_rain[~df_rain.Statistic.str.contains("Raindays (0.2mm or More)")]


# In[16]:


df_rain['Month'] = df_rain['Month'].str.replace('M','/')
df_rain['Month'] = pd.to_datetime(df_rain['Month'])
df_rain['Month'] = df_rain['Month'].dt.strftime('%Y')


# In[17]:


df_grp_rain = df_rain.groupby(['Month'])['VALUE'].sum().reset_index()


# In[18]:


df_grp_rain.rename(columns = {'Month':'Year','VALUE':'total_rain'}, inplace = True)


# In[19]:


df_grp_rain['Year'] = df_grp_rain['Year'].astype(np.int64)


# In[20]:


df_crop_temp_rain=df_crop_temp.merge(df_grp_rain, on='Year', how='left')


# In[21]:


df_ML1 = df_crop_temp_rain


# In[22]:


df_ML1.drop(['Statistic', 'Year','UNIT'], axis=1, inplace=True)


# In[23]:


df_dum = pd.get_dummies(df_ML1 ['Type of Crop'])
X = df_ML1.drop('Type of Crop', axis = 1) # setting X as independent features appart from Zone which will be our dependant features
y = df_ML1['Type of Crop'] # setting y as dependent feature Zone
#df_dum = pd.get_dummies(X['Meteorological Weather Station'])
X = pd.concat([X,df_dum], axis = 1)
#X = X.drop(['Meteorological Weather Station'], axis = 1)
X = X.select_dtypes(exclude=['object'])
X['Type of Crop'] = y


# In[ ]:


Xdata = X.copy()
Xdata .columns = Xdata .columns.str.replace(' ','_')
Xdata .Type_of_Crop = pd.Categorical(Xdata .Type_of_Crop)
Xdata ['Crop_Cat_Id'] = Xdata.Type_of_Crop.cat.codes
Xdata  = Xdata.drop(['Type_of_Crop'], axis = 1)
Xdata 


# In[30]:


from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

data = Xdata
    
X = data[['Yield_000_Tonnes', 'mean_temp', 'total_rain', 'Beans_and_peas',
       'Fodder_beet', 'Kale_and_field_cabbage', 'Oilseed_rape', 'Potatoes',
       'Spring_barley', 'Spring_oats', 'Spring_wheat', 'Sugar_beet',
       'Total_barley', 'Total_oats', 'Total_wheat',
       'Total_wheat,_oats_and_barley', 'Turnips', 'Winter_barley',
       'Winter_oats', 'Winter_wheat']]
y = data['Crop_Cat_Id']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

model = LogisticRegression()
model.fit(X_train, y_train)

explainer = ClassifierExplainer(model, X_test, y_test)
ExplainerDashboard(explainer).run()


# In[ ]:




