#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Iris Flower Classification using Machine Learning
#Importing some important basic libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
# matplotlib and seaborn are used for visualizations and warnings; we can ignore all the warnings we encounter.


# In[2]:


# Enter the path to the dataset file in the read_csv method. It will import the iris dataset.
##Import iris dataset
df=pd.read_csv('Iris.csv')


# In[3]:


# To view  the data frame.
df


# In[4]:


# View the info of the data frame that contains details like the count of non-null variables and the column’s datatype along with the column names. 
#It will also show the memory usage.
df.info()


# In[5]:


# If there are any missing values, then modify them before using the dataset. For modifying you can use the fillna() method. It will fill null values.
# checking for null values
df.isnull().sum()
# We can see that all values are 0. It means that there are no null values over the entire data frame.


# In[6]:


# To view the column names in the data frame, use columns
df.columns


# In[7]:


import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
#Import iris dataset
df=pd.read_csv('iris.csv')

print(df.info())

print(df.isnull().sum())

print(df.describe())


# In[8]:


#If we view the data frame, we can see two columns with the same id numbers. To delete the unwanted ID column use the drop method.
df=df.drop(columns="Id")


# In[9]:


# Now view the data frame.
df


# In[30]:


#View the count plot of species feature using seaborn.
df['Species'].value_counts()


# In[29]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('Iris.csv')

# Inspect the dataset
print(df.head())

# Convert categorical target variable to numeric if necessary
# Assuming the target column is named 'species' and contains values like 'Iris-setosa', 'Iris-versicolor', etc.
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

# Now you can safely convert other columns to float if needed
df['sepal_length'] = df['sepal_length'].astype(float)
df['sepal_width'] = df['sepal_width'].astype(float)
df['petal_length'] = df['petal_length'].astype(float)
df['petal_width'] = df['petal_width'].astype(float)

print(df.head())


# In[14]:


x=df.iloc[:,:4]
y=df.iloc[:,4]


# In[15]:


x


# In[16]:


y


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[18]:


x_train.shape


# In[19]:


x_test.shape


# In[20]:


y_train.shape


# In[21]:


y_test.shape


# In[22]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[23]:


model.fit(x_train,y_train)


# In[24]:


y_pred=model.predict(x_test)


# In[25]:


y_pred


# In[26]:


from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)


# In[27]:


accuracy=accuracy_score(y_test,y_pred)*100
print("Accuracy of the model is {:.2f}".format(accuracy))


# In[ ]:




