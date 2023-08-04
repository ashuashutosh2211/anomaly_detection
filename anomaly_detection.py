#!/usr/bin/env python
# coding: utf-8

# ## Importing the libraries

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import math
import seaborn as sns  
from collections import Counter


# In[2]:


data = pd.read_csv("Healthcare Providers.csv")


# In[3]:


data.head()


# # Data Cleaning

# In[4]:


# CHECKING NAN VALUES IN THE DATASET 
count_nan = data.isna().sum() 
count_nan


# In[ ]:





# In[5]:


# CHECKING FOR MISSING DATA / NAN VALUES USING HEATMAP 
sns.heatmap(data.isnull(),yticklabels = False,cbar = False , cmap = 'viridis')


# From the above plot we can see that two features (Street Address 2 of the Provider and Middle Initial of the Provider) has may nan values or missing data. 

# In[6]:


data.drop(['Street Address 2 of the Provider', 'Middle Initial of the Provider'],axis =1 ,inplace = True) 
data.head()


# For the gender column we can not directy drop this so we can replace the nan values by mode of that column. 

# In[7]:


data['Gender of the Provider'].fillna(data['Gender of the Provider'].mode()[0],inplace = True) 


# For remaining columns replacing by mode 

# In[8]:


for i in data.columns: 
    data[i].fillna(data[i].mode()[0], inplace = True)# REPLACING NAN VALUES BY MODE 


# We can drop many columns from the data containing names address etc. because they will not affect anomalies in the data

# In[9]:


# CHECKING FOR MISSING DATA / NAN VALUES USING HEATMAP 
sns.heatmap(data.isnull(),yticklabels = False,cbar = False , cmap = 'viridis')


# In[10]:


# HERE SOME OF THE FEATURES ARE NOT IMPORTANT FOR IDENTIFYING THE ANOMALIES SO WE CAN DROP THESE FEATRES ....
#.... THESE FEATURES ARE CATETGORICAL OR NOMINAL AND HAVE A LARGE NUMBER OF UNIQUE VALUES SO WILL NOT HELP IN CLASSIFICATION AND NAME , ADDRESS ALSO HAVE BEEN DROPPED.
columns_to_drop = ['index','National Provider Identifier','Last Name/Organization Name of the Provider','First Name of the Provider','Street Address 1 of the Provider']
data.drop(columns_to_drop,axis = 1, inplace = True) 


# As in the dataset many of the values contains (',','.' etc. in some columns) like [M.D. and MD ] in credentials of the provider column while both are same (as credential is same M.D. or MD) 

# In[11]:


i = 'Credentials of the Provider' 
# REPLCAING THE SPECIAL CHARACTERS FROM THE DATA FOR THE COLUMN 'CREDENTIALS OF THE PROVIDER'
for j in range(len(data)): 
        if(type(data[i][j]) == str): 
#             data[i][j] = data[i][j].replace(',','')
            new_str = ''
            for k in range(len(data[i][j])):
                if(ord(data[i][j][k])>=48 and ord(data[i][j][k])<=57):
                    new_str += data[i][j][k] 
                if(ord(data[i][j][k])>=65 and ord(data[i][j][k])<=90):
                    new_str += data[i][j][k] 
                if(ord(data[i][j][k])>=97 and ord(data[i][j][k])<=122):
                    new_str += data[i][j][k]  
            data[i][j] = new_str # repacling each value in the column after removing the special characters.


# In[12]:


data


# In[13]:


# function for converting the string to numeric which is numeric but given string and contains commas  
def convert_to_numeric(data,col): 
    for j in range(len(data)): 
        if(type(data[col][j]) == str):
            data[col][j] = data[col][j].replace(',','')   # replacing commas by null string
    data[i] = data[col].astype('float64')


# In[14]:


# The columns need to be converted to numeric data 
cols_to_numeric = data.columns[14:]     
# print(cols_to_numeric)
for i in cols_to_numeric: 
    convert_to_numeric(data,i) 


# In[15]:


data


# # Visualising the data

# In[16]:


continuous_features = list(data.columns[14:]) # getting a list of continuous features 
categorical_features = [] # list for storing the categorical featres 

for i in data.columns: 
    if i not in continuous_features and i!="Zip Code of the Provider": 
        categorical_features.append(i) 
print(continuous_features) 
print(categorical_features) 


# In[17]:


i = 'Credentials of the Provider' 
k = Counter(data[i]) 
plt.title(i)  
plt.ylim([0 ,  2000])
plt.bar(list(k.keys()),list(k.values())) 
plt.xticks([])
plt.show()


# In[18]:


for i in categorical_features: 
    if(i!= 'Credentials of the Provider'): 
        k = Counter(data[i]) 
        plt.title(i)  
        plt.ylim([0 ,  max(list(k.values()))])
        plt.bar(list(k.keys()),list(k.values())) 
        plt.xticks([])
        plt.show()


# In[19]:


for i in continuous_features: 
    plt.title(i)
#     plt.hist(data[i]) 
    sns.distplot(data[i])
    plt.show()


# # Checking the correlation of data for every features  and dropping the features with high correlation 

# In[20]:


from dython.nominal import associations 
correlations = associations(data)
# plotting the heatmap to check the correlations of the features 


# In[21]:


fig, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(correlations['corr'], annot = True,ax=ax)

# Show the plot
plt.show()


# In[22]:


corr = correlations['corr'] 
# correlation matrix 
corr


# In[23]:


for i in range(len(data.columns)):
    for j in range(i+1,len(data.columns)):
        # checking for the correlation between the features if high then we can delete one of them 
        
        if(corr[data.columns[i]][data.columns[j]]>=0.98): 
            print(data.columns[i],"  ->  ",data.columns[j] , ' and correlation  = ',corr[data.columns[i]][data.columns[j]]) 


# From the above output it's clear that we can drop the feature 'Entity Type of the Provider and the others are related to one or more so checking by creating a smaller heatmap for those [First -- (HCPCS Code , HCPCS Description, HCPCS Drug Indicator) and Second -- (Average Medicare Allowed Amount, Average Medicare Payment Amount,Average Medicare Standardized Amount)  and Third -- Number of Services , Number of Distinct Medicare Beneficiary/Per Day Services , Number of Medicare Beneficiaries] 

# In[24]:


data.drop(['Entity Type of the Provider'],axis = 1,inplace = True)


# In[25]:


# For HCPCS Code , HCPCS Description, HCPCS Drug Indicator 
df_temp = data[['HCPCS Code' , 'HCPCS Description', 'HCPCS Drug Indicator']]
associations(df_temp)


# Thus from here wee can see that we can keep any of these and drop the others as they have correlation approximately 0.99 and 1 

# In[26]:


data.drop(['HCPCS Code','HCPCS Description'],axis = 1,inplace = True)


# In[27]:


df_temp = data[['Number of Services' , 'Number of Distinct Medicare Beneficiary/Per Day Services' , 'Number of Medicare Beneficiaries']] 
associations(df_temp)


# Thus from the above plotted heatmp we can see that these are higly correalted so we can drop the two of these.

# In[28]:


data


# In[29]:


data.drop(['Number of Services' , 'Number of Distinct Medicare Beneficiary/Per Day Services'],axis = 1,inplace = True)


# In[30]:


# For Average Medicare Allowed Amount, Average Medicare Payment Amount,Average Medicare Standardized Amount
df_temp = data[['Average Medicare Allowed Amount', 'Average Medicare Payment Amount','Average Medicare Standardized Amount']]
associations(df_temp)


# From here again we can drop any two of these features . 

# In[31]:


data.drop(['Average Medicare Allowed Amount', 'Average Medicare Payment Amount'],axis = 1,inplace = True) 


# In[32]:


data


# In[33]:


correlations = associations(data)
fig, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(correlations['corr'], annot = True,ax=ax)

# Show the plot
plt.show()


# From the above plotted heatmap we can see that "Zip Code of the Provided" has good correlation with "State Code of the Provider" and "City of the Provider" but not good with "Country Code of the Provider" but "State Code of the Provider" and "City of the Provider" have good with Country Code of the Provider from these we can drop "Zip Code of the Provider" and "City of the Provider" 

# In[34]:


data.drop(["Zip Code of the Provider",'City of the Provider'],axis = 1,inplace = True) 
data


# In[35]:


data_copy = data.copy() 


# In[36]:


Encodings = {}


# In[37]:


data_copy


# In[38]:


df = data.copy()


# # Encoding the data and standardising the data 

# In[39]:


categorical_features = list(data.columns[0:8]) 
continuous_features = list(data.columns[8:]) 


# In[40]:


standardizations = {} 
standardizations['mean'] = {}
standardizations['mean']['Number of Medicare Beneficiaries'] = data['Number of Medicare Beneficiaries'].mean()
standardizations['mean']['Average Submitted Charge Amount']   = data['Average Submitted Charge Amount'].mean()
standardizations['mean']['Average Medicare Standardized Amount']   = data['Average Medicare Standardized Amount'].mean()
standardizations['std'] = {}
standardizations['std']['Number of Medicare Beneficiaries'] = data['Number of Medicare Beneficiaries'].std()
standardizations['std']['Average Submitted Charge Amount']   = data['Average Submitted Charge Amount'].std()
standardizations['std']['Average Medicare Standardized Amount']   = data['Average Medicare Standardized Amount'].std()


# In[41]:


standardizations


# In[42]:


for i in categorical_features:
    print(i,"-->",len(data[i].unique())) 


# For the columns "Gender of the Provider" , "Country Code of the Provider " , " Medicare Participation Indicator" , "Place of Service " "HCPCS Drug Indicator" we can provide labels as count are less using label encoder. 

# In[43]:


from sklearn.preprocessing import LabelEncoder


# In[44]:


cols = ["Gender of the Provider" , "Country Code of the Provider" , "Medicare Participation Indicator" , "Place of Service" ,"HCPCS Drug Indicator"] 
le = LabelEncoder() 
for i in cols: 
    print(i,"  --  ",dict(Counter(data[i]))) 
    


# In[45]:


Encodings


# In[46]:


data


# In[47]:


data_copy1 = data.copy()


# For the columns country code of the provider there is one country (US) is most occrring more than 99.99% so for rest we can put in a separate label like others for all other country than US and others can be treated as another group of countries and can be represented by a single encoder.  

# In[48]:


feature = 'Country Code of the Provider' 
for i in range(len(data)):
    if(data[feature][i] == 'US'): 
        data[feature][i] = 0 
    else:
        data[feature][i] = 1   # encoding the countries other than US as 1 
Encodings[feature] = {'US' : 0,'others' :1 }
data[feature].astype('int64')


# In[49]:


print(Encodings)


# In[50]:


data


# In[51]:


data


# Encoding the columns with less number of unique values ( ["Gender of the Provider" , "Medicare Participation Indicator" , "Place of Service" ,"HCPCS Drug Indicator"] ) 

# In[52]:


cols = ["Gender of the Provider", "Medicare Participation Indicator" , "Place of Service" ,"HCPCS Drug Indicator"]   

for i in cols:
    encod = 0 
    dict_encod = {}
    for j in range(len(data)): 
        if(data[i][j] not in dict_encod): 
            dict_encod[data[i][j]] = encod
            encod+=1 
        data[i][j] = dict_encod[data[i][j]]  
    Encodings[i] = dict_encod
    data[i].astype('int64') 


# In[53]:


Encodings


# In[54]:


data


# For the remaining categorical features let's see how the data is distributed for each of the columns.
# 
# [Credentials of the Provider --> 1145,
# State Code of the Provider --> 58,
# Provider Type --> 90]

# In[55]:


def sort_dict_by_value(Dict): 
    keys = list(Dict.keys())
    values = list(Dict.values())
    indexes = np.argsort(values)[::-1]
    new_dict = {} 
    for i in range(len(indexes)): 
        new_dict[keys[indexes[i]]] = values[indexes[i]]  
    return new_dict


# In[56]:


# FOR 'CREDENTIALS OF THE PROVIDER'  
column = data['Credentials of the Provider'] 
column_counter = dict(Counter(column))  
# sorting this dictionary in the decreaseing order of count ( values of dictionary ) so that we can get which ...
# ... value is occuring most of th times and how the values are throughout this column 
column_counter = sort_dict_by_value(column_counter) 
print(column_counter)


# So from above we can see that some of the values in this column has very less count so we can group them as others ( for less than 500) 

# In[57]:


df = data.copy() 
df.head()


# In[58]:


def encode_by_threshold(data,column,threshold): # For less than threshold labelling as 0 
    j = 1   
    labels = {}
    column_counter = Counter(data[column])
    for i in range(len(data)): 
        if(column_counter[data[column][i]]<threshold):  
            labels['others'] = 0  # if less than threshold then encoding as 0 
            data[column][i] = 0 
        else:
            if data[column][i] not in labels: 
                labels[data[column][i]] = j  # if greather than or equal to threshol then encoding by 1 , 2 ,3 
                j+=1 
            data[column][i] = labels[data[column][i]]
    return data,labels


# In[59]:


data,dictionary = encode_by_threshold(data,"Credentials of the Provider",500)  
Encodings["Credentials of the Provider"] = dictionary
Encodings


# In[60]:


data.head()


# In[61]:



# FOR 'STATE CODE OF THE PROVIDER'  
column = data["State Code of the Provider" ] 
column_counter = dict(Counter(column))  
# sorting this dictionary in the decreaseing order of count ( values of dictionary ) so that we can get which ...
# ... value is occuring most of th times and how the values are throughout this column 
column_counter = sort_dict_by_value(column_counter) 
print(column_counter)


# Here consider the threshold as 2500 for labeling 

# In[62]:


data,dictionary = encode_by_threshold(data,"State Code of the Provider",2500) 
Encodings["State Code of the Provider"] = dictionary


# In[63]:



# FOR 'Provider Type'  
column = data["Provider Type" ] 
column_counter = dict(Counter(column))  
# sorting this dictionary in the decreaseing order of count ( values of dictionary ) so that we can get which ...
# ... value is occuring most of th times and how the values are throughout this column 
column_counter = sort_dict_by_value(column_counter) 
print(column_counter)


# Considering the threshold as 1000 and labelling the features 

# In[64]:


data,dictionary = encode_by_threshold(data,"Provider Type",1000)
Encodings["Provider Type"] = dictionary


# In[65]:


data.head() 


# In[66]:


# Standardising the continuous features using standard scaler 
from sklearn.preprocessing import StandardScaler 
cols_to_scale = data.columns[8:]  
for i in cols_to_scale: 
    scaler = StandardScaler() 
    df_temp = np.array(data[i]).reshape(-1,1) 
    df_temp = scaler.fit_transform(df_temp).reshape(len(data),)
    data[i] = df_temp


# In[ ]:





# In[67]:


# ONE HOT ENCODING THE DATA FOR CATEGORICAL FEATURES with more than 2 unique values (if not binary )
features_to_encode_one_hot = ['Credentials of the Provider', 'State Code of the Provider', 'Provider Type']
df_temp = data
one_hot_encoded_data = pd.get_dummies(df_temp,columns = features_to_encode_one_hot) 
one_hot_encoded_data


# Dropping the last features for each one hot encoding i.e. 'Credentials of the Provider_12','State Code of the Provider_14' and 'Provider Type_26'

# In[68]:


one_hot_encoded_data.drop(['Credentials of the Provider_12','State Code of the Provider_14','Provider Type_26'],axis = 1,inplace = True)


# In[69]:


data = one_hot_encoded_data


# In[70]:


data.head()


# In[71]:


for i in data.columns:
    if(data[i].dtype ==object): 
        data[i] = data[i].astype('int64') 
data_copy = data.copy()


# In[72]:


data_copy


# In[73]:


data_copy.columns


# In[74]:


for i in Encodings:
    print(i,Encodings[i])


# # PCA

# In[76]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)   # pca with 95 % variance 
pca.fit(data) # fitting the data 

data_pca = pca.transform(data) # getting the transformed data 


# In[77]:


data_pca.shape


# In[78]:


data_recon = pca.inverse_transform(data_pca) # reconstruction of data by taking inverse  transform 
error = np.mean(np.abs(data - data_recon), axis=1) # calculation of reconstruction error 

threshold = np.percentile(error, 95) # taking threshold = 95th  percentile 
anomalous_indices_pca = np.where(error > threshold)[0] # iff error greater than the threshol d then anomalous


# In[79]:


anomalous_indices_pca # indices with anomaly 


# In[95]:


labels = [] 
non_anomalous = [] # list for storing the non annomalous 
for i in range(len(data)): 
    if(i in anomalous_indices_pca): 
        labels.append(1)  # label is 1 for anomalous 
    else:
        labels.append(0)  # lable is 0 for non anomalous 
        non_anomalous.append(i)
plt.title('Anomaly detection using PCA')
plt.scatter(data.iloc[anomalous_indices_pca].iloc[:,6] , data.iloc[anomalous_indices_pca].iloc[:,5] , c = 'yellow',label = 'Anomalies') 
plt.scatter(data.iloc[non_anomalous].iloc[:,6] , data.iloc[non_anomalous].iloc[:,5] , c = 'purple',label = 'Not Anomalies')  
plt.xlabel('Average Submitted Charge Amount') 
plt.ylabel('Number of Medicare Beneficiaries')
plt.legend()


# # ICA

# In[96]:


from sklearn.decomposition import FastICA


# In[97]:


ica = FastICA(n_components = 33, random_state=0) # ica with n component = 33 
data_ica = ica.fit_transform(data)


# In[98]:


data_ica.shape


# In[99]:


data_recon = ica.inverse_transform(data_ica)
error = np.mean(np.abs(data - data_recon), axis=1) # calculating the error in reconstruction 

# Flag anomalies
threshold = np.percentile(error, 95) # taking threshold = 95th  percentile 
anomalous_indices_ica = np.where(error > threshold)[0]  # iff error greater than the threshol d then anomalous


# In[100]:


print(anomalous_indices_ica)


# In[102]:


labels = [] 
non_anomalous = [] # list for storing non anomalous indices 
for i in range(len(data)): 
    if(i in anomalous_indices_ica): 
        labels.append(1) 
    else:
        labels.append(0)   
        non_anomalous.append(i) # non anomalous indices 

plt.title('Anomaly detection using ICA')
plt.scatter(data.iloc[anomalous_indices_pca].iloc[:,6] , data.iloc[anomalous_indices_pca].iloc[:,5] , c = 'yellow',label = 'Anomalies') 
plt.scatter(data.iloc[non_anomalous].iloc[:,6] , data.iloc[non_anomalous].iloc[:,5] , c = 'purple',label = 'Not Anomalies')  
plt.xlabel('Average Submitted Charge Amount') 
plt.ylabel('Number of Medicare Beneficiaries')
plt.legend()


# # ISOLATION FOREST With reduced data using PCA

# In[108]:


from sklearn.ensemble import IsolationForest


# In[109]:


clf = IsolationForest(n_estimators=100, contamination=0.05) # isolatoin forest training with n estimators = 100 and expected anomalies = 5% 
clf.fit(data_pca) # fitting the model with data transformed by pca 


# In[110]:


y_pred = clf.predict(data_pca) 


# In[111]:


print(y_pred)


# In[112]:


anomalies_index_isol_for_pca = [] 

for i in range(len(y_pred)): 
    if(y_pred[i] ==-1):
        anomalies_index_isol_for_pca.append(i)
#         print(i,end = " ")


# In[113]:


anomalies_index_isol_for_pca = np.array(anomalies_index_isol_for_pca) 


# In[114]:


labels = []  
non_anomalous = []
for i in range(len(data)): 
    if(i in anomalies_index_isol_for_pca): 
        labels.append(1) 
    else:
        labels.append(0)  
        non_anomalous.append(i)  # non anomalous indices 
plt.title('Anomaly detection using Isolation forest with pca')
plt.scatter(data.iloc[anomalous_indices_pca].iloc[:,6] , data.iloc[anomalous_indices_pca].iloc[:,7] , c = 'yellow',label = 'Anomalies') 
plt.scatter(data.iloc[non_anomalous].iloc[:,6] , data.iloc[non_anomalous].iloc[:,7] , c = 'purple',label = 'Not Anomalies')  
plt.xlabel('Average Submitted Charge Amount') 
plt.ylabel('Average Medicare Standardized Amount')
plt.legend()


# # ISOLATION FOREST With reduced data using ICA

# In[116]:


clf = IsolationForest(n_estimators=100, contamination=0.05)
clf.fit(data_ica) # by using data transformed by ica 


# In[117]:


y_pred = clf.predict(data_ica) 


# In[118]:


print(y_pred)


# In[119]:


anomalies_index_isol_for_ica = []
for i in range(len(y_pred)): 
    if(y_pred[i] ==-1):
        anomalies_index_isol_for_ica.append(i)
#         print(i,end = " ")


# In[120]:


anomalies_index_isol_for_ica = np.array(anomalies_index_isol_for_ica)


# In[121]:


labels = []  
non_anomalous = []
for i in range(len(data)): 
    if(i in anomalies_index_isol_for_ica): 
        labels.append(1) 
    else:
        labels.append(0)  
        non_anomalous.append(i)
# plt.scatter(data.iloc[:,6],data.iloc[:,7],c = labels) 
plt.title('Anomaly detection using Isolation forest with ica')
plt.scatter(data.iloc[anomalous_indices_pca].iloc[:,6] , data.iloc[anomalous_indices_pca].iloc[:,7] , c = 'yellow',label = 'Anomalies') 
plt.scatter(data.iloc[non_anomalous].iloc[:,6] , data.iloc[non_anomalous].iloc[:,7] , c = 'purple',label = 'Not Anomalies')  
plt.xlabel('Average Submitted Charge Amount') 
plt.ylabel('Average Medicare Standardized Amount')
plt.legend()


# # ISOLATION FOREST

# In[123]:


clf = IsolationForest(n_estimators=100, contamination=0.05)
clf.fit(data)


# In[124]:


y_pred = clf.predict(data)


# In[125]:


anomalies_indices_isol_for = [] 
for i in range(len(y_pred)): 
    if(y_pred[i]==-1): 
        anomalies_indices_isol_for.append(i) 
anomalies_indices_isol_for = np.array(anomalies_indices_isol_for)


# In[126]:


labels = []  
non_anomalous = []
for i in range(len(data)): 
    if(i in anomalies_indices_isol_for): 
        labels.append(1) 
    else:
        labels.append(0)  
        non_anomalous.append(i)
# plt.scatter(data.iloc[:,6],data.iloc[:,7],c = labels)/
plt.title('Anomaly detection using Isolation forest with original data with reduction')
plt.scatter(data.iloc[anomalous_indices_pca].iloc[:,6] , data.iloc[anomalous_indices_pca].iloc[:,7] , c = 'yellow',label = 'Anomalies') 
plt.scatter(data.iloc[non_anomalous].iloc[:,6] , data.iloc[non_anomalous].iloc[:,7] , c = 'purple',label = 'Not Anomalies')  
plt.xlabel('Average Submitted Charge Amount') 
plt.ylabel('Average Medicare Standardized Amount')
plt.legend()


# # DBSCAN

# In[128]:


from sklearn.cluster import DBSCAN 
from sklearn.neighbors import NearestNeighbors


# In[129]:


neigh = NearestNeighbors(n_neighbors= 20 ) # training nearest neighbors model with n_neighbors = 20 
nbrs = neigh.fit(data)
distances, indices = nbrs.kneighbors(data)
k_distances = np.sort(distances[:, 19], axis=0)

# Plot the k-distance graph
plt.plot(np.arange(len(data)), k_distances)
plt.xlabel('Points sorted by distance to their kth nearest neighbor')
plt.ylabel('k-Distance')
plt.show()


# From the plot looks like the value of k -distance where it changes is nearly 0 so taking value of eplison = 0.5

# <!-- From the plot knee points seems to be around 2(on y_axis) because after this value the distance of nearest neighbors become constant -->

# In[130]:


dbscan = DBSCAN(0.5) 
pred = dbscan.fit_predict(data)    
print(pred)


# In[131]:


anomalies_dbscan = [] 
for i in range(len(pred)): 
    if(pred[i] ==-1): 
        anomalies_dbscan.append(i) # finding the outliers and taking that as anomalies 


# In[132]:


print(len(anomalies_dbscan))


# In[133]:


labels = []  
non_anomalous = []
for i in range(len(pred)):
    if(pred[i]==-1):
        labels.append(1) 
    else:
        labels.append(0) 
        non_anomalous.append(i)


# In[136]:


anomalies_indices_dbscan = []
for i in range(len(pred)): 
    if(pred[i]==-1): 
        anomalies_indices_dbscan.append(i)


# In[137]:


anomalies_indices_dbscan = np.array(anomalies_indices_dbscan)


# In[138]:


plt.title('Anomaly detection using DBSCAN')
plt.scatter(data.iloc[anomalies_indices_dbscan].iloc[:,6] , data.iloc[anomalies_indices_dbscan].iloc[:,7] , c = 'yellow',label = 'Anomalies') 
plt.scatter(data.iloc[non_anomalous].iloc[:,6] , data.iloc[non_anomalous].iloc[:,7] , c = 'purple',label = 'Not Anomalies')  
plt.xlabel('Average Submitted Charge Amount') 
plt.ylabel('Average Medicare Standardized Amount')
plt.legend()


# # SVD

# In[139]:


U, s, V = np.linalg.svd(data, full_matrices=False) # finding the three components of svd 


error = np.sum((data - U @ np.diag(s) @ V)**2, axis=1)

threshold = np.percentile(error, 95)

# Identify anomalies
anomalies_indices_svd = np.where(error > threshold)[0]

print("Anomalies:", anomalies_indices_svd)


# In[140]:


labels = []  
non_anomalous = []
for i in range(len(data)): 
    if(i in anomalies_indices_svd): 
        labels.append(1) 
    else:
        labels.append(0)  
        non_anomalous.append(i)
# plt.scatter(data.iloc[:,6],data.iloc[:,7],c = labels) 
plt.title('Anomaly detection using SVD')
plt.scatter(data.iloc[anomalies_indices_svd].iloc[:,6] , data.iloc[anomalies_indices_svd].iloc[:,7] , c = 'yellow',label = 'Anomalies') 
plt.scatter(data.iloc[non_anomalous].iloc[:,6] , data.iloc[non_anomalous].iloc[:,7] , c = 'purple',label = 'Not Anomalies')  
plt.xlabel('Average Submitted Charge Amount') 
plt.ylabel('Average Medicare Standardized Amount')
plt.legend()


# In[142]:


print(anomalies_indices_svd)


# # Anomalies 

# In[143]:


print(anomalous_indices_ica) 
print(anomalous_indices_pca) 
print(anomalies_index_isol_for_pca)
print(anomalies_index_isol_for_ica) 
print(anomalies_indices_isol_for) 
print(anomalies_indices_dbscan) 
print(anomalies_indices_svd)


# In[144]:


print(len(anomalous_indices_ica))
print(len(anomalous_indices_pca))
print(len(anomalies_index_isol_for_pca))
print(len(anomalies_index_isol_for_ica) )
print(len(anomalies_indices_isol_for) )
print(len(anomalies_indices_dbscan)) 
print(len(anomalies_indices_svd))


# In[145]:


anomalies = list(anomalous_indices_ica) +list(anomalous_indices_pca) +list(anomalies_index_isol_for_pca)+list(anomalies_index_isol_for_ica) +list(anomalies_indices_isol_for) +list(anomalies_indices_dbscan) +list(anomalies_indices_svd)


# In[146]:


counter = Counter(anomalies)


# In[147]:


final_anomalies = [] 
for i in counter: 
    if(counter[i]>=2): 
        final_anomalies.append(i) 
print(len(final_anomalies))


# In[148]:


labels = []  
non_anomalous = []
for i in range(len(data)): 
    if(i in final_anomalies): 
        labels.append(1) 
    else: 
        labels.append(0)  
        non_anomalous.append(i)
# plt.scatter(data.iloc[:,6],data.iloc[:,7],c = labels) 
plt.title('Anomaly detection')
plt.scatter(data.iloc[anomalies_indices_svd].iloc[:,6] , data.iloc[anomalies_indices_svd].iloc[:,7] , c = 'yellow',label = 'Anomalies') 
plt.scatter(data.iloc[non_anomalous].iloc[:,6] , data.iloc[non_anomalous].iloc[:,7] , c = 'purple',label = 'Not Anomalies')  
plt.xlabel('Average Submitted Charge Amount') 
plt.ylabel('Average Medicare Standardized Amount')
plt.legend()


# In[149]:


new_df = data.copy() 
new_df['Fraud'] = np.zeros(len(df))  
new_df['Fraud'] = new_df['Fraud'].astype('int64')


# In[150]:


new_df


# In[151]:


for i in anomalies:
    new_df['Fraud'][i] = 1


# In[ ]:





# # PREDICTION FOR NEW TEST DATA 

# Now we will convert our unsupervised data to supervised and apply supervised classifier algorithms to get better results of data for the new coming data to predict whether it is fraud  or not. 

# In[152]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 


# In[153]:


clf1 = RandomForestClassifier() 
clf2 = GaussianNB() 
clf3 = SVC() 
clf4 = LogisticRegression() 
voting_clf = VotingClassifier(estimators=[('rf', clf1), ('gnb', clf2), ('svc', clf3),('lr',clf4)], voting='hard')


# In[154]:


voting_clf.fit(new_df.iloc[:,:-1],new_df.iloc[:,-1])  


# In[155]:


def predict(X_test,Encodings,standardizations,voting_clf,columns):  # function for prediction 
    # as we will have inputs in the original format and we have transformed that so ...
    # ... we need to change the original data in the form of transformed so we will first perform following....
    # .. before prediction 
    test = pd.DataFrame(columns = columns)  # creating the new test data with all 0's 
    for i in range(len(X_test)): 
        arr = []
        for j in range(60): 
            arr.append(0)  
#         print(len(test))
        test.loc[len(test)] = arr  
    iterator = 0 
    for i in X_test.index: 
        continuous_features = ['Number of Medicare Beneficiaries','Average Submitted Charge Amount','Average Medicare Standardized Amount'] 
        for j in continuous_features:  
#             print(X_test[j][i],standardizations['mean'][j],standardizations['std'][j] )
            test[j][iterator] = (X_test[j][i] - standardizations['mean'][j])/standardizations['std'][j]    
        not_one_hot = ['Gender of the Provider','Country Code of the Provider','Medicare Participation Indicator','Place of Service','HCPCS Drug Indicator'] 
 
        for j in X_test.columns:
            if j not in continuous_features:   # for the values which are not continuous .. 
                # if it is not encoded by one hot encoding then providing the code as in the dictionary Encodings 
                if j in not_one_hot: 
                    val = X_test[j][i]  
                    code = Encodings[j][val] 
                    if val not in Encodings[j]:
                        code = 1
                    
                    test[j][iterator]  = code 
                else:
                    val = X_test[j][i]  
                    if(val in Encodings[j]):
                        code = Encodings[j][val]  # getting the code 
                    else:
                        code = 0 
                    new_col = j + '_' + str(code)   # getting that column 
    #                 print(new_col)
                    if(new_col in test.columns): 
                        test[new_col][iterator] = 1 
        iterator+=1 
    y_pred = voting_clf.predict(test) 
    return y_pred


# This function will take 11 features from the data as input from the user and return the predicted value fraud or not fraud

# In[ ]:





# In[ ]:




