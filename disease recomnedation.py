#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np


# In[11]:


df= pd.read_csv("C:/Users/harsh/Downloads/disease recomendation/Training.csv")


# In[17]:


df.tail(10)


# In[15]:


df['prognosis'].value_counts()


# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# In[6]:


len(df['prognosis'].unique()) #41 unique disease present


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[8]:


x = df.drop(['prognosis'],axis=1)
y = df['prognosis']


# In[9]:


x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.2 , random_state=20)


# In[10]:


print(x.shape , x_train.shape , x_test.shape)


# In[11]:


#since y is in string formate usko interger me convert karna padega isiliye label incoder ka use karenge 
le = LabelEncoder()
le.fit(y)


# In[12]:


Y = le.transform(y)


# In[13]:


Y


# In[14]:


x_train , x_test , y_train , y_test = train_test_split(x,Y, test_size=0.2 , random_state=20)



# In[15]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier


# In[16]:


models = {
    'SCV':SVC(kernel="linear"),
    "RandomForese":RandomForestClassifier(n_estimators=100,random_state=20),
    "AdaBoostClassifier":AdaBoostClassifier(n_estimators=100,random_state=20),
    "NaiveBias":MultinomialNB()
}
for model_name ,model in models.items():
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test , pred)
    
    print(f"{model_name}  Accuracuy : {acc}") 
    
#since the dataset was properly trained hence the accuracy was 100%
    


# In[17]:


rc = RandomForestClassifier(n_estimators=100,random_state=20)
rc.fit(x_train,y_train)


# In[18]:


svc = SVC(kernel="linear")
svc.fit(x_train,y_train)


# In[19]:


nb = MultinomialNB()
nb.fit(x_train,y_train)


# In[20]:


knn = KNeighborsClassifier()


# In[21]:


sym = pd.read_csv("C:/Users/harsh/Downloads/disease recomendation/symtoms_df.csv")
precaution = pd.read_csv("C:/Users/harsh/Downloads/disease recomendation/precautions_df.csv")
workout = pd.read_csv("C:/Users/harsh/Downloads/disease recomendation/workout_df.csv")
medication = pd.read_csv("C:/Users/harsh/Downloads/disease recomendation/medications.csv")
desc = pd.read_csv("C:/Users/harsh/Downloads/disease recomendation/description.csv")


# In[22]:


arr = df.columns.drop(['prognosis'])
label_dict = {element: index for index, element in enumerate(arr)}
label_dict


# In[ ]:





# In[23]:


arr1= df['prognosis'].unique()
label2_dict = {}


# In[24]:


mapping_dict = {label: name for label, name in enumerate(le.classes_)}
mapping_dict


# In[25]:


sym_list = label_dict
sym_list


# In[26]:


dis_list = mapping_dict


# In[27]:


def get_dis(symptoms):
    unlisted_sym = [i for i in symptoms if i not in sym_list]
    valid = [i for i in symptoms if i  in sym_list]
    if unlisted_sym:
        print(f"These symptoms are not listed: {', '.join(unlisted_sym)}")
    
    if not valid:
        print("No valid symptoms provided.")
        return None
    
    input_ = np.zeros(len(sym_list))

    for i in valid:
        input_[sym_list[i]] = 1
    disease = dis_list[svc.predict([input_])[0]]
    return disease
    


# In[28]:


prob = input("enter symptoms: ")


# In[29]:


prob


# In[35]:


user_prob = [i.strip() for i in prob.split(',')]
user_prob = [s.strip("[] ''") for s in user_prob]
pred = get_dis(user_prob)
type(user_prob)


# In[36]:


sym = pd.read_csv("C:/Users/harsh/Downloads/disease recomendation/symtoms_df.csv")
precaution = pd.read_csv("C:/Users/harsh/Downloads/disease recomendation/precautions_df.csv")
workout = pd.read_csv("C:/Users/harsh/Downloads/disease recomendation/workout_df.csv")
medication = pd.read_csv("C:/Users/harsh/Downloads/disease recomendation/medications.csv")
desc = pd.read_csv("C:/Users/harsh/Downloads/disease recomendation/description.csv")


# In[37]:


workout


# In[41]:


import ast


# In[55]:


def get_info(disease_name):
    disease_desc = desc.loc[desc['Disease'] == disease_name, 'Description'].values[0]
    
    # Get the precautions
    precautions = precaution.loc[precaution['Disease'] == disease_name, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0]
    
    # Get the workout recommendations
    disease_workout = workout.loc[workout['disease'] == disease_name, 'workout'].values
    
    # Get the medications
    disease_medication = medication.loc[medication['Disease'] == disease_name, 'Medication'].values[0]
    
    # Return the combined information
    
    return disease_desc, precautions, disease_workout,disease_medication
    


# In[56]:


des,pre,work,med= get_info('Acne')


# In[59]:


import ast

print("==========DESCRIPTION============")
print(des)
print("\t")
print("==========PRECAUTION=============")
a = 1
for i in pre:
    print(a," : ", i)
    a+=1
print("\t")
print("=========WORKOUT=========")
k = 1 
for s in wrk:
    print(k," : ",s)
    k+=1
print("\t")
print("========MEDICATION========")
new_med = ast.literal_eval(med)
b=1
for j in new_med:
    print(b," : " , j)
    b+=1


# In[60]:


import pickle
pickle.dump(svc,open('C:/Users/harsh/Desktop/flask/svc.pkl','wb'))


# In[ ]:


ezy_sym = {}
def normalised(user_problem , sym_list):
    for sym in sym_list:
        name = sym.replace("_"," ").strip()
        ezy_sym[name] = sym


# In[46]:


pre


# In[61]:


med


# In[ ]:




