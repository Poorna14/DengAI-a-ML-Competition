import csv

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error

def _main(open_path,save_path,file_name):

    #file location
    open_file_location = open_path + file_name
    save_file_location = save_path + file_name

    #reading the file
    with open(open_file_location,"r") as _file:
        feature_names_arr=[]
        data_arr=[]
        
        dataset = csv.reader(_file)

        for row in dataset:
            feature_names_arr = row
            break

        for row in dataset:
            data_arr.append(row)
            
    return feature_names_arr,data_arr 

file_location="C:\\Users\\Poorna\\Desktop\\DengAI\\"
save_path="C:\\Users\\Poorna\\Desktop\\DengAI\\"

train_data_arr=_main(file_location,save_path,"dengue_features_train.csv")
test_data_arr=_main(file_location,save_path,"dengue_features_test.csv")
target_data_arr=_main(file_location,save_path,"dengue_labels_train.csv")

#train data
df_x=pd.DataFrame(train_data_arr[1],columns=train_data_arr[0])

#target data
a=np.array(target_data_arr[1])
b=a.ravel().tolist()
df_y=pd.DataFrame(b,columns=target_data_arr[0])

#test data
test=pd.DataFrame(test_data_arr[1],columns=test_data_arr[0])

for column in df_x.columns:
    if df_x[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        df_x[column] = le.fit_transform(df_x[column])

for column in df_y.columns:
    if df_y[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        df_y[column] = le.fit_transform(df_y[column])

for column in test.columns:
    if test[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        test[column] = le.fit_transform(test[column])        

reg=linear_model.LinearRegression()
reg.fit(df_x,df_y)
#print(reg.coef_)
y_pred=reg.predict(test)

print (y_pred)

#print(mean_absolute_error(df_y,y_pred))









    
