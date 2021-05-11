import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("mid_term_1.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('Dataset.csv')
dataset=dataset.drop(['Surname'], axis = 1)


# Extracting dependent and independent variables:
# Extracting independent variable:
X = dataset.iloc[:, :-1].values
# Extracting dependent variable:
y = dataset.iloc[:, -1].values


# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value='Male', verbose=1, copy=True)
#Fitting imputer object to the independent variables x.
imputer = imputer.fit(X[:, [2]])
#Replacing missing data with the calculated mean value
X[:, [2]]= imputer.transform(X[:, [2]])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.
imputer = imputer.fit(X[:, :])
#Replacing missing data with the calculated mean value
X[:, :]= imputer.transform(X[:,:])

# Splitting the Dataset into the Training set and Test set

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

def predict_note_authentication(cs,Gender,Geography,age,tenure,balance,Creditcard,ActiveMem,EstimatedSalary):
  if(Creditcard=='Yes'):
      cc=1
  else:
      cc=0
  if(ActiveMem=='Yes'):
      am=1
  else:
      am=0
  if(Gender=='Male'):
      ge=1
  else:
      ge=0
  if(Geography=='France'):
      go=1
  else:
      go=0


  output= model.predict(sc.transform([[cs,ge,go,age,tenure,balance,cc,am,EstimatedSalary]]))
  print("Person will ",output)
  if output==[0]:
    prediction="Person will Leave Bank"


  if output==[1]:
    prediction="Peson will not Leave Bank "


  print(prediction)
  return prediction
def main():

    html_temp = """
   <div class="" style="background-color:gray;" >
   <div class="clearfix">
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">MID TERM - I</p></center>
   <center><p style="font-size:30px;color:white;margin-top:10px;">NAME: Ronit Khowal</p></center>
   <center><p style="font-size:25px;color:white;margin-top:0px;">|| PIET18CS126 || Sec: C || Roll No 15 ||</p></center>
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Person Leaving Prediction")
    cs = st.number_input('Insert Credit Score')
    Gender = st.selectbox('Insert Gender', ('Male', 'Female'))
    Geography= st.selectbox('Insert Geography', ('France', 'Spain'))
    age= st.number_input('Insert a Age',0,150)
    tenure = st.number_input('Insert a tenure',0,9)
    balance = st.number_input('Insert a balance')
    Creditcard = st.selectbox('You have credit card',('Yes','No'))
    ActiveMem = st.selectbox('You are active member',('Yes','No'))
    EstimatedSalary = st.number_input('Insert a EstimatedSalary')

    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(cs,Gender,Geography,age,tenure,balance,Creditcard,ActiveMem,EstimatedSalary)
      st.success('Model has predicted that -> {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Ronit Khowal")
      st.subheader("C-Section,PIET")

if __name__=='__main__':
  main()
