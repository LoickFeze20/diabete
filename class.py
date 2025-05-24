import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle 
import time
import base64

@st.cache_data
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

#chargement du model
with open("model_dump.pkl", "rb") as file:
    model = pickle.load(file)

#save dataset    
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="diabete_predictions.csv">Download CSV File</a>'
    return href

st.sidebar.image('photo_.jpg')

def main():
    st.markdown("<h1 style='text-align:center;color: brown;'>Streamlit Diabetis App</h1>",unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color: black;'>Diabetis Study in Cameroon</h2>",unsafe_allow_html=True)
    menu = ["Home","Analysis","Data Visualization","Machine Learning"]
    choise = st.sidebar.selectbox("Menu",menu)
    
    data = load_data("diabetes.csv")
    if choise =='Home':
        left,middle,right = st.columns((2,3,2))
        with middle:
            st.image('photo.jpg')
        st.write("This is an app that will analyse diabetes Datas with some python tools that can optimize decisions")
        st.subheader('Diabetis Informations')
        st.write('In Cameroon, the prevalence of diabetes in adults in urban areas is currently estimated at 6 – 8%, with as much as 80% of people living with diabetes who are currently undiagnosed in the population. Further, according to data from Cameroon in 2002, only about a quarter of people with known diabetes actually had adequate control of their blood glucose levels. The burden of diabetes in Cameroon is not only high but is also rising rapidly. Data in Cameroonian adults based on three cross-sectional surveys over a 10-year period (1994–2004) showed an almost 10-fold increase in diabetes prevalence.')

    elif choise == 'Analysis':
        st.subheader('Data Analysis')
        st.write(data.head())
        
        if st.checkbox('summary'):
            st.write(data.describe())
        elif st.checkbox('correlation'):
            fig = plt.figure(figsize=(15,15))
            st.write(sns.heatmap(data.corr(),annot=True))
            st.pyplot(fig)
            
    elif choise == 'Data Visualization':
        st.subheader('Data Visualization')
        st.write(data.head())
        if st.checkbox('countplot'):
            fig = plt.figure(figsize=(5,5))
            st.write(sns.countplot(x=data['Outcome']))
            st.pyplot(fig)
        elif st.checkbox('scatterplot'):
            fig = plt.figure(figsize=(5,5))
            st.write(sns.scatterplot(x=data['Age'],y=data['Glucose'],hue=data['Outcome']))
            st.pyplot(fig)

    elif choise == 'Machine Learning':
        st.subheader('Machine Learning')
        st.write(data.head())
        tab1,tab2,tab3 = st.tabs([":clipboard: Data",":bar_chart: Visualisation",":mask: :smile: Prediction"])
        uploaded_file = st.sidebar.file_uploader('uploa your input csv',type=['csv'])
        if uploaded_file:
            df = load_data(uploaded_file)
            with tab1:
                st.subheader('Loaded dataset')
                st.write(df)
            with tab2:
                st.subheader('Histogramme Glucose')
                fig = plt.figure(figsize=(8,8))
                st.write(sns.histplot(df['Glucose']))
                st.pyplot(fig)
            with tab3:
                model = pickle.load(open('model_dump.pkl','rb'))
                prediction = model.predict(df)
                st.subheader('prediction')
                #transforme du aré de stream en dataset et l'introduire dans le dataset fourni
                pp = pd.DataFrame(prediction,columns=['Prediction'])
                dfn = pd.concat([df,pp],axis=1)
                dfn.Prediction.replace(0,'non diabete',inplace=True)
                dfn.Prediction.replace(1,'diabete',inplace=True)
                st.write(dfn)
                button = st.button('Download')
                if button:
                    st.markdown(filedownload(dfn),unsafe_allow_html=True)

if __name__ == '__main__':
    main()
