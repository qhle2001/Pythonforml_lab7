import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import cv2 as cv
import seaborn as sns
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
import sklearn.metrics as metrics
from sklearn.model_selection import cross_validate

#Label
st.title('Lab 07')
st.title('LinearRegression and LogisticRegression')
st.markdown("""
## 1.Upload Dataset
""")
#Upload file
uploaded_file = st.file_uploader("Chọn 1 file: ")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file, sep=(";"))
    if len(dataframe.columns) == 1:
        dataframe = pd.read_csv(uploaded_file, sep=(";"))
    st.write(dataframe)

def linearregression(dataframe, selected_features, output_feature):
    X = dataframe[selected_features]
    y = dataframe[output_feature]
    #Splitting the dataset to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regr = LinearRegression().fit(X_train, y_train)

    y_train_pred = regr.predict(X_train)
    y_pred = regr.predict(X_test)

    st.write("Train_MAE:", mean_absolute_error(y_true=y_train,y_pred=y_train_pred))
    st.write("Train_MSE:", mean_squared_error(y_true=y_train,y_pred=y_train_pred)) #default=True
    st.write("Train_RMSE:", mean_squared_error(y_true=y_train,y_pred=y_train_pred,squared=False))

    st.write("Test_MAE:", mean_absolute_error(y_true=y_test,y_pred=y_pred))
    st.write("Test_MSE:", mean_squared_error(y_true=y_test,y_pred=y_pred)) #default=True
    st.write("Test_RMSE:", mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False))
def logisticregression(dataframe, selected_features, output_feature):
    X = dataframe[selected_features]
    y = dataframe[output_feature]
    #Splitting the dataset to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the scaler and apply it to the data

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    lr = LogisticRegression().fit(X_train, y_train)
    train_pred = lr.predict(X_train)
    y_pred = lr.predict(X_test)
    st.write("Accuracy_train: ", accuracy_score(train_pred, y_train))
    st.write("Accuracy_test: ", accuracy_score(y_pred, y_test))
    st.write("F1_score_train: ", f1_score(train_pred, y_train, average='weighted'))
    st.write("F1_score_test: ", f1_score(train_pred, y_train))
#Input and output
if uploaded_file is not None:
    st.markdown("## 2.Chọn output feature: ")
    output_feature = st.selectbox("Select output", dataframe.columns)
    st.markdown("## 3.Chọn input feature: ")
    input_features = dataframe.columns.drop(output_feature)
    check_boxes = [st.checkbox(feature, key=feature) for feature in input_features]
    selected_features = [feature for feature, checked in zip(input_features, check_boxes) if checked]
    st.markdown("## 4.Lựa chọn thuật toán")
    option_model = st.radio("What is the model do you want?", ["LinearRegression", "LogisticRegression"])
    if option_model == "LinearRegression" and st.button("Run"):
        st.write("LinearRegression")
        X = dataframe[selected_features]
        y = dataframe[output_feature]
        # Splitting the dataset to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        regr = LinearRegression().fit(X_train, y_train)

        y_train_pred = regr.predict(X_train)
        y_pred = regr.predict(X_test)

        st.write("Train_MAE:", mean_absolute_error(y_true=y_train, y_pred=y_train_pred))
        st.write("Train_MSE:", mean_squared_error(y_true=y_train, y_pred=y_train_pred))  # default=True
        st.write("Train_RMSE:", mean_squared_error(y_true=y_train, y_pred=y_train_pred, squared=False))

        st.write("Test_MAE:", mean_absolute_error(y_true=y_test, y_pred=y_pred))
        st.write("Test_MSE:", mean_squared_error(y_true=y_test, y_pred=y_pred))  # default=True
        st.write("Test_RMSE:", mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False))

        st.markdown("## 5. Nhập dữ liệu để dự đoán")
        input = st.text_input("")
        if st.button("Predict"):
            input = input.split(" ")
            st.write("Nhâp dữ liệu (các thông số cách cách nhau bởi dấu space)", input)
    elif option_model == "LogisticRegression" and st.button("Run"):
        st.write("LogisticRegression")
        X = dataframe[selected_features]
        y = dataframe[output_feature]
        # Splitting the dataset to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create an instance of the scaler and apply it to the data

        scaler = MinMaxScaler()

        X_train = scaler.fit_transform(X_train)

        X_test = scaler.transform(X_test)

        lr = LogisticRegression().fit(X_train, y_train)
        train_pred = lr.predict(X_train)
        y_pred = lr.predict(X_test)
        st.write("Accuracy_train: ", accuracy_score(train_pred, y_train))
        st.write("F1_score_train: ", f1_score(train_pred, y_train, average='weighted'))
        cm = confusion_matrix(train_pred, y_train)
        report = classification_report(train_pred, y_train)
        st.text(report)
        st.write("Accuracy_test: ", accuracy_score(y_pred, y_test))
        st.write("F1_score_test: ", f1_score(train_pred, y_train, average='weighted'))
        cm = confusion_matrix(y_pred, y_test)
        report = classification_report(y_pred, y_test)
        st.text(report)

