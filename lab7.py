import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import ast
import numpy as np
#import cv2 as cv
import seaborn as sns
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error

#Label
st.title('Lab 07')
st.title('LinearRegression and LogisticRegression')
st.markdown("""
## 1.Upload Dataset
""")
#Upload file
uploaded_file = st.file_uploader("Chọn 1 file: ")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    if len(dataframe.columns) == 1:
        list_number = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        column_names = dataframe.columns.tolist()
        column_names_splitted = column_names[0].split(";")
        dataframe[column_names_splitted] = dataframe[column_names[0]].str.split(';', expand=True)
        dataframe.drop(column_names, axis=1, inplace=True)
        for idx in column_names_splitted:
            if dataframe[idx][0][0] in list_number:
                dataframe[idx] = dataframe[idx].astype(float)
    st.write(dataframe)


if uploaded_file is not None:
    string_cols = dataframe.select_dtypes('object').columns.tolist()
    if string_cols != []:
        non_string_cols = dataframe.select_dtypes(exclude=['object']).columns.tolist()
        one_hot_features = pd.get_dummies(dataframe[string_cols])
        dataframe = pd.concat([dataframe[non_string_cols], one_hot_features], axis=1)
        st.write(dataframe)
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
        filename = "model.joblib"
        if os.path.isfile(filename):
            os.remove(filename)
        st.write("LinearRegression")
        X = dataframe[selected_features]
        y = dataframe[output_feature]
        # Splitting the dataset to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        regr = LinearRegression().fit(X_train, y_train)

        # save the model to a file
        joblib.dump(regr, 'model.joblib')

        y_train_pred = regr.predict(X_train)
        y_pred = regr.predict(X_test)

        st.write("Train_MAE:", mean_absolute_error(y_true=y_train, y_pred=y_train_pred))
        st.write("Train_MSE:", mean_squared_error(y_true=y_train, y_pred=y_train_pred))  # default=True
        st.write("Train_RMSE:", mean_squared_error(y_true=y_train, y_pred=y_train_pred, squared=False))

        st.write("Test_MAE:", mean_absolute_error(y_true=y_test, y_pred=y_pred))
        st.write("Test_MSE:", mean_squared_error(y_true=y_test, y_pred=y_pred))  # default=True
        st.write("Test_RMSE:", mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False))
    elif option_model == "LogisticRegression" and st.button("Run"):
        filename = "model.joblib"
        if os.path.isfile(filename):
            os.remove(filename)
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

        # save the model to a file
        joblib.dump(lr, 'model.joblib')

        train_pred = lr.predict(X_train)
        y_pred = lr.predict(X_test)
        st.write("Accuracy_train: ", accuracy_score(train_pred, y_train))
        st.write("F1_score_train: ", f1_score(train_pred, y_train, average='weighted'))
        cm = confusion_matrix(train_pred, y_train)
        report = classification_report(train_pred, y_train)
        st.text(report)
        st.write("Accuracy_test: ", accuracy_score(y_pred, y_test))
        st.write("F1_score_test: ", f1_score(y_pred, y_test, average='weighted'))
        cm = confusion_matrix(y_pred, y_test)
        report = classification_report(y_pred, y_test)
        st.text(report)

        # plot the confusion matrix using Matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', fontsize='large')
        plt.xlabel('Predictions', fontsize='large')
        plt.ylabel('Actuals', fontsize='large')
        plt.title('Confusion Matrix', fontsize='x-large')

        # display the plot in Streamlit
        st.pyplot(fig)
#Cho người dùng nhập dữ liệu để dự đoán
    st.markdown("## 5. Test")
    input = st.text_input("Nhập input features (các thông số được nhập dưới dạng: [19,19000,...] [47,25000,...] hoặc [1.1,...] [1.3,...])")
    predict = st.button("Predict")
    if input and predict:
        load_model = joblib.load('model.joblib')
        text_array = input.strip().split(" ")
        list_obj = [ast.literal_eval(item) for item in text_array]
        if option_model == "LogisticRegression":
            scaler = MinMaxScaler()
            list_obj = scaler.fit_transform(list_obj)
        y_pred = load_model.predict(list_obj)
        st.write("Output: ", y_pred)
