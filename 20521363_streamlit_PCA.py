import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

# Label
st.title('Lab 08')
st.title('Streamlit and PCA')
st.markdown("""
## 1.Upload Dataset
""")
# Upload file
wine = load_wine()
dataframe = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names'] + ['target'])
st.write(dataframe)

# Input and output
st.markdown("## 2.Chọn input fearture: ")
features = dataframe.columns[:-1]
check_boxes = [st.checkbox(feature, key=feature) for feature in features]
selected_features = [feature for feature, checked in zip(features, check_boxes) if checked]
if selected_features is not None:
    X = dataframe[selected_features]
    y = dataframe[dataframe.columns[-1]]
    if 'State' in selected_features:
        a = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['State'])], remainder='passthrough')
        X = np.array(a.fit_transform(X))
    st.write(X)
    # st.write(y)

# Train test split:
st.markdown("## 3.Train/test split: ")
test_size = st.slider("Test size:", 0.0, 1.0, 0.0)

# Độ đo
st.markdown("## 4. Chọn độ đo: ")
metric_1 = st.checkbox('Log_loss score')
metric_2 = st.checkbox('f1_score')

# K-Fold Cross-Validation
st.markdown("## 5.K-fold cross-validation: ")
K_fold = st.checkbox('K-fold cross-validation')
number_K = 0
if K_fold:
    number_K = st.number_input('Nhập K: ', step=1, min_value=1)

# PCA option
st.markdown('## 6.PCA option')
PCA_option = st.checkbox('PCA option')
n = 0
if PCA_option:
    n = st.number_input('Nhập n: ', step=1, min_value=1)

# Run button
if st.button('RUN'):
    if number_K:
        barWidth = 0.25
        fig, ax = plt.subplots()
        labels = [str(i) if i != number_K + 1 else 'Mean' for i in range(1, number_K + 2)]
        br1 = np.arange(len(labels))
        br2 = [x + barWidth for x in br1]
        if PCA_option:
            df = make_pipeline(PCA(n_components=n), MinMaxScaler(), SVC(gamma='auto', probability=True))
        else:
            df = make_pipeline(MinMaxScaler(), SVC(gamma='auto', probability=True))

        if metric_1:
            scores = cross_validate(estimator=df, X=X, y=y, scoring='neg_log_loss', cv=number_K)
            results = -scores['test_score']
            results = np.append(results, np.mean(results))
            plt.bar(br2, results, width=barWidth, color='blue', label='Log-loss score')
        if metric_2:
            scores = cross_validate(estimator=df, X=X, y=y, scoring='f1_weighted', cv=number_K)
            results = scores['test_score']
            results = np.append(results, np.mean(results))
            plt.bar(br1, results, width=barWidth, color='red', label='F1-score')
        plt.title('PLOT ERROR', fontsize=28)
        plt.xlabel('Fold', fontsize=14)
        plt.ylabel('Error', fontsize=14)
        ax.set_yscale('log')
        plt.xticks([r + barWidth / 2 for r in range(len(labels))], labels)
        st.pyplot(fig)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        if PCA_option:
            pca = PCA(n_components=n)
            pca = pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

        df = make_pipeline(MinMaxScaler(), SVC(gamma='auto'))
        df.fit(X_train, y_train)
        y_pred = df.predict(X_test)

        if metric_1:
            st.write('log_loss score = ', metrics.log_loss(y_test, y_pred))
        if metric_2:
            st.write('f1_score = ', metrics.f1_score(y_test, y_pred))