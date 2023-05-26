import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

st.title('XGBoost')
uploaded_file = st.file_uploader('Dataset')
if uploaded_file is not None:
	dataframe = pd.read_csv(uploaded_file)
	st.write(dataframe)
	dataframe.loc[dataframe['Class'] == 2, ['class']] = 0
	dataframe.loc[dataframe['Class'] == 4, ['class']] = 1

	st.header('Chọn input feature:')

	features = dataframe.iloc[:, :-2]
	check_boxes = [st.checkbox(feature, key=feature, value=True) for feature in features]
	selected_features = [feature for feature, checked in zip(features, check_boxes) if checked]


	if selected_features is not None:
		X = dataframe[selected_features]
		y = dataframe[dataframe.columns[-1]]
		for column in selected_features:
			if type(dataframe[column][0]) == str:
				ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [column])], remainder='passthrough')
				X = np.array(ct.fit_transform(X))
		st.header('Input:')
		st.write(X)
		st.header('Ouput:')
		st.write(y)

	st.header('Train/Test split:')
	size = st.slider('Chọn train size:', 0.0, 1.0, 0.8)
	st.write('Train size: '+ str(int(size*100))+ '%')
	st.write('Test size: '+ str(int(100-size*100))+ '%')

	if st.button('RUN'):
		bst = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, objective='binary:logistic')
		svm = SVC(gamma='auto', probability=True)
		dt = DecisionTreeClassifier()
		clf = LogisticRegression()

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-size, random_state=0)

		# XGBoost
		bst.fit(X_train, y_train)
		bst_pred = bst.predict(X_test)
		bst_proba = bst.predict_proba(X_test)
		#SVM
		svm.fit(X_train, y_train)
		svm_pred = svm.predict(X_test)
		svm_proba = svm.predict_proba(X_test)
		#Decision Tree
		dt.fit(X_train, y_train)
		dt_pred = dt.predict(X_test)
		dt_proba = dt.predict_proba(X_test)
		#LogisticRegression
		clf.fit(X_train, y_train)
		clf_pred = clf.predict(X_test)
		clf_proba = clf.predict_proba(X_test)

		barWidth = 0.3
		fig, ax = plt.subplots()
		labels = ['XGBoost', 'SVM', 'DecisionTree', 'LogisticRegression']
		bar_positions = [np.arange(len(labels)),
						 [x + barWidth for x in np.arange(len(labels))],
						 [x + 2 * barWidth for x in np.arange(len(labels))]]
		# br1 = np.arange(len(labels))
		# br2 = [x + barWidth for x in br1]
		# br3 = [x + 2 + barWidth for x in br1]
		f1 = [metrics.f1_score(y_test, bst_pred, average='weighted'), metrics.f1_score(y_test, svm_pred, average='weighted'),
			  metrics.f1_score(y_test, dt_pred, average='weighted'), metrics.f1_score(y_test, clf_pred, average='weighted')]
		plt.bar(bar_positions[0], f1, width=barWidth, color='blue', label='F1-score')
		logloss = [metrics.log_loss(y_test, bst_proba), metrics.log_loss(y_test, svm_proba),
				   metrics.log_loss(y_test, dt_proba), metrics.log_loss(y_test, clf_proba)]
		plt.bar(bar_positions[1], logloss, width=barWidth, color='red', label='Log-loss score')
		acc = [metrics.accuracy_score(y_test, bst_pred), metrics.accuracy_score(y_test, svm_pred),
			   metrics.accuracy_score(y_test, dt_pred), metrics.accuracy_score(y_test, clf_pred)]
		plt.bar(bar_positions[2], acc, width=barWidth, color='green', label='Accuracy')
		plt.title('PLOT', fontsize=28)
		plt.xlabel('Model', fontsize=14)
		plt.ylabel('Error', fontsize=14)
		plt.xticks([r + barWidth/2 for r in range(len(labels))], labels)
		ax.legend()
		st.pyplot(fig)
		labels = ['XGBoost', 'SVM', 'DecisionTree', 'LogisticRegression']
		br1 = np.arange(len(labels))
		print(br1)
