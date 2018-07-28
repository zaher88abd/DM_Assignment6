from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from mlxtend.preprocessing import DenseTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from dataset import DataSet
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
import matplotlib.pyplot as plt
from MatrixPlot import plot_confusion_matrix
from sklearn.decomposition import TruncatedSVD
import random

data_set = DataSet()
data, label, class_names = data_set.get_train_data_set()

indexs = random.sample(range(len(data)), 50000)
data = data[indexs]
label = label[indexs]
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)

count_vect = CountVectorizer()
selectKBest = SelectKBest(k=2000)
truncatedSVD = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
combined_features = FeatureUnion([("chi2", chi2()), ("univ_select", selectKBest)])
dense_transformer = DenseTransformer()
clf_NB = GaussianNB()

pipeline_NB = Pipeline([
    ('count_v', CountVectorizer()), ('features', combined_features), ('to_dens', DenseTransformer()), ('clf', clf_NB)])

pipeline_NB = pipeline_NB.fit(X_train, y_train)
y_pred = pipeline_NB.predict(X_test)
print("F1 score - NB:", f1_score(y_test, pipeline_NB.predict(X_test), average='micro'))
print("Accuracy Score - NB:", accuracy_score(y_test, pipeline_NB.predict(X_test)))
cnf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
plt = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix NB')
plt.show()
