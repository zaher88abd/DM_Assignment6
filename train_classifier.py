from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import ChiChart
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from dataset import DataSet

ChiChart.getChiChart(3)

# data_set = DataSet()
# data, label = data_set.get_train_data_set()
#
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(data)
#
# chi2_model = SelectKBest(chi2, k=2)
# chi2_model = chi2_model.fit(X_train_counts, label)
# X_new = chi2_model.transform(X_train_counts)
