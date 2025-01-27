from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from dataset import DataSet
from ShowChart import showChart
import random


def getChiChart(numComp=2):
    data_set = DataSet()
    data, label = data_set.get_train_data_set()
    indexs = random.sample(range(len(data)), 10000)
    data = data[indexs]
    label = label[indexs]
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)

    chi2_model = SelectKBest(chi2, k=numComp)
    chi2_model = chi2_model.fit(X_train_counts, label)
    X_new = chi2_model.transform(X_train_counts)

    print(chi2_model.scores_.shape)
    print(chi2_model.scores_)
    showChart(x=X_new, y=label, title="Chart Chi for best Component ", range_=X_new.shape[0], limit=3000)
