from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from mlxtend.preprocessing import DenseTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from dataset import DataSet
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import random

data_set = DataSet()
data, label = data_set.get_train_data_set()

indexs = random.sample(range(len(data)), 50000)
data = data[indexs]
label = label[indexs]
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)

count_vect = CountVectorizer()
chi2_model = SelectKBest(chi2, k=1000)
dense_transformer = DenseTransformer()
clf_LG = LogisticRegression()

pipeline_LG = Pipeline(
    [('count_v', CountVectorizer()), ('selectKB', chi2_model), ('to_dens', DenseTransformer()), ('clf', clf_LG)])

pipeline_LG = pipeline_LG.fit(X_train, y_train)
print("F1 score - LG:", f1_score(y_test, pipeline_LG.predict(X_test), average='micro'))
print("Accuracy Score - LG:", accuracy_score(y_test, pipeline_LG.predict(X_test)))
