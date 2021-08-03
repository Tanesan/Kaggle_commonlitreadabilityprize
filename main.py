import mxnet as mx
import pandas as pd
# import lightgbm
from sklearn.model_selection import train_test_split
from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from statistics import mean, median,variance,stdev

ctxs = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu(0)

model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-cased')
scorer = MLMScorer(model, vocab, tokenizer, ctxs)

train_data = pd.read_csv('train.csv')
X = []
X1 = []
for i in range(len(train_data["excerpt"])):
    a = scorer.score_sentences(train_data["excerpt"][i])
    X1.append(a)
y = train_data["target"].values
x = {}
x["m"] = []
x["median"] = []
x["variance"] = []
x["stdev"] = []
for i in range(len(X1)):
    x["m"].append(mean(X1[i]))
    x["median"].append(median(X1[i]))
    x["variance"].append(variance(X1[i]))
    x["stdev"].append(stdev(X1[i]))
X = pd.DataFrame(data=x)

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0, random_state=1)

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train_norm)
X_test_std = stdsc.transform(X_test_norm)
# lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
lr = lightgbm.LGBMRegressor()
lr.fit(X_train_std, y_train)


test_data = pd.read_csv('/kaggle/input/commonlitreadabilityprize/test.csv')
sample_data = pd.read_csv('/kaggle/input/commonlitreadabilityprize/sample_submission.csv')

test = []
y_pred = lr.predict(test_data["excerpt"])
for i in range(len(test_data["excerpt"])):
    a = scorer.score_sentences(test_data["excerpt"][i])
    test.append(a)
xT = {}
xT["m"] = []
xT["median"] = []
xT["variance"] = []
xT["stdev"] = []
for i in range(len(test)):
    xT["m"].append(mean(test[i]))
    xT["median"].append(median(test[i]))
    xT["variance"].append(variance(test[i]))
    xT["stdev"].append(stdev(test[i]))
Test = pd.DataFrame(data=xT)
Test_test_norm = mms.fit_transform(Test)
Test_test_std = stdsc.transform(Test_test_norm)
sample_data["target"] = lr.predict(Test_test_std)

sample_data.to_csv('/kaggle/working/submission.csv', index=False)