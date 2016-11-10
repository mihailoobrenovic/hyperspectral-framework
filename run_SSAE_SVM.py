#from stacked_autoencoder0_data import StackedAutoEncoder
from segmented_stacked_autoencoder import SegmentedStackedAutoEncoder
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd

ssae = SegmentedStackedAutoEncoder(3, [35, 69, 96], [13, 13, 13], [3, 3, 4])
ssae.train(100)

[train_set_x, train_set_y] = ssae.mergeTrainData()
[val_set_x, val_set_y] = ssae.mergeValData()
[test_set_x, test_set_y] = ssae.mergeTestData()

train_set_y = train_set_y.argmax(axis=1)
val_set_y = val_set_y.argmax(axis=1)
test_set_y = test_set_y.argmax(axis=1)

klas = svm.SVC(kernel = 'rbf',gamma=100, C=100.)
klas.fit(train_set_x,train_set_y)

klas.score(train_set_x, train_set_y)
klas.score(val_set_x, val_set_y)
klas.score(test_set_x, test_set_y)

pred = klas.predict(test_set_x)
tacno = pd.Series(test_set_y,name="Acctual")
pred = pd.Series(pred,name="Predicted")
pd.crosstab(tacno,pred,margins = True)