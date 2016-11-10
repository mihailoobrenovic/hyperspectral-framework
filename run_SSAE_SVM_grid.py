#from stacked_autoencoder0_data import StackedAutoEncoder
from segmented_stacked_autoencoder import SegmentedStackedAutoEncoder
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd

ssae = SegmentedStackedAutoEncoder(3, [35, 69, 96], [13, 13, 13], [3, 3, 4])
ssae.train(2000)

[train_set_x, train_set_y] = ssae.mergeTrainData()
[val_set_x, val_set_y] = ssae.mergeValData()
[test_set_x, test_set_y] = ssae.mergeTestData()

train_set_y = train_set_y.argmax(axis=1)
val_set_y = val_set_y.argmax(axis=1)
test_set_y = test_set_y.argmax(axis=1)

mat=[]
train=[]
test=[]
i=0
for g in range(-3,4,1):
    for c in range(0,7,1):
        klas = svm.SVC(kernel = 'rbf',gamma=(10**g), C=(10**c))
        klas.fit(train_set_x,train_set_y)
        mat.append((10**g,10**c))
        train.append(klas.score(train_set_x,train_set_y))
        test.append( klas.score(val_set_x,val_set_y))

print train
print test
        
