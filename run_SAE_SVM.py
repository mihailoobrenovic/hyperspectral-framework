from stacked_autoencoder0 import StackedAutoEncoder
from classifier0 import Classifier0
import matplotlib.pyplot as plt
from sklearn import svm

sa = StackedAutoEncoder(2,200,[40,20])
sa.train(100)



podaci, labele = sa.get_train_data()
podaci = podaci.eval()
labele = labele.eval()
labele = labele.argmax(axis=1)

podaci_val, lab_val = sa.get_val_data()
podaci_val = podaci_val.eval()
lab_val = lab_val.eval()
lab_val = lab_val.argmax(axis=1)


klas = svm.SVC(kernel = 'rbf',gamma=100, C=100.)
klas.fit(podaci,labele)

klas.score(podaci, labele)
klas.score(podaci_val, lab_val)


pred = klas.predict(podaci_val)
tacno = pd.Series(lab_val,name="Acctual")
pred = pd.Series(pred,name="Predicted")
pd.crosstab(tacno,pred,margins = True)