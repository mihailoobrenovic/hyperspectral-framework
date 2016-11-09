#from stacked_autoencoder0_data import StackedAutoEncoder
from segmented_stacked_autoencoder import SegmentedStackedAutoEncoder
import matplotlib.pyplot as plt
from sklearn import svm

ssae = SegmentedStackedAutoEncoder(3,[35, 69, 96])
ssae.train(100)

[podaci, labele] = ssae.mergeTrainData()
podaci = podaci.eval()
labele = labele.eval()
labele = labele.argmax(axis=1)

podaci_val, lab_val = sa.get_val_data()
podaci_val = podaci_val.eval()
lab_val = lab_val.eval()
lab_val = lab_val.argmax(axis=1)


klas = svm.SVC(kernel = 'rbf',gamma=100, C=100.)
klas.fit(podaci,labele)


pred = klas.predict(podaci_val)
tacno = pd.Series(lab_val,name="Acctual")
pred = pd.Series(pred,name="Predicted")
pd.crosstab(tacno,pred,margins = True)