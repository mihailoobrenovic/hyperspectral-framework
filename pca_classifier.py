from autoencoder_sredjeno import AutoEncoder
from classifier import Classifier
from sklearn.decomposition import PCA


obj = AutoEncoder()
X  = obj.train_set_x.get_value()
Y_train = obj.train_set_y.get_value()
Y_val = obj.val_set_y.get_value()
Y_test = obj.test_set_y.get_value()


pca = PCA(n_components=100)
train = pca.fit_transform(obj.train_set_x.get_value())
val = pca.fit_transform(obj.val_set_x.get_value())
test = pca.fit_transform(obj.test_set_x.get_value())

klas  = Classifier(17,100)
klas.setTrain(train,Y_train)
klas.setVal(val,Y_val)
klas.setTest(test,Y_test)

klas.train()

