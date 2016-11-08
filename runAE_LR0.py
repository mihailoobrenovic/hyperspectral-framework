from autoencoder_sredjeno0 import AutoEncoder0
from classifier0 import Classifier0

obj = AutoEncoder0()

Y_train = obj.train_set_y.get_value()
Y_val = obj.val_set_y.get_value()
Y_test = obj.test_set_y.get_value()
X_train = obj.train_set_x.get_value()
X_val = obj.val_set_x.get_value()
X_test = obj.test_set_x.get_value()

obj.train()

train = obj.get_compressed_data(X_train).eval()
val = obj.get_compressed_data(X_val).eval()
test = obj.get_compressed_data(X_test).eval()

klas = Classifier0(16,100)
klas.setTrain(train,Y_train)
klas.setVal(val,Y_val)
klas.setTest(test,Y_test)

klas.train(2000)
klas.plot_cost_over_time()

print klas.train_confusion
print klas.valid_confusion