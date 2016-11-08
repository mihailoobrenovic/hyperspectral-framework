from autoencoder_sredjeno0_cost import AutoEncoder0Cost
from classifier0_cost import Classifier0Cost
import matplotlib.pyplot as plt

 # Funkcija za iscrtavanje grafika  
def plot_costovi(x,y,name,fig):
    plt.figure(fig)
    plt.plot(x,y,label = name)

obj = AutoEncoder0Cost()

Y_train = obj.train_set_y.get_value()
Y_val = obj.val_set_y.get_value()
Y_test = obj.test_set_y.get_value()
X_train = obj.train_set_x.get_value()
X_val = obj.val_set_x.get_value()
X_test = obj.test_set_x.get_value()
W_train = obj.train_set_wc.get_value()
W_val = obj.val_set_wc.get_value()
W_test = obj.test_set_wc.get_value()

obj.train()

train = obj.get_compressed_data(X_train).eval()
val = obj.get_compressed_data(X_val).eval()
test = obj.get_compressed_data(X_test).eval()

klas = Classifier0Cost(16,100)
klas.setTrain(train,Y_train,W_train)
klas.setVal(val,Y_val,W_val)
klas.setTest(test,Y_test,W_test)

klas.train(2000)

print klas.train_confusion
print klas.valid_confusion

        
# Funkcija za iscravanje cost-a    
epohe_x = []
fig = plt.figure("cost2")
n_epoha = 2000
for e in xrange(n_epoha):
    epohe_x.append(e)
    
plot_costovi(epohe_x,klas.costovi,"train","cost2")
plot_costovi(epohe_x,klas.validacija,"validation","cost2")
plt.legend()
plt.title("Train and validation cost over time")
fig.savefig("cost2.png")