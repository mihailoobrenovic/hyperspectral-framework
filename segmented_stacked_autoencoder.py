from autoencoder_sredjeno0 import AutoEncoder0
from stacked_autoencoder0_data import StackedAutoEncoder
import theano
import numpy


class SegmentedStackedAutoEncoder(object):
    
    # Broj epoha i learning_rate moze da se podesava za sbaki sloj u AE-u. Da li to ima smisla?
    # def __init__(self, n_layers=1,n_visible=200,hidden_neurons=[100]):
    def __init__(self, n_bands=3, length_bands=[35, 69, 96]):
        self.ssaes = []
        self.n_visible = length_bands
        self.n_hidden = [13, 13, 13]
        self.n_features = [6, 7, 7]

        [train_sets_X, val_sets_X, test_sets_X, train_sets_Y, val_sets_Y, test_sets_Y] = self.getSplitData(n_bands, length_bands)
        
        for i in range (n_bands):
            dataset = [train_sets_X[i], val_sets_X[i], test_sets_X[i], train_sets_Y[i], val_sets_Y[i], test_sets_Y[i]]
            self.ssaes.append(StackedAutoEncoder(dataset, 2, self.n_visible[i], [self.n_hidden[i], self.n_features[i]]))
        
        
    def getSplitData(self, n_bands, length_bands):
        dataAutoEncoder = AutoEncoder0(load_data=True,n_visible = 200)
        [train_set_x, train_set_y] = dataAutoEncoder.get_train_set()
        [val_set_x, val_set_y] = dataAutoEncoder.get_val_set()
        [test_set_x, test_set_y] = dataAutoEncoder.get_test_set()
        
        Y_train = train_set_y.get_value()
        Y_val = val_set_y.get_value()
        Y_test = test_set_y.get_value()
        X_train = train_set_x.get_value()
        X_val = val_set_x.get_value()
        X_test = test_set_x.get_value()
        
        train_sets_X = []
        val_sets_X = []
        test_sets_X = []
        train_sets_Y = []
        val_sets_Y = []
        test_sets_Y = []

        prev = 0
        for i in range (n_bands):
            next_one = prev + length_bands[i]
            X_train_part = X_train[:, prev:next_one]
            X_val_part = X_val[:, prev:next_one]
            X_test_part = X_test[:, prev:next_one]
            Y_train_part = Y_train[:, prev:next_one]
            Y_val_part = Y_val[:, prev:next_one]
            Y_test_part = Y_test[:, prev:next_one]

            train_sets_X.append(theano.shared(numpy.asarray(X_train_part,dtype=theano.config.floatX),borrow=True))
            val_sets_X.append(theano.shared(numpy.asarray(X_val_part,dtype=theano.config.floatX),borrow=True))
            test_sets_X.append(theano.shared(numpy.asarray(X_test_part,dtype=theano.config.floatX),borrow=True))
            train_sets_Y.append(theano.shared(numpy.asarray(Y_train_part,dtype=theano.config.floatX),borrow=True))
            val_sets_Y.append(theano.shared(numpy.asarray(Y_val_part,dtype=theano.config.floatX),borrow=True))
            test_sets_Y.append(theano.shared(numpy.asarray(Y_test_part,dtype=theano.config.floatX),borrow=True))

            prev = next_one
        return [train_sets_X, val_sets_X, test_sets_X, train_sets_Y, val_sets_Y, test_sets_Y]
                
    def train(self,n_epoha=10,learning_rate=0.1,mini_batch_size = 100):
        for i in range (len(self.ssaes)):
            self.ssaes[i].train(n_epoha=n_epoha,learning_rate=learning_rate,mini_batch_size = mini_batch_size)
            
    def mergeTrainData(self):
        podaci, labele = self.ssaes[0].get_train_data()
        featureData = podaci.eval()
        featureLabels = labele.eval()
        
        for i in range (1,len(self.ssaes)):
            podaci, labele = self.ssaes[i].get_train_data()
            featureData = numpy.concatenate((featureData, podaci.eval()),axis=1)
            #featureLabels = numpy.concatenate((featureLabels, labele.eval()),axis=1)
            
        return [featureData, featureLabels]

    def mergeValData(self):
        podaci, labele = self.ssaes[0].get_val_data()
        featureData = podaci.eval()
        featureLabels = labele.eval()
        
        for i in range (1,len(self.ssaes)):
            podaci, labele = self.ssaes[i].get_val_data()
            featureData = numpy.concatenate((featureData, podaci.eval()),axis=1)
            #featureLabels = numpy.concatenate((featureLabels, labele),axis=1)
            
        return [featureData, featureLabels]

    def mergeTestData(self):
        podaci, labele = self.ssaes[0].get_test_data()
        featureData = podaci.eval()
        featureLabels = labele.eval()
        
        for i in range (1,len(self.ssaes)):
            podaci, labele = self.ssaes[i].get_test_data()
            featureData = numpy.concatenate((featureData, podaci.eval()),axis=1)
            #featureLabels = numpy.concatenate((featureLabels, labele),axis=1)
            
        return [featureData, featureLabels]
        
        

        
        
        
            
        
        
                
        