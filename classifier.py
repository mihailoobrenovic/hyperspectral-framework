import theano
import numpy
from theano import tensor as T
import pandas as pd
import time


class Classifier(object):
    def __init__(self,n_class,dim_data,codder=None):
        self.n_class = n_class
        self.dim_data = dim_data
    
        self.numpy_rng =  numpy.random.RandomState()
        initial_W = numpy.asarray(
        self.numpy_rng.uniform(
        low=-4 * numpy.sqrt(6. / (self.dim_data + n_class)),
        high=4 * numpy.sqrt(6. / (self.dim_data + n_class)),
        size=(self.dim_data,n_class)
        ),
        dtype=theano.config.floatX
        )
        self.W_class = theano.shared(value=initial_W, name='W1', borrow=True)  
        
        self.b_class = theano.shared(
            value=numpy.zeros(
                n_class,
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
                    )
        if codder :
            self.train_set_x,self.train_set_y = self.autoEncoder.get_train_data()
            self.val_set_x,self.val_set_y = self.autoEncoder.get_val_data()
            self.test_set_x,self.test_set_y = self.autoEncoder.get_test_data()
            
            self.m = self.train_set_x.shape[0].eval()
        
        self.costovi = []
        self.validacija=[]
        self.accracy_train=[]
        self.accracy_val = []
    
    def setTrain(self,data_x,data_y):
        self.train_set_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=True)
        self.train_set_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=True)
        self.m = self.train_set_x.get_value().shape[0]
        
    def setVal(self,data_x,data_y):
        self.val_set_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=True)
        self.val_set_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=True)
        
    def setTest(self,data_x,data_y):
        self.test_set_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=True)
        self.test_set_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=True)
      
    def sigmoid(self,ulaz,tezine,bias):
        return T.nnet.sigmoid(T.dot(ulaz,tezine)+bias)
        
    
    def train(self,n_epoha=100,mini_batch_size=100,learning_rate=0.1):      
        x = T.dmatrix('x')
        lab = T.dmatrix('lab')
        index = T.lscalar('index')
        izlaz = self.sigmoid(x,self.W_class,self.b_class)
        L = - T.sum(lab * T.log(izlaz) + (1 - lab) * T.log(1 - izlaz), axis=1)
        cost = T.mean(L)
        params = [self.W_class,self.b_class]
        gparams = T.grad(cost, params)
        
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(params, gparams)
        ]
        
        funkcija_treniranja = theano.function([index],[cost],updates=updates,givens={ x:self.train_set_x[index:index+mini_batch_size,:], lab:self.train_set_y[index:index+mini_batch_size,:]} )
        funkcija_validacije = theano.function([],[cost],givens={ x:self.val_set_x, lab:self.val_set_y} )
        funkcija_accuracy_train = theano.function([],[izlaz],givens={ x:self.train_set_x} )  
        funkcija_accuracy_val = theano.function([],[izlaz],givens={ x:self.val_set_x} ) 
        print "Pocinjem"
        self.costovi = []
        self.validacija=[]
        self.accracy_train=[]
        self.accracy_val = []
        start_time = time.clock()
        for epoha in xrange(n_epoha):
            c = []
            print "Epoha ", epoha
            for row in xrange(0,self.m, mini_batch_size):
                cost1= funkcija_treniranja(row)
                c.append(cost1)
              
            avg_cost_epoha = numpy.mean(c)
            self.costovi.append(avg_cost_epoha)
            self.validacija.append(funkcija_validacije())
            print "Cost u epohi ",epoha, " je",avg_cost_epoha
            print "{:10.5f}".format(avg_cost_epoha)

            #Accuracy train
            tren_predvidjanje = funkcija_accuracy_train()[0] 
            rezultati_pred = tren_predvidjanje.argmax(axis=1)
            rezultati_stvarni = self.train_set_y.get_value().argmax(axis=1)
            tacno_predvidjeni = sum(rezultati_pred == rezultati_stvarni)
            print "TRAIN: Tacno predvidjeni", tacno_predvidjeni
            print "TRAIN: Ukupno primera ",len(rezultati_pred)
            ac = tacno_predvidjeni*1.0/len(rezultati_pred)             
            print "TRAIN: Accuracy : ", ac 
            self.accracy_train.append(ac)
            #Accuracy validation
            val_predvidjanje = funkcija_accuracy_val()[0] 
            rezultati_pred = val_predvidjanje.argmax(axis=1)
            rezultati_stvarni = self.val_set_y.get_value().argmax(axis=1)
            tacno_predvidjeni = sum(rezultati_pred == rezultati_stvarni)
            print "VALIDATION: Tacno predvidjeni", tacno_predvidjeni
            print "VALIDATION: Ukupno primera ",len(rezultati_pred)
            ac = tacno_predvidjeni*1.0/len(rezultati_pred)  
            print "VALIDATION: Accuracy : ", ac
            self.accracy_val.append(ac)

        end_time = time.clock()   
        print "Gotovo treniranje - ukupno " , end_time - start_time            
        print "TRAIN : Confusion matrix "
        tren_predvidjanje = funkcija_accuracy_train()[0] 
        rezultati_pred = tren_predvidjanje.argmax(axis=1)
        rezultati_stvarni = self.train_set_y.get_value().argmax(axis=1)
        tacno = pd.Series(rezultati_stvarni,name="Acctual")
        pred = pd.Series(rezultati_pred,name="Predicted")
        self.train_confusion = pd.crosstab(tacno,pred,margins = True)
        
        print "VALIDATION : Confusion matrix "
        tren_predvidjanje = funkcija_accuracy_val()[0] 
        rezultati_pred = tren_predvidjanje.argmax(axis=1)
        rezultati_stvarni = self.val_set_y.get_value().argmax(axis=1)
        tacno = pd.Series(rezultati_stvarni,name="Acctual")
        pred = pd.Series(rezultati_pred,name="Predicted")
        self.valid_confusion = pd.crosstab(tacno,pred,margins = True)