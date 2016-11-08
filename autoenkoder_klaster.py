import theano
from theano import tensor as T
import cPickle
import gzip
import os
import numpy
import time
#import Image
#from utils import tile_raster_images
import matplotlib.pyplot as plt
import pandas
import scipy 
from scipy import io
from random import shuffle

class AutoEncoder(object):
    
    def __init__(self,n_visible=200,n_hidden=100,W=None,bhid=None,bvis=None):
        self.train_set_x,self.train_set_y,self.val_set_x,self.val_set_y,self.test_set_x,self.test_set_y = self.load_indian_pines()
        #self.train_set_x  = podaci
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        numpy_rng =  numpy.random.RandomState()
        self.numpy_rng = numpy_rng
        self.n = self.train_set_x.shape[1].eval()
        self.m = self.train_set_x.shape[0].eval()
        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)
        ### Zasto je bvis jednako broju neurna u ulaznom sloju? 
        ### Zar ne bi trebalo da bude samo jedan broj?
        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.x = T.fmatrix('x')
        self.costovi = []
        self.validacija=[]
        self.epohe=0;
        self.accracy_train=[]
        self.accracy_val = []
        self.epohe_classifier = 0;
        self.params = [self.W, self.b, self.b_prime]
        



    def reset(self):
        numpy_rng = self.numpy_rng
        initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_hidden + self.n_visible)),
                    high=4 * numpy.sqrt(6. / (self.n_hidden + self.n_visible)),
                    size=(self.n_visible, self.n_hidden)
                ),
                dtype=theano.config.floatX
            )
        W = theano.shared(value=initial_W, name='W', borrow=True)
 
        bvis = theano.shared(
                value=numpy.zeros(
                    self.n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        bhid = theano.shared(
            value=numpy.zeros(
                self.n_hidden,
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
            )
        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.x = T.fmatrix('x')
        self.costovi = []
        self.validacija=[]
        self.epohe=0;
        self.params = [self.W, self.b, self.b_prime]
    def sigmoid(self,ulaz,tezine,bias):
        rez = T.nnet.sigmoid(T.dot(ulaz,tezine)+bias)
        print "Rezultat je ", rez
        return rez
        
    def get_hidden_values(self,ulaz):
        return self.sigmoid(ulaz,self.W,self.b)
        #T.nnet.sigmoid(T.dot(ulaz, self.W) + self.b)
    
    def get_reconstructed_input(self, hidden):
        return self.sigmoid(hidden,self.W_prime,self.b_prime)
        #T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
      
    def get_cost(self,learning_rate):
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)
        # gradient (izvod) po parametrima
        gparams = T.grad(cost, self.params)
        
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)
    def train(self,n_epoha=1000, mini_batch_size = 100,learning_rate=0.1,write_each=100):
        #lscalar je 64-bit integer
        #iscalar je 32-bitni integer
        print "poocinjem"
        self.epohe = n_epoha
        index = T.iscalar('index')
        x = T.fmatrix('x')
        y = self.get_hidden_values(x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1)
        cost = T.mean(L)
        print "Stigao sam do cost-a"
        # gradient (izvod) po parametrima
        gparams = T.grad(cost, self.params)
        
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        print "Stigao sam do funkcije"
        #cost,updates = self.get_cost(learning_rate)
        funkcija_treniranja = theano.function([index],[cost,x,z],updates=updates,givens={ x:self.train_set_x[index:index+mini_batch_size,:]} )
        funkcija_provere = theano.function([index],[z],givens={ x:self.train_set_x[index:index+1,:]})
        funkcija_validacije = theano.function([],[cost],givens={ x:self.val_set_x})
        print "Pocinjem"
        self.costovi = []
        self.validacija=[]
        start_time = time.clock()
        #plt_x_osa = numpy.arange(0,self.n_visible)
        #pocetna_vrednost = self.train_set_x[0].eval()
        #plt.plot(plt_x_osa,pocetna_vrednost, linewidth=3,label="Original")
        for epoha in xrange(n_epoha):
            c = []
            v = []
            print "Epoha ", epoha
            for row in xrange(0,self.m, mini_batch_size):
                #print "Row = ", row
                cost1,p_x,p_z = funkcija_treniranja(row)
                #print p_x
                #print p_z
                c.append(cost1)
            #epoch_column = numpy.ones(self.n_visible)*epoha
            #weights = numpy.c_[epoch_column, self.W]
            
            self.log_params(epoha)
            
            if epoha%write_each==0:
                plt_y_osa = funkcija_provere(0)
                plt_y_osa = plt_y_osa[0][0]
#                    print size(plt_x_osa)
#                    print size(plt_y_osa)
#                    print plt_y_osa
                
                #plt.plot(plt_x_osa,numpy.transpose(plt_y_osa),label = epoha)
                    
            avg_cost_epoha =    numpy.mean(c)
            self.costovi.append(avg_cost_epoha)
            
            self.validacija.append(funkcija_validacije()[0])
            print "Cost u epohi ",epoha, " je",avg_cost_epoha
            print "{:10.5f}".format(avg_cost_epoha)
            
        
        self.log_cost()    
        end_time = time.clock()   
        print "Gotovo treniranje - ukupno " , end_time - start_time            
        #plt.legend()
        
    def train_classifier(self, n_epoha=100,n_klasa=17, learning_rate=0.1,mini_batch_size=100):
        initial_W = numpy.asarray(
        self.numpy_rng.uniform(
        low=-4 * numpy.sqrt(6. / (self.n_hidden + n_klasa)),
        high=4 * numpy.sqrt(6. / (self.n_hidden + n_klasa)),
        size=(self.n_hidden,n_klasa)
        ),
        dtype=theano.config.floatX
        )
        W_class = theano.shared(value=initial_W, name='W1', borrow=True)  
        
        b_class = theano.shared(
            value=numpy.zeros(
                n_klasa,
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
            )        
        
        self.epohe_classifier=n_epoha
            
        x = T.dmatrix('x')
        lab = T.dmatrix('lab')
        index = T.lscalar('index')
        y = self.get_hidden_values(x)
        izlaz = self.sigmoid(y,W_class,b_class)
        L = - T.sum(lab * T.log(izlaz) + (1 - lab) * T.log(1 - izlaz), axis=1)
        cost = T.mean(L)
        params = [W_class,b_class]
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

            #Accuracy
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
        
        
    def log_params(self, epoha):
        if (epoha==0):
            fileMode = 'w'
        else:
            fileMode = 'a'

        weights_df = pandas.DataFrame(self.W.get_value())
        weights_df.to_csv(path_or_buf='log_weights.csv', sep=",", mode=fileMode, index=False, header=False)
        del weights_df
        
        b_vis_df = pandas.DataFrame(self.b_prime.get_value())
        b_vis_df.to_csv(path_or_buf='log_b_vis.csv', sep=",", mode=fileMode, index=False, header=False)
        del b_vis_df
        
        b_hid_df = pandas.DataFrame(self.b.get_value())
        b_hid_df.to_csv(path_or_buf='log_b_hid.csv', sep=",", mode=fileMode, index=False, header=False)
        del b_hid_df
        
    def log_cost(self):
        cost_series = pandas.Series(self.costovi)
        cost_series.to_csv(path='log_cost.csv', sep=",", mode="w", index=False, header=False)
        validation_series = pandas.Series(self.validacija)
        validation_series.to_csv(path='log_cost.csv', sep=",", mode="a", index=False, header=False)
        
        
        """
    def plot_result_weights(self):
        
        image = Image.fromarray(
        tile_raster_images(X=self.W,
                           img_shape=(28, 28), tile_shape=(10, 10),
                          tile_spacing=(1, 1)))
        image.save('filters_corruption_0.png')
        """
        
    def plot_cost_over_time(self):
        epohe_x = []
        for e in xrange(self.epohe):
            epohe_x.append(e)
        plt.plot(epohe_x,self.costovi,label="Train")
        plt.plot(epohe_x,self.validacija,label="Validation")
        plt.legend()
    def plot_accuracy_over_time(self):
        epohe_x = []
        for e in xrange(self.epohe_classifier):
            epohe_x.append(e)
        plt.plot(epohe_x,self.accracy_train,label="Train")
        plt.plot(epohe_x,self.accracy_val,label="Validation")
        
        plt.legend()
    
    #def load_indian_pines(self,dataset='/lustre/home/mobrenovic/francuska2016/indian-pines.txt',dim1=145,dim2=145,bands=200):
    def load_indian_pines(self,dataset='C:\\Users\\jecak_000\\Desktop\\Hyperspectral\\indian_pines.txt',dim1=145,dim2=145,bands=200):
#        arr=numpy.loadtxt(dataset)
#        hyper=numpy.ndarray((dim1,dim2,bands))
#        total = dim1 * dim2
#        matrica=numpy.ndarray((total,bands))
#        cnt=0
#        for j in range(0,bands):
#            for i in range(0,total):
#                matrica[i,j]=arr[cnt]
#                cnt+=1

        slika = scipy.io.loadmat('Indian_pines_corrected.mat')
        slika =  slika["indian_pines_corrected"]
        labele = scipy.io.loadmat('Indian_pines_gt.mat')
        labele =  labele["indian_pines_gt"]
        
        total = dim1 * dim2
        # TODO : Proveriti da li su elementi matrice ucitani po vrstama tj. da li su slagani po vrstama
        # koliko ima klasa
        max_class = numpy.max(labele)+1
        print "Klasa ima", max_class
        #matrica z je class x ukupno_piksela. Za svaki piksel se kao kolona te matrice cuva labela
        #mat_lab = numpy.zeros([max_class,total])        
        mat_lab = numpy.zeros([total,max_class])                
        matrica=numpy.ndarray([total,bands])
        
        for i in range(0,dim1):
            for j in range(0,dim2):
                matrica[i*dim2+j] = slika[i,j,:]
                mat_lab[dim2*i+j,labele[i,j]]=1
    
        
        euc_dist = numpy.linalg.norm(matrica,axis=1) 
        # Na ovaj nacin se svaka vrsta matrice deli sa jednom elementom iz vektora
        matrica = matrica / euc_dist[:,None]
#        m = matrica.max()
#        matrica = matrica/m

    #TODO : Napraviti da radi genericki a ne samo za jedan dataset

        # Shuffle dataset
        mat = []
        lab = []
        index_shuf = range(total)
        shuffle(index_shuf)
        for i in index_shuf:
            mat.append(matrica[i])
            #lab.append(mat_lab[:,i])
            lab.append(mat_lab[i])
        matrica = mat
        mat_lab = lab

        
        train_set_x = matrica[:int(total*0.6)]
        val_set_x = matrica[int(total*0.6):int(total*0.8)]
        test_set_x  =matrica[int(total*0.8):] 
        train_set_y = mat_lab[:int(total*0.6)]
        val_set_y = mat_lab[int(total*0.6):int(total*0.8)]
        test_set_y  =mat_lab[int(total*0.8):]
       
        return (theano.shared(numpy.asarray(train_set_x,dtype=theano.config.floatX),borrow=True),
                theano.shared(numpy.asarray(train_set_y,dtype=theano.config.floatX),borrow=True),
                theano.shared(numpy.asarray(val_set_x,dtype=theano.config.floatX),borrow=True),
                theano.shared(numpy.asarray(val_set_y,dtype=theano.config.floatX),borrow=True),
                theano.shared(numpy.asarray(test_set_x,dtype=theano.config.floatX),borrow=True),
                theano.shared(numpy.asarray(test_set_y,dtype=theano.config.floatX),borrow=True))
        
        
    
    
    def napravi_shared_dataset(self,data_xy,borrow = True):
        # Svaki od dataset-ova je dat kao uredjen par - slika (matrica) i labela
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    
    
    def load_dataset(self,dataset = '/home/jelica/Desktop/Hyperspectral/mnist.pkl.gz'):
        
        # razdvajam putanju         
         data_dir, data_file = os.path.split(dataset)
         if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
             print "Greska sa ulaznim fajlom"
         
         print('... loading data')
        # otvara se za citanje gzip!         
         f = gzip.open(dataset, 'rb')
         # minst je u Pickle formatu pa se zato koristi cPicle 
         # Format je uredjena trojka, train, valid and test set
         train_set, valid_set, test_set = cPickle.load(f)
         f.close()
         test_set_x, test_set_y = self.napravi_shared_dataset(test_set)
         valid_set_x, valid_set_y = self.napravi_shared_dataset(valid_set)
         train_set_x, train_set_y = self.napravi_shared_dataset(train_set)
        
         rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
         return train_set_x,valid_set_x,test_set_x
                 