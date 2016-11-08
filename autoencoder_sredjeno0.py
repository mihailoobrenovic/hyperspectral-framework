import theano
from theano import tensor as T
import numpy
import time
import matplotlib.pyplot as plt
import scipy 
from scipy import io
from random import shuffle

class AutoEncoder0(object):

    # Konstruktor klase 

    def __init__(self,load_data=True,n_visible=200,n_hidden=100,W=None,bhid=None,bvis=None,dataset='Indian_pines_corrected.mat',labels='Indian_pines_gt.mat',data_name="indian_pines_corrected",label_name="indian_pines_gt"):
        if load_data:            
            self.train_set_x,self.train_set_y,self.val_set_x,self.val_set_y,self.test_set_x,self.test_set_y = self.load_hyperspectral0(dataset,labels,data_name,label_name)
            # Ulazni sloj ima onoliko neurona koliko ima band-ova u slici. Trenig set je organizovan tako da je svaka vrsta jedan piksel
            # pa je broj kolona trening skupa  je jedak broju bandova              
#            self.n_visible = self.train_set_x.get_value().shape[1]
#            if self.n_visible != n_visible:
#                print "Dimenzije se ne slazu!"
#                exit()
            self.n_visible = n_visible
            self.n = self.train_set_x.shape[1].eval()
            self.m = self.train_set_x.shape[0].eval()
        else:
            self.train_set_x=self.train_set_y=self.val_set_x=self.val_set_y=self.test_set_x=self.test_set_y= None 
            self.n_visible = n_visible
            self.n = self.m =  None
            
        
        self.n_hidden = n_hidden
        numpy_rng =  numpy.random.RandomState()
        self.numpy_rng = numpy_rng

        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_hidden + self.n_visible)),
                    high=4 * numpy.sqrt(6. / (self.n_hidden + self.n_visible)),
                    size=(self.n_visible, self.n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)
 
        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    self.n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
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
        self.x = T.dmatrix('x')
        self.costovi = []
        self.validacija=[]
        self.epohe=0;
        self.accracy_train=[]
        self.accracy_val = []
        self.epohe_classifier = 0;
        self.train_confusion=None
        self.valid_confusion = None
        self.params = [self.W, self.b, self.b_prime]
    
    # Reset - Sluzi za vracanje svih vrednosti na inicijalne, pri ponovnom treniranju    
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
        self.x = T.dmatrix('x')
        self.costovi = []
        self.validacija=[]
        self.epohe=0;
        self.params = [self.W, self.b, self.b_prime]
    
    # Racuna sigmoid funkciju od predatih argumenata    
    def sigmoid(self,ulaz,tezine,bias):
        return T.nnet.sigmoid(T.dot(ulaz,tezine)+bias)
       
   # Racuna prvi skriveni sloj autoenkodera     
    def get_hidden_values(self,ulaz):
        return self.sigmoid(ulaz,self.W,self.b)
        
    # Racuna rekonstrukciju ulaza 
    def get_reconstructed_input(self, hidden):
        return self.sigmoid(hidden,self.W_prime,self.b_prime)
       
    # Funkcija za iscrtavanje grafika  
    def plot_costovi(self,x,y,name,fig):
        plt.figure(fig)
        plt.plot(x,y,label = name)
        
    # Funkcija za treniranje AE-a
    def train(self,n_epoha=100, mini_batch_size = 100,learning_rate=0.1,write_each=10):
        #lscalar je 64-bit integer
        print "poocinjem"
        self.epohe = n_epoha
        index = T.lscalar('index')
        x = T.dmatrix('x')
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

        funkcija_treniranja = theano.function([index],[cost,x,z],updates=updates,givens={ x:self.train_set_x[index:index+mini_batch_size,:]} )
        funkcija_provere = theano.function([index],[z],givens={ x:self.train_set_x[index:index+1,:]})
        funkcija_validacije = theano.function([],[cost],givens={ x:self.val_set_x})
        print "Pocinjem"
        self.costovi = []
        self.validacija=[]
        start_time = time.clock()
        plt.figure("reconstruction")
        plt_x_osa = numpy.arange(0,self.n_visible)
        pocetna_vrednost = self.train_set_x[0].eval()
        self.plot_costovi(plt_x_osa,pocetna_vrednost,"Original","reconstruction")
        for epoha in xrange(n_epoha):
            c = []
            print "Epoha ", epoha
            for row in xrange(0,self.m, mini_batch_size):
                cost1,p_x,p_z = funkcija_treniranja(row)
                c.append(cost1)
            if epoha%write_each==0:
                plt_y_osa = funkcija_provere(0)
                plt_y_osa = plt_y_osa[0][0]
                self.plot_costovi(plt_x_osa,numpy.transpose(plt_y_osa),epoha,"reconstruction")
           
            avg_cost_epoha =    numpy.mean(c)
            self.costovi.append(avg_cost_epoha)
            self.validacija.append(funkcija_validacije())
            print "Cost u epohi ",epoha, " je",avg_cost_epoha
            print "{:10.5f}".format(avg_cost_epoha)
            
        end_time = time.clock()   
        print "Gotovo treniranje - ukupno " , end_time - start_time 
        fig = plt.figure("reconstruction")           
        plt.legend()
        plt.title("Reconstruction over time")
        fig.savefig("reconstruction.png")
  
    # Funkcija za iscravanje cost-a    
    def plot_cost_over_time(self):
        epohe_x = []
        fig = plt.figure("cost")
        for e in xrange(self.epohe):
            epohe_x.append(e)
            
        self.plot_costovi(epohe_x,self.costovi,"train","cost")
        self.plot_costovi(epohe_x,self.validacija,"validation","cost")
        plt.legend()
        plt.title("Train and validation cost over time")
        fig.savefig("cost.png")
        
    # Ucitavanje podataka iz Matlab fajlova    
    def load_hyperspectral0(self,dataset,labels,data_name='indian_pines_corrected',label_name='indian_pines_gt'):

        slika = scipy.io.loadmat(dataset)
        slika =  slika[data_name]
        labele = scipy.io.loadmat(labels)
        labele =  labele[label_name]
        dim1 = slika.shape[0]
        dim2 = slika.shape[1]
        bands = slika.shape[2]
        total = sum(sum(labele!=0))
   
        max_class = numpy.max(labele)
        print "Klasa ima", max_class
        
        mat_lab = numpy.zeros([total,max_class])                
        matrica=numpy.ndarray([total,bands])
        
        k = 0
        for i in range(0,dim1):
            for j in range(0,dim2):
                if (labele[i,j]!=0):
                    matrica[k] = slika[i,j,:]
                    mat_lab[k,labele[i,j]-1]=1
                    k+=1
    
        
        euc_dist = numpy.linalg.norm(matrica,axis=1) 
        # Na ovaj nacin se svaka vrsta matrice deli sa jednom elementom iz vektora
        matrica = matrica / euc_dist[:,None]

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

        # Deljenje podataka u odnosu 60%-20%-20%
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
    
    # Vraca matricu tezina izmedju prvog sloja i skrivenog    
    def get_w(self):
        return self.W
        
    # Vraca bias izmedju prvog sloja i skrivenog
    def get_b(self):
        return self.b
    
    def set_train_set(self,data_x,data_y):
        self.train_set_x = data_x
        self.train_set_y = data_y
        
#        n_visible = self.train_set_x.get_value().shape[1]
#        if self.n_visible != n_visible:
#            print "Dimenzije se ne slazu!"
#            exit()        
        self.n = self.train_set_x.shape[1].eval()
        self.m = self.train_set_x.shape[0].eval()
        
    def set_valid_set(self,data_x,data_y):
        self.val_set_x = data_x
        self.val_set_y = data_y
        
    def set_test_set(self,data_x,data_y):
        self.test_set_x = data_x
        self.test_set_y = data_y        
    # Vraca rezultat autoencodera-a za dati ulaz. 
    # Data mora da bude tensor matrica!
    def get_compressed_data(self,data):
        x = T.dmatrix('x')
        x = data
        y = self.get_hidden_values(x)
        #z = self.get_reconstructed_input(y)
        return y
        
    
# Funkcija za brzo testiranje AE-a, sa podrazumevanim vrednostima
def start():
    obj = AutoEncoder0()
    obj.train()
    obj.plot_cost_over_time()

    
    