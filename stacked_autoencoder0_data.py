from autoencoder_sredjeno0 import AutoEncoder0


class StackedAutoEncoder(object):
    
    # Broj epoha i learning_rate moze da se podesava za sbaki sloj u AE-u. Da li to ima smisla?
    def __init__(self, dataset, n_layers=1,n_visible=200,hidden_neurons=[100]):
        
        self.n_layers=n_layers
        # self.dataAutoEncoder = AutoEncoder0(load_data=True,n_visible = n_visible,n_hidden = hidden_neurons[0])
        self.autoEncoders = []
        self.dataset = dataset
        for i in range(self.n_layers):
            if i==0 :
                self.autoEncoders.append(AutoEncoder0(load_data=False,n_visible = n_visible,n_hidden = hidden_neurons[0]))
            else:
                self.autoEncoders.append(AutoEncoder0(load_data=False,n_visible = hidden_neurons[i-1],n_hidden = hidden_neurons[i]))
            
        
    def train(self,n_epoha=10,learning_rate=0.1,mini_batch_size = 100):
        [train_set_x, val_set_x, test_set_x, train_set_y, val_set_y, test_set_y] = self.dataset
        # Trenira se prvi AE        
        self.autoEncoders[0].set_train_set(train_set_x,train_set_y)
        self.autoEncoders[0].set_valid_set(val_set_x, val_set_y)
        self.autoEncoders[0].set_test_set(test_set_x,test_set_y)
            
        self.autoEncoders[0].train(n_epoha=n_epoha,learning_rate=learning_rate,mini_batch_size = mini_batch_size)        
        # Ulaz u svaki sledeci AE je izlaz iz prethodnog. Svaki AE se trenira posebno
        for i in range(1,self.n_layers,1):
            prev = self.autoEncoders[i-1]
            self.autoEncoders[i].set_train_set(prev.get_compressed_data(prev.train_set_x),prev.train_set_y)
            self.autoEncoders[i].set_valid_set(prev.get_compressed_data(prev.val_set_x),prev.val_set_y)
            self.autoEncoders[i].set_test_set(prev.get_compressed_data(prev.test_set_x),prev.test_set_y)
            self.autoEncoders[i].train(n_epoha=n_epoha,learning_rate=learning_rate,mini_batch_size = mini_batch_size) 
    
    def get_train_data(self):
        
        data = self.autoEncoders[0].get_compressed_data(self.autoEncoders[0].train_set_x)
        for i in range(1,self.n_layers,1):
            data = self.autoEncoders[i].get_compressed_data(data)
            
        return data,self.autoEncoders[0].train_set_y
        
    def get_val_data(self):
        
        data = self.autoEncoders[0].get_compressed_data(self.autoEncoders[0].val_set_x)
        for i in range(1,self.n_layers,1):
            data = self.autoEncoders[i].get_compressed_data(data)
            
        return data,self.autoEncoders[0].val_set_y

    def get_test_data(self):
        
        data = self.autoEncoders[0].get_compressed_data(self.autoEncoders[0].test_set_x)
        for i in range(1,self.n_layers,1):
            data = self.autoEncoders[i].get_compressed_data(data)
            
        return data,self.autoEncoders[0].test_set_y
        
        
        
        
            
        
        
                
        