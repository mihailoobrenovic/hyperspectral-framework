from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
import numpy as np
import matplotlib.cm as cm
import autoencoder_sredjeno 
import stacked_autoencoder


from sklearn.decomposition import PCA

def vis_data(data,classes):

    X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(data)
    plt.figure()
    colors = cm.rainbow(np.linspace(0, 1, 17))
    for i in range(17):
        ind = np.where(classes==i)
        plt.scatter(X_embedded[ind,0],X_embedded[ind,1],color = colors[i],marker ='x',label = i)
    plt.legend()

# Raw data
obj = AutoEncoder()
X = obj.test_set_x.get_value()
Y = obj.test_set_y.get_value()
Y = Y.argmax(axis = 1)
vis_data(X,Y)

# Autoencoder
obj.train(n_epoha=50)
X = obj.get_compressed_data(obj.test_set_x).eval()

# PCA
pca = PCA(n_components=100)
X = pca.fit_transform(obj.test_set_x.get_value())
vis_data(X,Y)


#Stacked AutoEncoder

SA = StackedAutoEncoder(n_layers=2,hidden_neurons=[100,50])
X,Y1 = sa.get_test_data()
X = X.eval()
vis_data(X,Y)
