#!/usr/bin/python

import sys, getopt
from autoenkoder_klaster import AutoEncoder

def main(argv):
   arg_epochs = 1000
   arg_batch = 100
   try:
      opts, args = getopt.getopt(argv,"he:b:",["epochs=","batch="])
   except getopt.GetoptError:
      print 'run.py -e <num_epochs> -b <mini_batch_size>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'run.py -e <num_epochs> -b <mini_batch_size>'
         sys.exit()
      elif opt in ("-e", "--epochs"):
         arg_epochs = int(arg)
      elif opt in ("-b", "--batch"):
         arg_batch = int(arg)
   print 'Number of epochs is ', arg_epochs
   print 'Mini batch size is ', arg_batch
   
   obj = AutoEncoder()
   obj.train(n_epoha=arg_epochs, mini_batch_size=arg_batch)
   """
   dataset='/lustre/home/mobrenovic/francuska2016/indian-pines.txt',dim1=145,dim2=145,bands=200
   learning rate, write_each
   n_visible,n_hidden
   """

if __name__ == "__main__":
   main(sys.argv[1:])