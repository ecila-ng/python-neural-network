import numpy as np

class ANN:
    
    def __init__(self, X, y, layer_sizes, activation = 'sigmoid', weights = None):
        #Class attributes
        self.X = np.array(X) #store the training features
        self.y = np.array(y) #store the training labels
        self.layer_sizes = layer_sizes
        self.activation = activation
        
        #The number of training samples
        self.N = len(self.y)

        #The number of predictors
        self.M = self.X.shape[1]
        assert self.M == layer_sizes[0]

        #The number of non-input layers
        self.D = len(layer_sizes) - 1
        
        #The number of classes
        self.classes = np.unique(y)
        self.K = len(self.classes)
        assert self.K == layer_sizes[-1]
        
        #Activation function
        if activation == 'relu':
            self.a = lambda x: np.where(x > 0, x, 0)
        else:
            self.a = lambda x: 1 / (1 + np.exp(-x))
            
        #If nothing is passed to weights, create a new list
        if weights == None:
            self.weights = []
            
            for i in range(0, self.D):
                rows = layer_sizes[i] + 1
                cols = layer_sizes[i+1]         
                wts = np.random.uniform(low=-1, high=1, size=(rows, cols))           
                self.weights.append(wts)
        else: #else set it equal to the weight list
                self.weights = weights     
                
        #Indicator matrix
        self.T = np.zeros((self.N, self.K))
        for i in range(0, self.N):
            self.T[i, y[i]] = 1
                    
    def predict_proba(self,X):
        X = np.array(X) #store a feature array
        ones = np.ones(shape = (X.shape[0], 1))
    
        #Set A equal to X
        #Create a list called self.activations. It should initially contain only A.
        A = X
        self.activations = [A]
        
        for i in range(0, self.D):
            A = np.insert(A, [0], ones, axis = 1)
            Z = np.dot(A, self.weights[i])
            
            if (i == self.D - 1):
                A = self.softmax(Z)
            else: 
                A = self.a(Z)
            self.activations.append(A)
        return A
            
    def predict(self, X):
         X = np.array(X)
         pred_class = self.predict_proba(X)
         entries = np.argmax(pred_class, axis = 1)
         return entries
    
    def score(self, X, y):        
        X = np.array(X)
        y = np.array(y)
        
        #The model’s loss
        prob = self.predict_proba(X)
        loss = 0
        
        #The model’s accuracy
        acc = np.mean(self.predict(X) == y)
        
        for i in range(0, len(y)):
            loss -= np.log(prob[i, y[i]])
        
        #The method should return a list or tuple containing two values
        return (loss, acc)
    
    def softmax(self, X):
        return np.exp(X) / np.sum(np.exp(X), axis = 1, keepdims = True)
    
    def gradient(self):
        grad_list =[]
        self.predict(self.X)
        ones = np.ones(shape= ((len(self.X), 1))) 
        
        Aout = self.activations[-1]
        Ain = self.activations[-2]
        Ain = np.hstack((ones, Ain))       
        
        grad_by_obs = Ain[:,:,np.newaxis] * (self.T - Aout)[:,np.newaxis,:] 
        grad = np.sum(grad_by_obs, axis=0) 
        grad_list.append(grad)
        
        for i in range(2, len(self.activations)):
            Aout = self.activations[-i]
            Ain = self.activations[-(i+1)]
            ones = np.ones((Ain.shape[0], 1))
            Ain=  np.hstack((ones, Ain))
            nextW = self.weights[-(i-1)][1:,:]
            next_gbo = grad_by_obs[:,1:,:]
                       
            #For each node in the current layer, calculate the dot product 
            #of the weights and the gradient values for 
            #each edge coming out of that node. 
            C1 = (next_gbo * nextW[np.newaxis, :, :]).sum(axis=2)            
            #If the activation function is set to 'sigmoid', then: 
            if(self.activation == 'sigmoid'):
                C2 = (1 - Aout) * C1
            #If the activation function is set to'relu', then set: 
            if(self.activation == 'relu'):
                C2 = np.where(Aout == 0, 0, 1 / (Aout + 10e-10)) * C1                
            grad_by_obs = Ain[:,:,np.newaxis] * C2[:,np.newaxis,:]
            grad = grad_by_obs.sum(axis=0)            
            #Insert grad in to the front of grad_list
            grad_list.insert(0,grad)
        return grad_list    
    def train(self, epochs, lr=0.001, display=1):
        grads = []
        for i in range(1 , epochs+1):
            grads = self.gradient()
            for ii in range(self.D):
                self.weights[ii] += lr * grads[ii]
            loss2 = self.score(self.X, self.y)
            #If display is greater than 0, and the number of 
            #the current epoch is a multiple of display
            if (display > 0) and (i%display == 0):
                print('epoch', i, ': loss = ', loss2[0])
                print ('acc = ',loss2[1])