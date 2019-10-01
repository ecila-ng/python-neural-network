import numpy as np
class ANN:
    
    
    def __init__(self, X, y, layer_sizes, activation='sigmoid', weights=None):
        
        self.X = np.array(X)                    # Store data
        self.y = np.array(y)
        self.layer_sizes = layer_sizes          # Layer sizes, w/o bias nodes
        self.activation = activation            # Record the activation fn. 

        self.N = len(self.y)                    # N = Number of samples
        self.M = self.X.shape[1]                # M = Number of predictors
        self.K = len(np.unique(y))              # K = Number of classes
        self.D = len(layer_sizes) - 1           # D = Network depth

        assert(layer_sizes[0] == self.M)        # Check input layer size
        assert(layer_sizes[-1] == self.K)       # Check output layer size
        
        # Define activation
        if activation == 'relu':
            self.a = lambda x : np.where(x > 0, x, 0)
        else:
            self.a = lambda x : 1 / (1 + np.exp(-x))

        # If weights are provided, use those instead
        
        if (weights == None):                    # Create random weight matrices
            self.weights = []
            for i in range(0, self.D):
                rows = layer_sizes[i] + 1       # +1 to accomodate bias input
                cols = layer_sizes[i+1]         # Number of nodes in next layer
                    
                # Randomly generate all weights
                wts = np.random.uniform(low=-1, high=1, size=(rows, cols))
                self.weights.append(wts)
        
        else: 
            self.weights = weights              # Use supplied weights. 
        
        # Create Indicator Matrix (One hot encoding)
        self.T = np.zeros([self.N,self.K])
        for i in range(0,self.N):
            self.T[i,y[i]] = 1
                   
            
    def predict_proba(self, X):
        
        X = np.array(X)
        ones = np.ones((X.shape[0],1))          # Create column of ones        
        A = X
        
        self.activations = [A]                  # Create list of activations
                       
        # Loop over all layers
        for i in range(0, self.D):       
            A = np.hstack([ones, A])            # Append column of ones
            Z = np.dot(A, self.weights[i])      # Find weighted sums
            
            if i == self.D - 1:                 # Final layer
                exp = np.exp(Z)                 # Apply sofmax
                A = exp / np.sum(exp, axis=1, keepdims=True)
                
            else:                               # Hidden layer
                A = self.a(Z)                   # Apply activation
                                          
            self.activations.append(A)          # Store activation values
        
        return A                                # Return output of final layer
    
    def predict(self, X):
        X = np.array(X)
        prob = self.predict_proba(X)        
        pred = np.argmax(prob, axis=1)
        return pred    

    def score(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        prob = self.predict_proba(X)
        pred = self.predict(X)
        
        # Calculate the loss
        loss = 0
        for i in range(len(y)):
            loss -= np.log(prob[i, y[i]])
            
        # Calculate the accuracy
        acc = np.mean(pred == y)
        
        return (loss, acc)
    
    def gradient(self):
        
        grad_list=[]
        self.predict(self.X)
        ones = np.ones((len(self.X),1))
        
        Aout = self.activations[-1]
        Ain = self.activations[-2]
        Ain=  np.hstack((ones, Ain ))
        grad_by_obs = Ain[:,:,np.newaxis] * (self.T - Aout)[:,np.newaxis,:]
        grad = np.sum(grad_by_obs, axis=0)
        
        grad_list.append(grad)
        
        #for i in range(self.D-1, 0 , -1) :
        l = len(self.activations)
        for i in range(2, l):
            Aout = self.activations[-i]
            Ain = self.activations[-(i+1)]
            ones =np.ones((Ain.shape[0],1))
            Ain=  np.hstack((ones, Ain ))
            nextW = self.weights[-(i-1)][1:,:]
            next_gbo = grad_by_obs[:,1:,:]
            #nextW = nextW[1:,:]

            C1 =(next_gbo *nextW[np.newaxis,:,:]).sum(axis=2)
            if(self.activation == 'sigmoid'):
                C2 =(1-Aout)*C1
            if(self.activation == 'relu'):
                C2 =np.where(Aout==0,0,1 / (Aout+10e-10))*C1
            grad_by_obs = Ain[:,:,np.newaxis]* C2[:,np.newaxis,:]
            grad = grad_by_obs.sum(axis=0)
            grad_list.insert(0,grad)
        return grad_list
    def train(self, epochs, lr=0.001, display=1):
        gr=[]
        for i in range(1 , epochs+1):
            gr= self.gradient()
            for e in range(self.D):
                self.weights[e] +=lr*gr[e]
                #self.weights= self.weights + np.multiply(lr,gr)
            loss_acc = self.score(self.X,self.y)
            if (display>0) and (i%display ==0 ):
                #the number of the current epoch is a multiple of display,
                print('epoch', i, ': loss = ',loss_acc[0] , 'acc = ',loss_acc[1])