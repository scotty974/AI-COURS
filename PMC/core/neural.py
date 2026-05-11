class Models():
    def __init__(self):
        self.layers = []
        self.input = None
        self.target = None
        self.output = None

    def add_layers(self, layers):
        self.layers.append(layers)
    
    def fit(self, inputs, targets):
        self.input = inputs
        self.target = targets
    

    
    def forward(self):
        
        x = self.input
        
        for layer in self.layers:
            x = layer.forward(x)
            
        self.output = x
        
        return self.output
        
        
    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
   
    
    
    
            