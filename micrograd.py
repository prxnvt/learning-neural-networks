"""
Value wrapper class for float types 
Neuron, Layer, and MLP
Creating a testing set 
Gradient Descent Example
"""

import math
import random
import numpy as np
np.random.seed(42)
import matplotlib

# -----------------------------------------------------------------------------
# Value wrapper class for float types 

class Value(): 

    def __init__(self, data, _children = (), _op = '', label = ''): 
        self.data = data 
        self.grad = 0.0 
        self._backward = lambda:None 
        self._prev = set(_children) 
        self._op = _op 
        self.label = label 
    
    def __repr__(self): 
        out = f"Value(data = {self.data})" 
        return out 
    
    def __add__(self, other): 
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other), label = '+') 

        def _backward(): 
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad 
        out._backward = _backward 

        return out 
    
    def __radd__(self, other): 
        return self + other 

    def __mul__(self, other): 
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data * other.data, (self, other), label = '*') 

        def _backward(): 
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 
        out._backward = _backward 

        return out 
    
    def __rmul__(self, other):
        return self * other 

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = self + (-1.0 * other)

        return out 

    def __rsub__(self, other):
        return Value(other) + (-1.0 * self)
    
    def __pow__(self, other):
        assert(isinstance(other, (int, float)))
        out = Value(self.data ** other, (self, ), f'^{other}')

        def _backward():
            self.grad += out.grad * (other * (self.data ** (other - 1.0)))
        out._backward = _backward

        return out 
    
    def __rpow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return pow(other, self)
    
    def __truediv__(self, other):
        return self * (other ** (-1))
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def exp(self):
        out = Value(math.exp(self.data), (self, ), label = 'exp')

        def _backward():
            self.grad += math.exp(self.data) * out.grad 
        out._backward = _backward

        return out 

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

# -----------------------------------------------------------------------------
# Neuron, Layer, and Multi-Layer Perceptron 

class Neuron():
    
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1.0, 1.0)) for i in range(nin)]
        self.b = Value(random.uniform(-1.0, 1.0))

    def __call__(self, x):
        # act = w*x + b
        act = 0.0 

        for wi, xi in zip(self.w, x): 
            act += wi * xi 
        act += self.b 

        out = act.tanh() 
        return out 

    def parameters(self):
        return self.w + [self.b]

class Layer():

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for i in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        # fix: always returned a list, so a 1-neuron output layer gave back [Value] and broke `yt - yp` 
        return outs[0] if len(outs) == 1 else outs 

    def parameters(self):
        out = []
        for neuron in self.neurons:
            for p in neuron.parameters():
                out.append(p)
        return out 

class MLP():

    def __init__(self, nin, nouts):
        sz = [nin] + nouts 
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        out = []
        for layer in self.layers:
            for p in layer.parameters():
                out.append(p)
        return out 

# -----------------------------------------------------------------------------
# Creating a testing set 
    
a = [float(x) for x in np.random.uniform(-5, 5, 100)]
b = [float(x) for x in np.random.uniform(-5, 5, 100)]
y = [float(x) for x in np.random.choice([-1, 1], 100).astype(float)]

# -----------------------------------------------------------------------------
# Gradient Descent Example 

network = MLP(2, [16, 1])
step = 0.05

costs = []

for epoch in range(100):
    
    # forward pass:
    # take predictions
    ypred = []
    for (ai, bi) in list(zip(a, b)):
        ypred.append(network((ai, bi)))
    # take loss using MSE
    loss = 0.0
    for yt, yp in zip(y, ypred):
        loss += (yt - yp) ** 2
    loss = loss * (1.0 / len(y)) # structured this way to preserve Value properties 
    
    # back pass:
    # reset gradients
    for p in network.parameters():
        p.grad = 0.0
    
    loss.backward()

    # update 
    for p in network.parameters():
        p.data -= step * p.grad 

    costs.append(loss.data) 

import matplotlib.pyplot as plt 
plt.plot(costs)