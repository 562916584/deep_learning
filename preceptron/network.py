# encoding=utf-8
import numpy as np

# sigmoid 激活函数
def sigmoid_function(x):
    return 1/(np.exp(-x)+1)

# 最后的激活函数根据不同的问题设置不同的函数
def identify_function(x):
    return x

def init_network():
    network = {}
    network["w1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["w2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["w3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    w1, w2, w3 = network["w1"], network["w2"], network["w3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid_function(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid_function(a2)
    a3 = np.dot(z2, w3) + b3
    z3 = identify_function(a3)

    return z3

if __name__=='__main__':
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)