# encoding = utf-8
import numpy as np

# 与感知机
def AND(x1, x2):
    A = np.array([x1, x2])
    # 这个是w1 w2 权重，用来衡量输入信号的重要程度
    B = np.array([0.5, 0.5])
    # 偏置 用来衡量神经元被激活的难易程度
    b=-0.1
    res = A*B+b
    if res <= 0 :
        return 0
    else:
        return 1

# 或感知机
def OR(x1, x2):
    A = np.array([x1, x2])
    B = np.array([0.5, 0.5])
    b = -0.4
    res = A*B+b
    if res <= 0 :
        return 0
    else:
        return 1

# 与非感知机
def NAND(x1, x2):
    A = np.array([x1, x2])
    B = np.array([-0.5, -0.5])
    b = 1.0
    res = A*B+b
    if res <=0:
        return 0
    else:
        return 1

# 单层的感知机无法分离出非线性空间，例如异或门
# 采用多层感知机表示
def XOR(x1, x2):
    A = NAND(x1, x2)
    B = OR(x1, x2)
    res = AND(A, B)
    return res

if __name__=="__main__":
    pass