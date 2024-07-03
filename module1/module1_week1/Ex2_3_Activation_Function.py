# câu 2: viết function mô phỏng theo 3 activation function
import math
# hàm sigmoid


def sigmoid(x):
    return 1 / (1 + math.e**(-x))

# hàm ReLU


def relu(x):
    if x > 0:
        return x
    else:
        return 0

# hàm elu


def elu(x):
    a = 0.01
    if x > 0:
        return x
    else:
        return a*(math.e**(x) - 1)


def is_number(x):
    try:
        float(x)
    except ValueError:
        print('x must be a number')
        return False
    return True


def activation_function(x, activation_function_name):
    if activation_function_name == 'sigmoid':
        return sigmoid(x)
    elif activation_function_name == 'relu':
        return relu(x)
    elif activation_function_name == 'elu':
        return elu(x)
    else:
        print(f"{activation_function_name} "'is not supported')


x = -0.5
is_number(x)
activation_function_name = 'sigmoid'
print('Output:', activation_function(x, activation_function_name))
