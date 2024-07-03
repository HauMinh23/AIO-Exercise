# Câu 3: Viết function lựa chọn regression loss function để tính loss:
import math
import random


def calc_mae(n):
    mae = 0
    for _ in range(n):
        y = random.uniform(0.0, 10.0)
        y_hat = random.uniform(0.0, 10.0)
        abs_error = abs(y - y_hat)
        mae = mae + abs_error
    mae = mae/n
    return mae


def calc_mse(n):
    mse = 0
    for _ in range(n):
        y = random.uniform(0.0, 10.0)
        y_hat = random.uniform(0.0, 10.0)
        square_error = (y - y_hat)**2
        mse = mse + square_error
    mse = mse/n
    return mse


def calc_rmse(n):
    rmse = math.sqrt.calc_mse(n)
    return rmse


def loss_name(name):
    if name == 'MAE':
        return calc_mae(n)
    elif name == 'MSE':
        return calc_mse(n)
    elif name == 'RMSE':
        return calc_rmse(n)
    else:
        print(f"{name} "'is not supported')


def sample(n):
    for i in range(n):
        print('sample-' f"{i}")


def predict(n):
    for _ in range(n):
        predict = random.uniform(0.0, 10.0)
        print(predict)


def target(n):
    for _ in range(n):
        target = random.uniform(0.0, 10.0)
        print(target)


n = int(3)
name = 'MAE'
print('* loss name:', name)
print('* sample:')
sample(n)
print('* predict:')
predict(n)
print('* target:')
target(n)
print('* loss:', loss_name(name))
