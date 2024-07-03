# Câu 4: Viết 4 functions để ước lượng các hàm số sau.
import math


def factorial(n):
    result = 1
    for i in range(n):
        result *= (i + 1)
    return result


def approx_sin(x, n):
    sin_approx = 0
    for i in range(n+1):
        coef = (-1)**i
        num = x**(2*i + 1)
        denom = factorial(2*i + 1)
        sin_approx += coef * (num/denom)
    return sin_approx


def approx_cos(x, n):
    cos_approx = 0
    for i in range(n+1):
        coef = (-1)**i
        num = x**(2*i)
        denom = factorial(2*i)
        cos_approx += coef * (num/denom)
    return cos_approx


def approx_sinh(x, n):
    sinh_approx = 0
    for i in range(n+1):
        num = x**(2*i)
        denom = factorial(2*i+1)
        sinh_approx += (num/denom)
    return sinh_approx


def approx_cosh(x, n):
    cosh_approx = 0
    for i in range(n+1):
        num = x**(2*i)
        denom = factorial(2*i)
        cosh_approx += (num/denom)
    return cosh_approx


print(approx_sin(math.pi, 10))
print(approx_cos(math.pi, 10))
print(approx_sinh(math.pi, 10))
print(approx_cosh(math.pi, 10))
