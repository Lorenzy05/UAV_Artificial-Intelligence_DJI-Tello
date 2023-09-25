import numpy as np
import matplotlib.pyplot as plt





def h(x, a, b, c, d):
    result = a * x**3 + b*x**2 + c*x + d
    return result

def Derivetive(x, a, b, c, d):
    step1 = a * x**3 + b*x**2 + c*x + d
    step2 = a * (x + 1e-8)**3 + b*(x + 1e-8)**2 + c*(x + 1e-8) + d

    result = (step2 - step1) / 1e-8

    return result

x = list(np.arange(-10, 10, 0.1))
y = [h(x, 1, 7, 12, 5) for x in x]

Derivetive = [Derivetive(x, 1, 7, 12, 5) for x in x]


#plt.plot(x, y)
plt.axhline(0, color='black', linewidth=1)  # 绘制水平的x轴线
plt.axvline(0, color='black', linewidth=1)
plt.grid()

plt.xlabel('x-as')
plt.ylabel('y-as')
plt.title('f(x) = x^3 + 7x^2 + 12x + 5')
plt.plot(x, y)
#plt.plot(x, Derivetive)



def newton_method(f, df, initial_guess, tolerance=1e-6, max_iterations=100):
    x = initial_guess
    iteration = 0

    while abs(f(x, a, b, c, d)) > tolerance and iteration < max_iterations:
        x = x - f(x, a, b, c, d) / df(x, a, b, c)
        iteration += 1

        plt.scatter(x, h(x, a, b, c, d), c='red')

    return x

# 测试
def derivative_h(x, a, b, c):
    result = 3 * a * x**2 + 2 * b * x + c
    return result

a, b, c, d = -3, 7, 12, 5
initial_guess = 0  # 初始猜测值
result = newton_method(h, derivative_h, initial_guess)

plt.scatter(result, h(result, a, b, c, d), c='black')

if abs(h(result, a, b, c, d)) < 1e-6:
    print(f"斜率为0的点：x = {result}, y = {h(result, a, b, c, d)}")
else:
    print("未找到斜率为0的点")

plt.scatter(-0.6227971460220768, 2.2069457372708712e-11)
plt.show()