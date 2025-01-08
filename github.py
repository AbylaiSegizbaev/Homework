import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x_start, learning_rate, num_iterations, gradient_func):
    x = x_start
    trajectory = [x]
    
    for i in range(num_iterations):
        gradient = gradient_func(x)
        x = x - learning_rate * gradient
        trajectory.append(x)
    
    return trajectory, x

if __name__ == "__main__":

    gradient_func = lambda x: 2 * x
    trajectory, minimum = gradient_descent(x_start=10, learning_rate=0.1, num_iterations=20, gradient_func=gradient_func)

    print(f"Траектория: {trajectory}")
    print(f"Найденный минимум: {minimum}")
    
    x_values = np.linspace(-12, 12, 400) 
    y_values = x_values**2 

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Целевая функция f(x) = x^2', color='blue')
    plt.scatter(trajectory, [x**2 for x in trajectory], color='red', label='Траектория градиентного спуска')
    plt.plot(trajectory, [x**2 for x in trajectory], color='red', linestyle='dashed')
    plt.title("Градиентный спуск для функции f(x) = x^2")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

