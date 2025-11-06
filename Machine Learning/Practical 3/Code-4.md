# Practical-4 (Gradient Descent Algorithm)

Problem Statement: Implement Gradient Descent Algorithm to find the local minima of a function. For example, find the local minima of the function y=(x+3)² starting from the point x=2.

---
 
## Steps

1. Define the function and its derivative
2. Initialize parameters for Gradient Descent
3. Gradient Descent Loop
4. Print the result
5. Plotting

---

## Code

### 0. Import libraries:

```python3
import numpy as np
import matplotlib.pyplot as plt
```

### 1. Define the function and its derivative:

```python3
def f(x):
    return (x + 3)**2

def grad_f(x):
    return 2 * (x + 3)  # derivative of f(x)
```

### 2. Initialize parameters for Gradient Descent:

```python3
x_current = 2          # starting point
learning_rate = 0.1    # step size
tolerance = 1e-6       # convergence tolerance
max_iterations = 25    # maximum iterations
history = [x_current]  # sotring history
```

### 3. Gradient Descent Loop:

```python3
for i in range(max_iterations):
    gradient = grad_f(x_current)
    x_next = x_current - learning_rate * gradient  # update step
    
    # Check convergence
    if abs(x_next - x_current) < tolerance:
        print(f"Converged after {i+1} iterations.")
        break
    
    x_current = x_next
    history.append(x_current)
    print(f"Iteration {i+1}: x = {x_current:.4f}, f(x) = {f(x_current):.4f}")
```

### 4. Print the result:

```python3
print("Local minima at x =", x_current)
print("Function value at local minima y =", f(x_current))
```

### 5. Plotting:

```python3
plt.plot(history, [f(val) for val in history], marker='o')
plt.xlabel("x values")
plt.ylabel("f(x)")
plt.title("Gradient Descent Convergence")
plt.grid()
plt.show()
```

---

