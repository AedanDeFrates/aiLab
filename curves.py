import numpy as np


def curve0(x, y):
    z = x**2 + y**2
    return z
        
    
def curve1(x, y):
    w1 = 0.45
    w2 = 0.5
    x = np.array(x)
    y = np.array(y)
    r = np.sqrt(x**2 + y**2)
    case1 = r - np.pi / (w1 * w2)
    case2 = case1 * np.cos(w1 * w2 * r)
    if x.shape == () and y.shape == ():
        return case1 if case1 >= 0 else case2
    return np.where(case1 >= 0, case1, case2)
            
            
def curve2(x, y):
    w1 = 0.45
    w2 = 0.5
    z = np.sqrt(x**2 + y**2) - 1/(w1*w2)*np.cos(w1*x)*np.cos(w2*y)
    return z


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    x = np.linspace(-20, 20, 200)
    y = np.linspace(-20, 20, 200)
    
    X, Y = np.meshgrid(x, y)
    
    Z0 = curve0(X, Y)
    Z1 = curve1(X, Y)
    Z2 = curve2(X, Y)
    
    fig = plt.figure(figsize=(18, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z0, cmap='viridis')
    ax1.set_title('Curve 0')
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Z1, cmap='viridis')
    ax2.set_title('Curve 1')
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, Z2, cmap='viridis')
    ax3.set_title('Curve 2')
    
    plt.show()