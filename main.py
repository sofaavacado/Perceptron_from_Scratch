import numpy as np
import matplotlib.pyplot as plt

# Генерация данных с нормализацией
def generate_data(m, n):
    np.random.seed(42)
    X = np.random.randn(m, n)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    w_true = np.random.randn(n)
    b_true = np.random.randn()
    z = np.dot(X, w_true) + b_true
    y_prob = 1 / (1 + np.exp(-z))
    y = (y_prob > 0.5).astype(int)
    return X, y

# Сигмоида
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500))) 

# Функция потерь
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Обучение
def train_perceptron(X, y, learning_rate=0.1, epochs=100, l2_lambda=0.01):
    m, n = X.shape
    w = np.random.randn(n) * 0.01
    b = 0.0
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        
        # Градиенты с учетом производной сигмоиды и L2-регуляризацией
        error = y_pred - y
        sigmoid_deriv = y_pred * (1 - y_pred)
        dw = (np.dot(X.T, error * sigmoid_deriv) / m) + l2_lambda * w
        db = np.mean(error * sigmoid_deriv)
        
        w -= learning_rate * dw
        b -= learning_rate * db
        
        loss = cross_entropy_loss(y, y_pred) + 0.5 * l2_lambda * np.sum(w**2)
        acc = accuracy(y, y_pred)
        history['loss'].append(loss)
        history['accuracy'].append(acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}%")
    
    return w, b, history

# Оценка точности
def accuracy(y_true, y_pred):
    y_pred_class = (y_pred > 0.5).astype(int)
    return np.mean(y_true == y_pred_class) * 100

# Визуализация
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid()
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    m = 1000000  
    n = 30
    
    X, y = generate_data(m, n)
    split_idx = int(0.8 * m)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    w, b, history = train_perceptron(X_train, y_train, learning_rate=0.1, epochs=100, l2_lambda=0.01)
    plot_training_history(history)
    
    # Оценка на тестовой выборке
    z_test = np.dot(X_test, w) + b
    y_test_pred = sigmoid(z_test)
    test_acc = accuracy(y_test, y_test_pred)
    print(f"\nTest Accuracy: {test_acc:.4f}%")
