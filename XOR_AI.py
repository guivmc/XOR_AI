import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Entradas
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Saídas esperadas
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(42)

input_neurons = 2    # número de entradas
hidden_neurons = 2   # número de neurônios na camada oculta
output_neurons = 1   # número de saídas
learning_rate = 0.5
epochs = 20000

# Inicialização aleatória dos pesos
W_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
b_hidden = np.random.uniform(size=(1, hidden_neurons))

W_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
b_output = np.random.uniform(size=(1, output_neurons))

# Treinamento
for epoch in range(epochs):
    # --- Feedforward ---
    hidden_input = np.dot(X, W_input_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W_hidden_output) + b_output
    final_output = sigmoid(final_input)

    # --- Cálculo do erro ---
    error = y - final_output

    # --- Backpropagation ---
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(W_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # --- Atualização dos pesos ---
    W_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    b_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    W_input_hidden += X.T.dot(d_hidden) * learning_rate
    b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Mostrar erro a cada 1000 épocas
    if (epoch+1) % 1000 == 0:
        print(f"Época {epoch+1}, Erro médio: {np.mean(np.abs(error)):.4f}")

# Resultados finais
print("\nResultados após o treinamento:")
for i in range(len(X)):
    print(f"Entrada: {X[i]} -> Saída prevista: {final_output[i][0]:.4f} (esperado {y[i][0]})")