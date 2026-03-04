"""
==============================================================
TUTORIAL PRÁTICO: IMPLEMENTANDO UM PERCEPTRON DO ZERO
Disciplina: Inteligência Artificial
Professor: Alexandre "Montanha" de Oliveira
Data: 26/02/2026
==============================================================

Arquivo único executável contendo:

1) Implementação do Perceptron
2) Experimento AND
3) Experimento OR
4) Experimento XOR (falha esperada)
5) Impacto da taxa de aprendizado
6) Conceitos finais

Requisitos:
pip install numpy matplotlib
==============================================================
"""

# ==============================================================
# IMPORTAÇÕES
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

np.random.seed(42)

print("✅ Bibliotecas carregadas com sucesso!\n")


# ==============================================================
# IMPLEMENTAÇÃO DO PERCEPTRON
# ==============================================================

class Perceptron:
    """
    Implementação do Perceptron de Rosenblatt (1958)
    """

    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors_history = []

    def step_function(self, z):
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.step_function(linear_output)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        self.errors_history = []

        print("🎯 Iniciando treinamento...")
        print(f"Pesos iniciais: {self.weights}")
        print(f"Bias inicial: {self.bias}")
        print("-" * 50)

        for epoch in range(self.n_iterations):
            errors = 0

            for idx, x_i in enumerate(X):
                prediction = self.predict(x_i.reshape(1, -1))[0]
                error = y[idx] - prediction

                if error != 0:
                    update = self.learning_rate * error
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1

            self.errors_history.append(errors)

            if errors == 0:
                print(f"✅ Convergência na época {epoch + 1}")
                break

        print(f"Pesos finais: {self.weights}")
        print(f"Bias final: {self.bias}")
        print("-" * 50)
        return self


# ==============================================================
# FUNÇÃO PARA PLOTAR FRONTEIRA DE DECISÃO
# ==============================================================

def plot_decision_boundary(X, y, perceptron, title):
    plt.figure()

    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', label="Classe 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='*', label="Classe 1")

    x1_boundary = np.linspace(-0.5, 1.5, 100)

    if perceptron.weights[1] != 0:
        x2_boundary = -(perceptron.weights[0] * x1_boundary + perceptron.bias) / perceptron.weights[1]
        plt.plot(x1_boundary, x2_boundary)

    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==============================================================
# EXPERIMENTO 1 — AND
# ==============================================================

print("\n================ EXPERIMENTO AND =================")

X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_and = np.array([0, 0, 0, 1])

perceptron_and = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron_and.fit(X_and, y_and)

predictions_and = perceptron_and.predict(X_and)
accuracy_and = np.mean(predictions_and == y_and) * 100

print("Acurácia AND:", accuracy_and, "%")
plot_decision_boundary(X_and, y_and, perceptron_and, "Perceptron - AND")


# ==============================================================
# EXPERIMENTO 2 — OR
# ==============================================================

print("\n================ EXPERIMENTO OR =================")

X_or = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_or = np.array([0, 1, 1, 1])

perceptron_or = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron_or.fit(X_or, y_or)

predictions_or = perceptron_or.predict(X_or)
accuracy_or = np.mean(predictions_or == y_or) * 100

print("Acurácia OR:", accuracy_or, "%")
plot_decision_boundary(X_or, y_or, perceptron_or, "Perceptron - OR")


# ==============================================================
# EXPERIMENTO 3 — XOR (FALHA ESPERADA)
# ==============================================================

print("\n================ EXPERIMENTO XOR =================")

X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_xor = np.array([0, 1, 1, 0])

perceptron_xor = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron_xor.fit(X_xor, y_xor)

predictions_xor = perceptron_xor.predict(X_xor)
accuracy_xor = np.mean(predictions_xor == y_xor) * 100

print("Acurácia XOR:", accuracy_xor, "%")
plot_decision_boundary(X_xor, y_xor, perceptron_xor, "Perceptron - XOR (Falha Esperada)")


# ==============================================================
# EXPERIMENTO 4 — TAXA DE APRENDIZADO
# ==============================================================

print("\n================ TAXA DE APRENDIZADO =================")

learning_rates = [0.01, 0.1, 1.0]

for lr in learning_rates:
    print(f"\nTestando taxa: {lr}")
    p = Perceptron(learning_rate=lr, n_iterations=100)
    p.fit(X_and, y_and)

    plt.plot(range(len(p.errors_history)), p.errors_history, label=f"η={lr}")

plt.xlabel("Época")
plt.ylabel("Erros")
plt.title("Impacto da Taxa de Aprendizado")
plt.legend()
plt.grid(True)
plt.show()


# ==============================================================
# CONCLUSÃO
# ==============================================================

print("\n====================================================")
print("🎓 CONCLUSÃO")
print("====================================================")
print("""
Você implementou:

✅ Perceptron do zero
✅ AND e OR (problemas linearmente separáveis)
❌ XOR (não-linearmente separável)
✅ Impacto da taxa de aprendizadoq

O perceptron é o bloco fundamental das redes neurais modernas.
""")