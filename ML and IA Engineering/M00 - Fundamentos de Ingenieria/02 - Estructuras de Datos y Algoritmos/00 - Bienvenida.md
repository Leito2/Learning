# 🧮 Bienvenida a Estructuras de Datos y Algoritmos

Este curso te da las herramientas algorítmicas para construir sistemas de ML escalables, eficientes y elegantes. No se trata de resolver puzzles de entrevistas, sino de entender las estructuras que usan PyTorch, Redis, bases de datos vectoriales y sistemas de recomendación bajo el capó.

---

## 🗂️ Módulos del curso

1. [[01 - Complejidad Algoritmica|Complejidad Algorítmica]]
2. [[02 - Arrays, Listas y Hash Tables|Arrays, Listas y Hash Tables]]
3. [[03 - Arboles y Heaps|Árboles y Heaps]]
4. [[04 - Grafos|Grafos]]
5. [[05 - Algoritmos de Busqueda y Ordenamiento|Algoritmos de Búsqueda y Ordenamiento]]
6. [[06 - Programacion Dinamica|Programación Dinámica]]
7. [[07 - Caso Practico - Sistema de Recomendacion en Tiempo Real|Caso Práctico: Sistema de Recomendación en Tiempo Real]]

---

## 🤔 ¿Por qué estructuras de datos para ML?

| Estructura/Algoritmo | Dónde aparece en ML |
|---------------------|---------------------|
| **Hash tables** | Embeddings, feature hashing, diccionarios de vocabulario |
| **Árboles (BST, kd-tree)** | Decision trees, Random Forest, XGBoost, búsqueda de vecinos cercanos |
| **Heaps** | Beam search, Top-K sampling, priority queues en entrenamiento |
| **Grafos** | Redes neuronales (computational graphs), GNNs, PageRank, knowledge graphs |
| **Programación dinámica** | Algoritmos de alineación de secuencias (BioNLP), Viterbi en HMMs |
| **Búsqueda binaria** | Hyperparameter tuning optimizado, búsqueda en embeddings ordenados |

> 💡 **Dato:** Un algoritmo con complejidad `O(n)` vs `O(n log n)` puede ser la diferencia entre entrenar un modelo en 1 hora vs 1 día cuando `n = 10^9`.

---

## 📋 Glosario rápido

| Término | Significado |
|---------|-------------|
| **Big-O** | Cota superior del crecimiento del tiempo/espacio |
| **Big-Ω** | Cota inferior (mejor caso) |
| **Big-Θ** | Cota ajustada (crecimiento exacto) |
| **Amortizado** | Costo promedio por operación en una secuencia |
| **In-place** | Algoritmo que usa espacio adicional constante `O(1)` |
| **Estable** | Algoritmo que preserva el orden relativo de elementos iguales |
