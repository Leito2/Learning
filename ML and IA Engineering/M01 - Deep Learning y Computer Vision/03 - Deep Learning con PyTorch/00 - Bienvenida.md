# 🧠 Bienvenida al Curso de Deep Learning con PyTorch

Este curso está diseñado para llevarte desde los fundamentos matemáticos de las redes neuronales hasta la implementación de soluciones avanzadas en visión por computadora y procesamiento de secuencias, utilizando PyTorch como framework principal.

![Red neuronal artificial](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/640px-Colored_neural_network.svg.png)

```mermaid
flowchart LR
    A[Redes Neuronales] --> B[CNNs]
    B --> C[RNNs]
    C --> D[Training Strategies]
    D --> E[Transfer Learning]
    E --> F[Caso Médico]
```

Comprender Deep Learning no se trata solo de saber invocar capas predefinidas; se trata de entender por qué ciertas arquitecturas emergen, cómo fluye la información y el gradiente, y cómo tomar decisiones informadas en problemas reales de Machine Learning.

---

## 📚 Índice del Curso

1. [[01 - Introduccion a Redes Neuronales]]
2. [[02 - CNNs y Arquitecturas Modernas]]
3. [[03 - RNNs y Secuencias]]
4. [[04 - Training Strategies]]
5. [[05 - Transfer Learning]]
6. [[06 - Caso Practico - Clasificador de Imagenes Medico]]

---

## 📖 Glosario de Términos

| Término | Definición |
|---------|------------|
| **Perceptrón** | Unidad computacional que recibe entradas, las pondera, suma un sesgo y aplica una función de activación. Es la célula fundamental de una red neuronal. |
| **Backpropagation** | Algoritmo que calcula el gradiente de la función de pérdida con respecto a los parámetros de la red aplicando la regla de la cadena, propagando el error desde la salida hacia la entrada. |
| **Gradiente** | Vector de derivadas parciales que indica la dirección y magnitud del cambio más pronunciado de una función. En Deep Learning, guía la actualización de los pesos. |
| **Epoch** | Una pasada completa sobre todo el conjunto de datos de entrenamiento. |
| **Batch** | Subconjunto de datos procesado en una única iteración de forward y backward pass. |
| **Loss** | Función que cuantifica la discrepancia entre la predicción del modelo y el valor real. El objetivo del entrenamiento es minimizarla. |
| **Optimizer** | Algoritmo que ajusta los parámetros de la red utilizando los gradientes calculados (ej. SGD, Adam). |
| **CNN** | Convolutional Neural Network; arquitectura diseñada para datos con estructura de cuadrícula, especialmente imágenes. Utiliza convoluciones para extraer características espaciales. |
| **RNN** | Recurrent Neural Network; arquitectura para datos secuenciales donde la salida anterior influye en la siguiente, manteniendo un estado oculto. |
| **LSTM** | Long Short-Term Memory; variante de RNN que utiliza compuertas (input, forget, output) para mitigar el problema del gradiente desvanecido y capturar dependencias a largo plazo. |
| **GRU** | Gated Recurrent Unit; simplificación de LSTM con dos compuertas (update y reset) que reduce la cantidad de parámetros. |
| **Dropout** | Técnica de regularización que desactiva aleatoriamente neuronas durante el entrenamiento para prevenir el sobreajuste. |
| **Batch Norm** | Normalización de las activaciones dentro de un mini-batch para estabilizar y acelerar el entrenamiento, reduciendo la covarianza interna de las variables. |
| **Transfer Learning** | Reutilización de un modelo pre-entrenado en un dominio o tarea grande como punto de partida para una tarea específica, ahorrando tiempo y datos. |
| **Fine-tuning** | Proceso de ajustar los pesos de un modelo pre-entrenado (total o parcialmente) en un nuevo conjunto de datos para adaptarlo a una tarea particular. |

---

## 🎯 Objetivos de Aprendizaje

Al finalizar este curso serás capaz de:

1. Explicar matemáticamente el forward pass y el backpropagation en una red neuronal profunda.
2. Seleccionar y diseñar arquitecturas de CNNs y RNNs según la naturaleza del problema.
3. Implementar estrategias de entrenamiento avanzadas (optimizadores adaptativos, regularización, augmentación) para mejorar la convergencia y generalización.
4. Aplicar transfer learning de manera efectiva, decidiendo qué capas congelar y cómo ajustar tasas de aprendizaje diferenciadas.
5. Construir un pipeline completo en PyTorch para un problema médico real, incluyendo manejo de desbalanceo, métricas clínicas e interpretabilidad.

---

💡 **Tip mnemotécnico:** Piensa en PyTorch como un ecosistema de *tensores dinámicos* (computación diferenciable automática). Si recuerdas que todo se reduce a operaciones sobre tensores y un grafo de derivadas, dominarás cualquier arquitectura.
