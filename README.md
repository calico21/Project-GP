# Project-GP — End-to-End Differentiable Formula Student Digital Twin

> **Ter26 Formula Student  | 2026**
> 
> A 100% native JAX/Flax differentiable digital twin of the Ter26 FS vehicle, designed for safety-biased setup optimization, stochastic optimal control, and driver coaching.

---

## At a Glance

| Property | Value |
|---|---|
| **Architecture** | 100% JAX/Flax End-to-End Differentiable |
| **State Dimension** | 46 (14 Mechanical DOF + Thermal + Transient Slip) |
| **Tire Model** | Pacejka MF6.2 + 5-Node Thermal + PINN + Matérn 5/2 Sparse GP |
| **Optimal Control** | Diff-WMPC (3-Level Daubechies-4 Wavelet MPC) + UT Stochastic Tubes |
| **Setup Search** | MORL-SB-TRPO (Safety-Biased Trust Region Policy Optimization) |

---

## Table of Contents

1. [Project Overview & Philosophy](#1-project-overview--philosophy)
2. [Neural Port-Hamiltonian Vehicle Dynamics](#2-neural-port-hamiltonian-vehicle-dynamics)
3. [Multi-Fidelity Tire Modeling (MF6.2 + ML)](#3-multi-fidelity-tire-modeling-mf62--ml)
4. [Differentiable Wavelet MPC (Diff-WMPC)](#4-differentiable-wavelet-mpc-diff-wmpc)
5. [Setup Optimization (MORL-SB-TRPO)](#5-setup-optimization-morl-sb-trpo)
6. [Pipeline Execution & File Structure](#6-pipeline-execution--file-structure)

---

## 1. Project Overview & Philosophy

Project-GP abandons traditional numerical simulation frameworks (like CasADi, IPOPT, or point-mass solvers) in favor of a **Deep Learning compiler architecture (JAX/XLA)**. 

Because every equation in the vehicle model is written in pure JAX, the entire physics engine is **differentiable**. We can pass a simulated lap time directly into `jax.grad()`, and the compiler will trace the exact partial derivatives of the lap time with respect to the vehicle's spring rates, damper coefficients, and roll-center heights. 

To achieve this without the model diverging into unphysical states, the mathematics must be strictly bounded. We achieve this by blending classical rigid-body dynamics with Physics-Informed Neural Networks (PINNs), Gaussian Processes (GPs), and Symplectic Integrators.

---

## 2. Neural Port-Hamiltonian Vehicle Dynamics

Instead of using standard Newton-Euler equations of motion, the 14-DOF chassis is modeled as a **Port-Hamiltonian System**. 

### 2.1 The 46-Dimensional State Vector
The state $x$ is evolved forward in time and consists of 46 continuous variables:
* `x[0:14]`: Positions **$q$** (X, Y, Z, roll, pitch, yaw, suspension heave, wheel rotations).
* `x[14:28]`: Velocities **$v$** (Chassis spatial velocities, suspension/wheel rates).
* `x[28:38]`: Tire thermals (Core, ribs, internal gas temperature per axle).
* `x[38:46]`: Transient slip states ($\alpha_t, \kappa_t$) for first-order carcass lag.

### 2.2 The Port-Hamiltonian Equation
The core mechanical evolution is governed by the equation:
$$\dot{x} = (J - R)\nabla H(x) + F_{ext}(x, u)$$

Where:
* $J$ is the skew-symmetric interconnection matrix representing conservative (energy-preserving) dynamics.
* $R$ is the symmetric, positive semi-definite dissipation matrix representing energy loss (damping/friction).
* $H(x)$ is the Hamiltonian (total energy of the system).
* $F_{ext}$ represents the non-conservative external forces (tire forces, aero, gravity).

### 2.3 Neural Energy & Dissipation Networks
To capture real-world compliance (chassis torsional flex) and unmodeled friction that analytical models miss, $H$ and $R$ are augmented by Neural Networks:

1.  **`H_net` (Neural Energy Landscape):** A 128-64-1 Multi-Layer Perceptron (MLP) learns the residual structural energy. 
    * *Mathematical Trick:* The output of `H_net` is multiplied by the squared suspension displacement ($z^2$). By the product rule, the derivative (force) is exactly **0.0** when displacement is zero. This guarantees that the neural network cannot "hallucinate" ghost forces when the car is stationary at equilibrium.
2.  **`R_net` (Neural Dissipation Matrix):** A 128-64 MLP predicts the elements of a lower-triangular Cholesky factor $L$. The matrix is then computed as $R = L L^T$. This mathematically guarantees that $R$ is positive semi-definite, ensuring the network can only *remove* energy from the system (like a real damper) and never spontaneously generate energy.

### 2.4 Differentiable Aero & Symplectic Integration
* **Aero Map:** A 32-32 MLP replaces discrete wind-tunnel lookup tables. Using the `swish` activation function ($f(x) = x \cdot \sigma(x)$), it provides an infinitely differentiable surface mapping pitch, roll, and heave to $C_l$ and $C_d$.
* **Leapfrog Integrator:** Standard Runge-Kutta (RK4) integrators artificially bleed energy over long horizons. To preserve the Hamiltonian energy structure, we use a 5-substep Symplectic Leapfrog (Störmer-Verlet) integrator compiled directly into a `jax.lax.scan` loop.

---

## 3. Multi-Fidelity Tire Modeling (MF6.2 + ML)



Tire grip is the single most critical limit on lap time. The `models/tire_model.py` utilizes a three-layer hybrid approach.

### 3.1 Analytical Pacejka (MF6.2) & Turn Slip
The baseline forces are calculated using the full Pacejka 6.2 Magic Formula, utilizing Hoosier R20 coefficients fitted by the Tire Test Consortium (TTC). It includes load sensitivity, camber sensitivity, and combined-slip reduction ($G_{yk}$ and $G_{xa}$).

**Turn Slip Correction:** When navigating tight radii (like an FS skidpad), the contact patch sees varying slip angles from front to back. We apply a mathematical turn-slip penalty:
$$\phi_t = \frac{a}{R_{path}}$$
Where $a$ is the contact patch half-length and $R_{path}$ is the path curvature (derived from yaw rate). This prevents the model from over-predicting grip by ~1-2% on tight radius events.

### 3.2 5-Node Thermodynamics & Jaeger Flash Temperatures
Tire temperatures dynamically shift the friction coefficient $\mu$. We use the **Jaeger analytical solution** for a sliding semi-infinite solid to compute instantaneous contact patch flash temperatures:
$$T_{flash} = \frac{q \cdot a}{k \sqrt{\pi V_{slide} a / \alpha}}$$
This flash temperature feeds into a 5-node system of differential equations tracking convection, conduction, and friction generation across: Surface Inner, Surface Mid, Surface Outer, Core, and Internal Gas. (Gas temperature dynamically alters tire pressure via Gay-Lussac's law, modifying the vertical stiffness).

### 3.3 PINN and Sparse Gaussian Process (GP)
The analytical baseline is modulated by `TireOperatorPINN`:
1.  **Deterministic Drift:** A Physics-Informed Neural Network predicts minor nonlinear deviations from the Pacejka baseline based on state features like $\sin(2\alpha)$ and $\kappa^3$.
2.  **Stochastic Uncertainty ($\sigma$):** A Sparse Gaussian Process using a Matérn 5/2 kernel calculates a calibrated standard deviation based on the tire's distance from known, well-tested operating conditions (inducing points).
$$k(d) = \sigma^2 \left(1 + \sqrt{5}d + \frac{5}{3}d^2\right) \exp(-\sqrt{5}d)$$
This $\sigma$ allows the optimal control solver to apply a pessimistic Lower Confidence Bound penalty to grip when pushing into unexplored physics regimes.

---

## 4. Differentiable Wavelet MPC (Diff-WMPC)



To find the theoretical minimum lap time, we solve a Non-Linear Programming (NLP) optimal control problem using `DiffWMPCSolver`. 

### 4.1 3-Level Db4 Wavelet Compression
Instead of optimizing steering and throttle inputs at 128 discrete time steps (which results in massive matrices and jagged, un-drivable inputs), we project the control horizon into the frequency domain. 

We apply a **3-Level Discrete Wavelet Transform (DWT)** using Daubechies-4 coefficients. The optimizer searches for the optimal wavelet coefficients, which are then run through an Inverse DWT. This inherently limits the high-frequency bandwidth of the control signals, ensuring the generated steering and throttle traces resemble smooth, biologically feasible human inputs.

### 4.2 Unscented Transform (UT) & Stochastic Tubes


Because the GP tire model and environmental wind yield probabilistic uncertainties, we cannot simulate a single deterministic path. We use the **Unscented Transform**:
1.  We generate 5 sigma points representing nominal, worst-case, and best-case grip/wind conditions.
2.  We simulate all 5 paths simultaneously via `jax.vmap`.
3.  We reconstruct the spatial mean $\mu_n$ and variance $\sigma^2_n$ of the car's lateral position.
This generates a "Stochastic Tube." Track boundaries are enforced via log-barrier penalty functions against the edge of this tube, ensuring the car leaves a safety margin proportional to its current physical uncertainty.

### 4.3 L-BFGS-B Gradient Engineering
The massive unrolled graph is differentiated by `jax.value_and_grad` and solved using SciPy's L-BFGS-B algorithm. 
**Handling NaNs:** If the optimizer explores an unstable input that causes the vehicle states to explode, JAX will return a `NaN` gradient, which normally crashes L-BFGS-B. We implement a Python-level interception: when `NaN` is detected, we return an L2 fallback gradient ($\nabla(1e6 + 0.5\|c\|^2) = c$). This effectively acts as a bowl centered at zero, smoothly pushing the solver back toward smaller, stable wavelet coefficients without aborting the solve.

---

## 5. Setup Optimization (MORL-SB-TRPO)



Optimizing a car for pure lap time often results in a setup so stiff and peaky that human drivers spin out. We deploy **Multi-Objective Reinforcement Learning** to map the Pareto frontier between Maximum Grip and Dynamic Stability.

### 5.1 The Differentiable Evolution Ensemble
Rather than using genetic mutations (NSGA-II), we initialize an ensemble of 20 distinct vehicle setups (springs, ARBs, dampers, CG height, brake bias). Because our physics engine is differentiable, we use the `Adam` optimizer to directly backpropagate the setup gradients from the simulation.

### 5.2 Chebyshev Node Spacing
The loss function balances grip and stability using a weighting factor $\omega$:
$$Reward = \omega \cdot Grip + (1 - \omega) \cdot Stability$$
If $\omega$ is distributed linearly among the 20 members, too many members waste time exploring low-grip, overly stable setups. We distribute $\omega$ using Chebyshev nodes:
$$\omega_i = 0.5 \left( 1 - \cos\left(\frac{i \cdot \pi}{N - 1}\right) \right)$$
This geometrically concentrates ~65% of the ensemble into the critical high-grip, high-performance boundary.

### 5.3 Safety-Biased TRPO & Maximum Entropy
To prevent the 20 setups from prematurely collapsing into a single local minimum:
1.  **TRPO (Trust Region Policy Optimization):** We maintain a 10-iteration lagged reference policy. We apply a Kullback-Leibler (KL) divergence penalty to heavily penalize massive gradient leaps.
$$D_{KL}( \pi_{old} || \pi_{new} ) \le \delta$$
2.  **Maximum Entropy RL:** We apply a bonus $0.005 \sum \log \sigma$ to the loss. This rewards the optimizer for maintaining a wider exploration variance, forcing it to explore the search space until it finds the true global basin.
3.  **Stability Overshoot Cap:** A hard boundary cost activates if the frequency response overshoot exceeds **5.0 rad/s**, completely filtering out corner-pinned numerical artifacts (e.g., $k_f$ pinned at 15,000 N/m). 

---

## 6. Pipeline Execution & File Structure

The pipeline is sequentially executed via `main.py`.

```bash
# Verify the physics engine and graph compilations
python sanity_checks.py

# Run the evolutionary setup optimizer to generate Pareto setups
python main.py --mode setup

# Run the full pipeline (Telemetry -> Track Gen -> Ghost Car -> Coaching)
python main.py --mode full --log /path/to/motec_telemetry.csv

```
---

# Project-GP — Gemelo Digital de Formula Student Diferenciable de Extremo a Extremo

> **Ter26 Formula Student | 2026**
> 
> Un gemelo digital del vehículo FS Ter26, 100% nativo en JAX/Flax y diferenciable, diseñado para la optimización de configuraciones (setup) orientada a la seguridad, control óptimo estocástico y asistencia al piloto (driver coaching).

---

## De un Vistazo

| Propiedad | Valor |
|---|---|
| **Arquitectura** | 100% JAX/Flax Diferenciable de Extremo a Extremo |
| **Dimensión del Estado** | 46 (14 GDL Mecánicos + Térmicos + Deslizamiento Transitorio) |
| **Modelo de Neumático** | Pacejka MF6.2 + Térmico de 5 Nodos + PINN + GP Disperso Matérn 5/2 |
| **Control Óptimo** | Diff-WMPC (MPC de Wavelets Daubechies-4 de 3 Niveles) + Tubos Estocásticos UT |
| **Búsqueda de Setup** | MORL-SB-TRPO (Optimización de Políticas de Región de Confianza Orientada a la Seguridad) |

---

## Índice

1. [Descripción del Proyecto y Filosofía](#1-descripción-del-proyecto-y-filosofía)
2. [Dinámica del Vehículo Neuronal Puerto-Hamiltoniana](#2-dinámica-del-vehículo-neuronal-puerto-hamiltoniana)
3. [Modelado de Neumáticos de Múltiple Fidelidad (MF6.2 + ML)](#3-modelado-de-neumáticos-de-múltiple-fidelidad-mf62--ml)
4. [Control Predictivo por Modelo de Wavelets Diferenciable (Diff-WMPC)](#4-control-predictivo-por-modelo-de-wavelets-diferenciable-diff-wmpc)
5. [Optimización de Configuración (MORL-SB-TRPO)](#5-optimización-de-configuración-morl-sb-trpo)
6. [Ejecución del Pipeline y Estructura de Archivos](#6-ejecución-del-pipeline-y-estructura-de-archivos)

---

## 1. Descripción del Proyecto y Filosofía

Project-GP abandona los marcos de simulación numérica tradicionales (como CasADi, IPOPT o solucionadores de masa puntual) en favor de una **arquitectura de compilación de Deep Learning (JAX/XLA)**. 

Debido a que cada ecuación en el modelo del vehículo está escrita en JAX puro, todo el motor de física es **diferenciable**. Podemos pasar un tiempo de vuelta simulado directamente a `jax.grad()`, y el compilador rastreará las derivadas parciales exactas del tiempo de vuelta con respecto a las constantes elásticas (spring rates), los coeficientes de los amortiguadores y las alturas del centro de balanceo (roll-center). 

Para lograr esto sin que el modelo diverja hacia estados no físicos, las matemáticas deben estar estrictamente limitadas. Logramos esto combinando la dinámica clásica de cuerpos rígidos con Redes Neuronales Informadas por la Física (PINNs), Procesos Gaussianos (GPs) e Integradores Simplécticos.

---

## 2. Dinámica del Vehículo Neuronal Puerto-Hamiltoniana

En lugar de utilizar las ecuaciones de movimiento estándar de Newton-Euler, el chasis de 14 grados de libertad (GDL) se modela como un **Sistema Puerto-Hamiltoniano**. 

### 2.1 El Vector de Estado de 46 Dimensiones
El estado $x$ se propaga en el tiempo y consta de 46 variables continuas:
* `x[0:14]`: Posiciones **$q$** (X, Y, Z, alabeo (roll), cabeceo (pitch), guiñada (yaw), desplazamiento vertical de la suspensión (heave), rotación de las ruedas).
* `x[14:28]`: Velocidades **$v$** (Velocidades espaciales del chasis, tasas de suspensión/ruedas).
* `x[28:38]`: Estados térmicos de los neumáticos (Temperatura del núcleo, costillas y gas interno por eje).
* `x[38:46]`: Estados de deslizamiento transitorio ($\alpha_t, \kappa_t$) para el retraso de la carcasa de primer orden.

### 2.2 La Ecuación Puerto-Hamiltoniana
La evolución mecánica central se rige por la ecuación:
$$\dot{x} = (J - R)\nabla H(x) + F_{ext}(x, u)$$

Donde:
* $J$ es la matriz de interconexión antisimétrica que representa la dinámica conservativa (que preserva la energía).
* $R$ es la matriz de disipación simétrica y semidefinida positiva que representa la pérdida de energía (amortiguación/fricción).
* $H(x)$ es el Hamiltoniano (energía total del sistema).
* $F_{ext}$ representa las fuerzas externas no conservativas (fuerzas de los neumáticos, aerodinámica, gravedad).

### 2.3 Redes Neuronales de Energía y Disipación
Para capturar la flexibilidad real (flexión torsional del chasis) y la fricción no modelada que los modelos analíticos pasan por alto, $H$ y $R$ se aumentan mediante Redes Neuronales:

1.  **`H_net` (Paisaje de Energía Neuronal):** Un Perceptrón Multicapa (MLP) de 128-64-1 aprende la energía estructural residual. 
    * *Truco Matemático:* La salida de `H_net` se multiplica por el cuadrado del desplazamiento de la suspensión ($z^2$). Por la regla del producto, la derivada (fuerza) es exactamente **0.0** cuando el desplazamiento es cero. Esto garantiza que la red neuronal no pueda "alucinar" fuerzas fantasma cuando el coche está estacionario en equilibrio.
2.  **`R_net` (Matriz de Disipación Neuronal):** Un MLP de 128-64 predice los elementos de un factor de Cholesky triangular inferior $L$. Luego, la matriz se calcula como $R = L L^T$. Esto garantiza matemáticamente que $R$ sea semidefinida positiva, asegurando que la red solo pueda *eliminar* energía del sistema (como un amortiguador real) y nunca generar energía espontáneamente.

### 2.4 Aerodinámica Diferenciable e Integración Simpléctica
* **Mapa Aerodinámico:** Un MLP de 32-32 reemplaza las tablas de búsqueda discretas del túnel de viento. Utilizando la función de activación `swish` ($f(x) = x \cdot \sigma(x)$), proporciona una superficie infinitamente diferenciable que mapea el cabeceo, alabeo y la altura (heave) con $C_l$ y $C_d$.
* **Integrador Leapfrog (Salto de Rana):** Los integradores estándar de Runge-Kutta (RK4) purgan artificialmente la energía en horizontes largos. Para preservar la estructura de energía Hamiltoniana, utilizamos un integrador simpléctico Leapfrog (Störmer-Verlet) de 5 subpasos compilado directamente en un bucle `jax.lax.scan`.

---

## 3. Modelado de Neumáticos de Múltiple Fidelidad (MF6.2 + ML)

[Imagen de la curva de la Fórmula Mágica de Pacejka]

El agarre de los neumáticos es el límite más crítico para el tiempo de vuelta. El archivo `models/tire_model.py` utiliza un enfoque híbrido de tres capas.

### 3.1 Pacejka Analítico (MF6.2) y Corrección de Deslizamiento en Curva
Las fuerzas base se calculan utilizando la Fórmula Mágica completa de Pacejka 6.2, utilizando los coeficientes Hoosier R20 ajustados por el Tire Test Consortium (TTC). Incluye sensibilidad a la carga, sensibilidad al camber y reducción por deslizamiento combinado ($G_{yk}$ y $G_{xa}$).

**Corrección de Deslizamiento en Curva (Turn Slip):** Al navegar por radios cerrados (como un skidpad de FS), el parche de contacto experimenta ángulos de deriva variables de adelante hacia atrás. Aplicamos una penalización matemática por el deslizamiento de giro:
$$\phi_t = \frac{a}{R_{path}}$$
Donde $a$ es la mitad de la longitud del parche de contacto y $R_{path}$ es la curvatura de la trayectoria (derivada de la tasa de guiñada). Esto evita que el modelo sobreestime el agarre en un ~1-2% en eventos de radio cerrado.

### 3.2 Termodinámica de 5 Nodos y Temperaturas Flash de Jaeger
Las temperaturas de los neumáticos cambian dinámicamente el coeficiente de fricción $\mu$. Utilizamos la **solución analítica de Jaeger** para un sólido semi-infinito deslizante para calcular las temperaturas flash instantáneas del parche de contacto:
$$T_{flash} = \frac{q \cdot a}{k \sqrt{\pi V_{slide} a / \alpha}}$$
Esta temperatura flash alimenta un sistema de 5 nodos de ecuaciones diferenciales que rastrean la convección, conducción y generación de fricción a través de: Superficie Interior, Superficie Media, Superficie Exterior, Núcleo y Gas Interno. (La temperatura del gas altera dinámicamente la presión del neumático mediante la ley de Gay-Lussac, modificando la rigidez vertical).

### 3.3 PINN y Proceso Gaussiano (GP) Disperso
La línea base analítica es modulada por `TireOperatorPINN`:
1.  **Deriva Determinista:** Una Red Neuronal Informada por la Física predice pequeñas desviaciones no lineales de la línea base de Pacejka basadas en características de estado como $\sin(2\alpha)$ y $\kappa^3$.
2.  **Incertidumbre Estocástica ($\sigma$):** Un Proceso Gaussiano Disperso (Sparse GP) que utiliza un kernel Matérn 5/2 calcula una desviación estándar calibrada basada en la distancia del neumático a las condiciones operativas conocidas y probadas (puntos inductores).
$$k(d) = \sigma^2 \left(1 + \sqrt{5}d + \frac{5}{3}d^2\right) \exp(-\sqrt{5}d)$$
Esta $\sigma$ permite al solucionador de control óptimo aplicar una penalización pesimista (Límite Inferior de Confianza o Lower Confidence Bound) al agarre cuando se empuja hacia regímenes físicos inexplorados.

---

## 4. Control Predictivo por Modelo de Wavelets Diferenciable (Diff-WMPC)

[Imagen de la Wavelet de Daubechies 4]

Para encontrar el tiempo de vuelta mínimo teórico, resolvemos un problema de control óptimo de Programación No Lineal (NLP) utilizando `DiffWMPCSolver`.

### 4.1 Compresión Wavelet Db4 de 3 Niveles
En lugar de optimizar las entradas de dirección y acelerador en 128 pasos de tiempo discretos (lo que resulta en matrices masivas y entradas dentadas imposibles de conducir), proyectamos el horizonte de control en el dominio de la frecuencia. 

Aplicamos una **Transformada Wavelet Discreta (DWT) de 3 Niveles** utilizando coeficientes de Daubechies-4. El optimizador busca los coeficientes wavelet óptimos, que luego pasan por una DWT Inversa. Esto limita inherentemente el ancho de banda de alta frecuencia de las señales de control, asegurando que las trazas de dirección y aceleración generadas se asemejen a entradas humanas suaves y biológicamente factibles.

### 4.2 Tubos Estocásticos y Transformada Unscented (UT)
[Imagen de los puntos sigma de la Transformada Unscented]

Debido a que el modelo de neumático GP y el viento ambiental producen incertidumbres probabilísticas, no podemos simular una sola trayectoria determinista. Utilizamos la **Transformada Unscented**:
1.  Generamos 5 puntos sigma que representan condiciones de agarre/viento nominales, peores y mejores.
2.  Simulamos las 5 trayectorias simultáneamente a través de `jax.vmap`.
3.  Reconstruimos la media espacial $\mu_n$ y la varianza $\sigma^2_n$ de la posición lateral del coche.
Esto genera un "Tubo Estocástico". Los límites de la pista se hacen cumplir a través de funciones de penalización de barrera logarítmica contra el borde de este tubo, asegurando que el coche deje un margen de seguridad proporcional a su incertidumbre física actual.

### 4.3 Ingeniería de Gradientes L-BFGS-B
El grafo masivo desenrollado se diferencia mediante `jax.value_and_grad` y se resuelve utilizando el algoritmo L-BFGS-B de SciPy. 
**Manejo de NaNs:** Si el optimizador explora una entrada inestable que hace que los estados del vehículo exploten, JAX devolverá un gradiente `NaN`, lo que normalmente bloquea L-BFGS-B. Implementamos una intercepción a nivel de Python: cuando se detecta `NaN`, devolvemos un gradiente de respaldo L2 ($\nabla(1e6 + 0.5\|c\|^2) = c$). Esto actúa efectivamente como un cuenco centrado en cero, empujando suavemente al solucionador hacia coeficientes wavelet más pequeños y estables sin abortar la resolución.

---

## 5. Optimización de Configuración (MORL-SB-TRPO)

[Imagen de la Curva de Compromiso del Frente de Pareto]

Optimizar un coche para el tiempo de vuelta puro a menudo resulta en un setup tan rígido y extremo que los pilotos humanos hacen un trompo. Implementamos **Aprendizaje por Refuerzo Multiobjetivo (MORL)** para mapear la frontera de Pareto entre el Agarre Máximo (Grip) y la Estabilidad Dinámica.

### 5.1 El Conjunto de Evolución Diferenciable
En lugar de utilizar mutaciones genéticas (NSGA-II), inicializamos un conjunto (ensemble) de 20 setups de vehículos distintos (resortes, barras estabilizadoras, amortiguadores, altura del centro de gravedad, balance de frenada). Debido a que nuestro motor de física es diferenciable, utilizamos el optimizador `Adam` para propagar hacia atrás los gradientes de configuración directamente desde la simulación.

### 5.2 Espaciado de Nodos de Chebyshev
La función de pérdida equilibra el agarre y la estabilidad utilizando un factor de peso $\omega$:
$$Reward = \omega \cdot Grip + (1 - \omega) \cdot Stability$$
Si $\omega$ se distribuye linealmente entre los 20 miembros, demasiados miembros pierden el tiempo explorando configuraciones de bajo agarre y excesivamente estables. Distribuimos $\omega$ utilizando los nodos de Chebyshev:
$$\omega_i = 0.5 \left( 1 - \cos\left(\frac{i \cdot \pi}{N - 1}\right) \right)$$
Esto concentra geométricamente ~65% del conjunto en el límite crítico de alto agarre y alto rendimiento.

### 5.3 TRPO Orientado a la Seguridad y Máxima Entropía
Para evitar que los 20 setups colapsen prematuramente en un único mínimo local:
1.  **TRPO (Optimización de Políticas de Región de Confianza):** Mantenemos una política de referencia rezagada de 10 iteraciones. Aplicamos una penalización por divergencia de Kullback-Leibler (KL) para penalizar fuertemente los saltos masivos de gradiente.
$$D_{KL}( \pi_{old} || \pi_{new} ) \le \delta$$
2.  **RL de Máxima Entropía:** Aplicamos un bono de $0.005 \sum \log \sigma$ a la pérdida. Esto recompensa al optimizador por mantener una varianza de exploración más amplia, obligándolo a explorar el espacio de búsqueda hasta encontrar la verdadera cuenca global.
3.  **Límite de Sobreimpulso de Estabilidad (Overshoot Cap):** Un costo de límite estricto se activa si el sobreimpulso de la respuesta de frecuencia excede los **5.0 rad/s**, filtrando completamente los artefactos numéricos atrapados en los límites (por ejemplo, $k_f$ clavado en 15,000 N/m). 

---

## 6. Ejecución del Pipeline y Estructura de Archivos

El pipeline se ejecuta secuencialmente a través de `main.py`.

```bash
# Verifica el motor de física y las compilaciones de grafos
python sanity_checks.py

# Ejecuta el optimizador evolutivo de configuración para generar setups de Pareto
python main.py --mode setup

# Ejecuta el pipeline completo (Telemetría -> Generación de Pista -> Coche Fantasma -> Asistencia/Coaching)
python main.py --mode full --log /ruta/a/motec_telemetria.csv