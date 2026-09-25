# VQE del modelo de Ising con campo transverso en GPU

Este experimento simula un modelo de Ising unidimensional de 20 spins con
interacción J = 1 y campo transversal h variando de 0.2 a 2.0 en pasos de
0.2. Para cada valor de h se dejan preparados tres VQE con semillas
independientes.

El Hamiltoniano usa condiciones abiertas:

    H = -J sum_i Z_i Z_(i+1) - h sum_i X_i

El ansatz es:

    [(RX, RY, RZ) en los 20 spins -> anillo de CNOTs]
        x 3 anillos de CNOTs
    [(RX, RY, RZ) en los 20 spins]

Es decir, hay 4 bloques de rotaciones, 3 anillos periódicos de CNOTs y
4 x 20 x 3 = 240 parámetros variacionales. El estado se representa como un
tensor de orden 20 y cada puerta se contrae con sus índices físicos mediante
PyTorch. El tensor y las contracciones se ejecutan en CUDA cuando está
disponible; no se construye una matriz densa de Hamiltoniano de tamaño
2**20 x 2**20.

## Entorno

El entorno existente se llama tensor-networks (Conda no admite el espacio en
el nombre). La dependencia principal añadida es PyTorch con CUDA:

    conda activate tensor-networks
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

Verificación rápida:

    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

## Notebook

Abrir ising_tfim_gpu_vqe.ipynb desde este directorio. La celda de ejecución
está configurada deliberadamente con RUN_ALL = False: corre un solo caso de
validación (h = 0.2, semilla 11) y guarda
results/verification_single_vqe.json. Para ejecutar el barrido completo,
cambiar explícitamente RUN_ALL = True; eso lanzará 30 optimizaciones.

La configuración ajustada usa 500 pasos de Adam para los tres VQE de h=0.2,
150 pasos de Adam para cada campo posterior, learning rate 0.03 y hasta 80
iteraciones de LBFGS. Cada semilla recicla los parámetros finales del campo
anterior. El nuevo barrido se guarda en
results/all_vqe_results_adam500_later150_lbfgs80.json.

La validación comprueba que CUDA fue seleccionada, que el estado final está
normalizado, que el número de parámetros es 240 y que el optimizador reduce la
energía inicial.

## Análisis VQE frente a ED

El notebook ising_tfim_observables_analysis.ipynb reconstruye los 30 estados
VQE a partir de sus parámetros finales y los compara con los estados ED.
Calcula Mx, Mz, |Mz|, Mz², correlaciones ZZ a varias distancias, energía
reconstruida, error energético y fidelidad |<psi_ED|psi_VQE>|². Las figuras y
la tabla completa se guardan en results/.

El notebook ising_tfim_gradient_analysis.ipynb muestra la norma del gradiente
durante las 80 iteraciones para las 30 ejecuciones y la media con dispersión
para cada campo.
