Procesamiento paralelo de imágenes con CUDA C++

Universidad Politécnica Salesiana (UPS) · Computación Paralela

GPU utilizada: NVIDIA GeForce RTX 3070 (CUDA 8.6)

Objetivo

Comparar la convolución secuencial en CPU y paralela en GPU mediante CUDA, evaluar el rendimiento y validar la exactitud de los resultados.

Filtros implementados

- Media: suavizado con máscaras de 3×3, 9×9 y 21×21.

- Sharpen (3×3): realce de nitidez.

- Laplaciano (3×3): detección de cambios de intensidad.

Se emplea extensión con ceros en los bordes de la imagen.
Requisitos

- Windows con GPU NVIDIA compatible con CUDA.

- NVIDIA CUDA Toolkit (nvcc).

- Compilador C++ compatible con CUDA (Visual Studio C++ en Windows).

Estructura del proyecto

practica1/

├── codigo/

│   ├── preparar_imagenes.cpp

│   ├── filtros.cu

│   ├── experimentos.cu

│   ├── experimentos_optimizado.cu

│   ├── main.cu

│   ├── stb_image.h

│   └── stb_image_write.h

├── imagenes/

│   ├── 6x6orig.jpg

│   ├── imagen_2000.pgm

│   ├── imagen_4000.pgm

│   └── imagen_6000.pgm

└── resultados/

    ├── mediciones.csv

    ├── resumen.csv

    ├── mediciones_optimizadas.csv

    └── resumen_optimizado.csv

Las imágenes de entrada y los archivos generados deben estar disponibles en las rutas indicadas para reproducir las pruebas.
Compilación y ejecución

Desde la carpeta raíz practica1 en PowerShell:

nvcc -O2 -std=c++17 codigo\main.cu -o codigo\main.exe

nvcc -O2 -std=c++17 codigo\filtros.cu -o codigo\filtros.exe

nvcc -O2 -std=c++17 codigo\experimentos.cu -o codigo\experimentos.exe

nvcc -O2 -std=c++17 codigo\experimentos_optimizado.cu -o codigo\experimentos_optimizado.exe

.\codigo\main.exe

.\codigo\filtros.exe

.\codigo\experimentos.exe

.\codigo\experimentos_optimizado.exe

Experimentos y resultados

Se evaluaron imágenes de 2000×2000, 4000×4000 y 6000×6000, distintas máscaras y bloques de 128, 256 y 512 hilos.

Cada una de las 9 configuraciones se ejecutó 5 veces en las versiones original y optimizada.

Prueba destacada	CPU optimizada	 GPU total optimizada      Aceleración total

Media 9×9,           2000×2000	      154.026 ms	3.010 ms	 51.17×
Media 9×9,           4000×4000	      601.530 ms	10.986 ms	 54.75×
Media 9×9,           6000×6000	      1365.360 ms	46.069 ms	 29.64×


Los valores corresponden a medianas de las mediciones. Los tiempos pueden variar según la carga y el estado de la GPU.

Validación

- 9/9 configuraciones correctas en cada implementación.

- Comparación CPU/GPU y referencias manuales para una matriz de 5×5.

- Prueba de dimensiones no múltiplos del bloque: 1001×1003.

- Tolerancia numérica: 1e-4.

Conclusión

CUDA redujo el tiempo de procesamiento en los experimentos realizados. La versión optimizada obtuvo menores tiempos totales de GPU en las nueve configuraciones; 256 hilos por bloque presentó la mejor mediana observada para Media 9×9 de 4000×4000, aunque se detectó variabilidad entre repeticiones.
