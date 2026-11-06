
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstddef>

using namespace std;

// CONFIGURACION GENERAL

const double TOLERANCIA = 1e-4;
const int HILOS_POR_BLOQUE = 256;

#define CUDA_CHECK(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
        cerr << "Error CUDA: "                              \
             << cudaGetErrorString(err)                     \
             << " en linea " << __LINE__ << endl;           \
        exit(EXIT_FAILURE);                                \
    }                                                       \
} while (0)

// ESTRUCTURA DE FILTRO

struct Filtro {
    string nombre;
    int tam;
    vector<float> mascara;
};
// DEFINICION DE FILTROS

Filtro filtroMedia(int k) {
    Filtro f;

    f.nombre = "Media";
    f.tam = k;

    f.mascara.assign(
        k * k,
        1.0f / static_cast<float>(k * k)
    );

    return f;
}

Filtro filtroSharpen() {
    return {
        "Sharpen",
        3,
        {
             0, -1,  0,
            -1,  5, -1,
             0, -1,  0
        }
    };
}

Filtro filtroLaplaciano() {
    return {
        "Laplaciano",
        3,
        {
             0,  1,  0,
             1, -4,  1,
             0,  1,  0
        }
    };
}
// CONVOLUCION SECUENCIAL CPU
// Bordes: valores fuera de la imagen = 0

void convolucionCPU(
    const vector<float>& entrada,
    vector<float>& salida,
    int ancho,
    int alto,
    const Filtro& filtro
) {
    int radio = filtro.tam / 2;

    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {

            float suma = 0.0f;

            for (int ky = 0; ky < filtro.tam; ky++) {
                for (int kx = 0; kx < filtro.tam; kx++) {

                    int ix = x + kx - radio;
                    int iy = y + ky - radio;

                    if (
                        ix >= 0 &&
                        ix < ancho &&
                        iy >= 0 &&
                        iy < alto
                    ) {
                        suma +=
                            entrada[iy * ancho + ix] *
                            filtro.mascara[
                                ky * filtro.tam + kx
                            ];
                    }
                }
            }

            salida[y * ancho + x] = suma;
        }
    }
}
// CONVOLUCION PARALELA GPU
// Cada hilo CUDA calcula un pixel

__global__ void convolucionGPU(
    const float* entrada,
    float* salida,
    int ancho,
    int alto,
    const float* mascara,
    int tam
) {
    int x =
        blockIdx.x * blockDim.x + threadIdx.x;

    int y = blockIdx.y;

    // Evitar accesos fuera de la imagen.
    if (x >= ancho || y >= alto) {
        return;
    }

    int radio = tam / 2;
    float suma = 0.0f;

    for (int ky = 0; ky < tam; ky++) {
        for (int kx = 0; kx < tam; kx++) {

            int ix = x + kx - radio;
            int iy = y + ky - radio;

            if (
                ix >= 0 &&
                ix < ancho &&
                iy >= 0 &&
                iy < alto
            ) {
                suma +=
                    entrada[iy * ancho + ix] *
                    mascara[ky * tam + kx];
            }
        }
    }

    salida[y * ancho + x] = suma;
}

// EJECUTAR CONVOLUCION EN GPU

void ejecutarGPU(
    const vector<float>& entrada,
    vector<float>& salida,
    int ancho,
    int alto,
    const Filtro& filtro,
    int hilos
) {
    size_t total =
        static_cast<size_t>(ancho) *
        static_cast<size_t>(alto);

    size_t bytesImagen =
        total * sizeof(float);

    size_t bytesMascara =
        filtro.mascara.size() * sizeof(float);

    float* dEntrada = nullptr;
    float* dSalida = nullptr;
    float* dMascara = nullptr;

    // Reservar memoria GPU.
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&dEntrada),
        bytesImagen
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&dSalida),
        bytesImagen
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&dMascara),
        bytesMascara
    ));

    // Copiar imagen CPU -> GPU.
    CUDA_CHECK(cudaMemcpy(
        dEntrada,
        entrada.data(),
        bytesImagen,
        cudaMemcpyHostToDevice
    ));

    // Copiar mascara CPU -> GPU.
    CUDA_CHECK(cudaMemcpy(
        dMascara,
        filtro.mascara.data(),
        bytesMascara,
        cudaMemcpyHostToDevice
    ));

    // Configuracion de lanzamiento.
    dim3 bloque(hilos);

    dim3 rejilla(
        (ancho + hilos - 1) / hilos,
        alto
    );

    // Ejecutar kernel.
    convolucionGPU<<<rejilla, bloque>>>(
        dEntrada,
        dSalida,
        ancho,
        alto,
        dMascara,
        filtro.tam
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copiar resultado GPU -> CPU.
    CUDA_CHECK(cudaMemcpy(
        salida.data(),
        dSalida,
        bytesImagen,
        cudaMemcpyDeviceToHost
    ));

    // Liberar memoria GPU.
    CUDA_CHECK(cudaFree(dEntrada));
    CUDA_CHECK(cudaFree(dSalida));
    CUDA_CHECK(cudaFree(dMascara));
}

// MOSTRAR MATRIZ

void mostrarMatriz(
    const vector<float>& datos,
    int ancho,
    int alto
) {
    cout << fixed << setprecision(4);

    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {

            cout << setw(10)
                 << datos[y * ancho + x];
        }

        cout << "\n";
    }

    cout << defaultfloat;
}

// VALIDACION NUMERICA

bool validar(
    const vector<float>& referencia,
    const vector<float>& resultado,
    const string& descripcion
) {
    if (referencia.size() != resultado.size()) {
        cerr << "ERROR: dimensiones incompatibles.\n";
        return false;
    }

    double errorMaximo = 0.0;
    double sumaErrores = 0.0;

    for (size_t i = 0; i < referencia.size(); i++) {

        double error = fabs(
            static_cast<double>(referencia[i]) -
            static_cast<double>(resultado[i])
        );

        if (!isfinite(error)) {
            cerr << "ERROR: valor no finito.\n";
            return false;
        }

        errorMaximo = max(errorMaximo, error);
        sumaErrores += error;
    }

    double errorMedio = referencia.empty()
        ? 0.0
        : sumaErrores / referencia.size();

    bool correcta =
        errorMaximo <= TOLERANCIA;

    cout << scientific << setprecision(8);

    cout << descripcion << "\n";

    cout << "Error maximo: "
         << errorMaximo << "\n";

    cout << "Error medio: "
         << errorMedio << "\n";

    cout << "Tolerancia: "
         << TOLERANCIA << "\n";

    cout << "Validacion: "
         << (correcta ? "CORRECTA" : "REVISAR")
         << "\n\n";

    cout << defaultfloat;

    return correcta;
}
// PRUEBA 1
// VALIDACION MANUAL CON MATRIZ 5x5

bool pruebaManual() {

    cout << "\n====================================\n";
    cout << " PRUEBA 1: VALIDACION MANUAL 5x5\n";
    cout << "====================================\n";

    const int ancho = 5;
    const int alto = 5;

    vector<float> entrada = {
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    cout << "\nMatriz de entrada:\n";
    mostrarMatriz(entrada, ancho, alto);

    vector<Filtro> filtros = {
        filtroMedia(3),
        filtroSharpen(),
        filtroLaplaciano()
    };

    // Resultados de referencia calculados
    // manualmente con bordes de valor cero.

    vector<vector<float>> esperados = {

        // MEDIA 3x3
        {
            16.0f/9,  27.0f/9,  33.0f/9,  39.0f/9,  28.0f/9,
            39.0f/9,  63.0f/9,  72.0f/9,  81.0f/9,  57.0f/9,
            69.0f/9, 108.0f/9, 117.0f/9, 126.0f/9,  87.0f/9,
            99.0f/9, 153.0f/9, 162.0f/9, 171.0f/9, 117.0f/9,
            76.0f/9, 117.0f/9, 123.0f/9, 129.0f/9,  88.0f/9
        },

        // SHARPEN 3x3
        {
            -3, -1,  1,  3, 11,
            11,  7,  8,  9, 21,
            21, 12, 13, 14, 31,
            31, 17, 18, 19, 41,
            67, 49, 51, 53, 81
        },

        // LAPLACIANO 3x3
        {
             4,   3,   2,   1,  -6,
            -5,   0,   0,   0, -11,
           -10,   0,   0,   0, -16,
           -15,   0,   0,   0, -21,
           -46, -27, -28, -29, -56
        }
    };

    bool todasCorrectas = true;

    for (size_t i = 0; i < filtros.size(); i++) {

        const Filtro& filtro = filtros[i];

        vector<float> salidaCPU(25);
        vector<float> salidaGPU(25);

        convolucionCPU(
            entrada,
            salidaCPU,
            ancho,
            alto,
            filtro
        );

        ejecutarGPU(
            entrada,
            salidaGPU,
            ancho,
            alto,
            filtro,
            HILOS_POR_BLOQUE
        );

        cout << "\n------------------------------------\n";
        cout << "Filtro: " << filtro.nombre << "\n";

        cout << "\nResultado CPU:\n";
        mostrarMatriz(salidaCPU, ancho, alto);

        cout << "\nResultado GPU:\n";
        mostrarMatriz(salidaGPU, ancho, alto);

        bool manualCPU = validar(
            esperados[i],
            salidaCPU,
            "Referencia manual vs CPU"
        );

        bool manualGPU = validar(
            esperados[i],
            salidaGPU,
            "Referencia manual vs GPU"
        );

        bool cpuGPU = validar(
            salidaCPU,
            salidaGPU,
            "CPU vs GPU"
        );

        if (!manualCPU || !manualGPU || !cpuGPU) {
            todasCorrectas = false;
        }
    }

    return todasCorrectas;
}

// PRUEBA 2
// DIMENSIONES NO MULTIPLOS DE 256

bool pruebaDimensionesIrregulares() {

    cout << "\n====================================\n";
    cout << " PRUEBA 2: DIMENSIONES IRREGULARES\n";
    cout << "====================================\n";

    const int ancho = 1001;
    const int alto = 1003;
    const int hilos = HILOS_POR_BLOQUE;

    size_t total =
        static_cast<size_t>(ancho) *
        static_cast<size_t>(alto);

    vector<float> entrada(total);

    // Generacion reproducible de pixeles.
    for (size_t i = 0; i < total; i++) {
        entrada[i] = static_cast<float>(
            (i * 17 + 13) % 256
        );
    }

    Filtro filtro = filtroMedia(3);

    vector<float> salidaCPU(total);
    vector<float> salidaGPU(total);

    convolucionCPU(
        entrada,
        salidaCPU,
        ancho,
        alto,
        filtro
    );

    ejecutarGPU(
        entrada,
        salidaGPU,
        ancho,
        alto,
        filtro,
        hilos
    );

    int bloquesPorFila =
        (ancho + hilos - 1) / hilos;

    int bloquesTotales =
        bloquesPorFila * alto;

    int hilosSobrantesPorFila =
        bloquesPorFila * hilos - ancho;

    cout << "Resolucion: "
         << ancho << " x " << alto << "\n";

    cout << "Pixeles: "
         << total << "\n";

    cout << "Hilos por bloque: "
         << hilos << "\n";

    cout << "Bloques por fila: "
         << bloquesPorFila << "\n";

    cout << "Bloques totales: "
         << bloquesTotales << "\n";

    cout << "Hilos sobrantes por fila: "
         << hilosSobrantesPorFila << "\n\n";

    return validar(
        salidaCPU,
        salidaGPU,
        "Validacion CPU vs GPU (1001x1003)"
    );
}
// PROGRAMA PRINCIPAL

int main() {

    cout << "====================================\n";
    cout << " VALIDACIONES DE CONVOLUCION CUDA\n";
    cout << "====================================\n";

    int dispositivos = 0;

    CUDA_CHECK(
        cudaGetDeviceCount(&dispositivos)
    );

    if (dispositivos == 0) {
        cerr << "No se encontro una GPU CUDA.\n";
        return EXIT_FAILURE;
    }

    cudaDeviceProp propiedades;

    CUDA_CHECK(cudaGetDeviceProperties(
        &propiedades,
        0
    ));

    cout << "GPU: "
         << propiedades.name << "\n";

    cout << "Capacidad CUDA: "
         << propiedades.major << "."
         << propiedades.minor << "\n";

    // Ejecutar las dos pruebas.
    bool prueba1 = pruebaManual();

    bool prueba2 =
        pruebaDimensionesIrregulares();

    // Resumen.
    cout << "\n====================================\n";
    cout << " RESUMEN FINAL\n";
    cout << "====================================\n";

    cout << "Validacion manual 5x5: "
         << (prueba1 ? "CORRECTA" : "REVISAR")
         << "\n";

    cout << "Dimensiones 1001x1003: "
         << (prueba2 ? "CORRECTA" : "REVISAR")
         << "\n";

    bool todoCorrecto =
        prueba1 && prueba2;

    cout << "\nRESULTADO GENERAL: "
         << (todoCorrecto ? "CORRECTO" : "REVISAR")
         << "\n";

    return todoCorrecto
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
