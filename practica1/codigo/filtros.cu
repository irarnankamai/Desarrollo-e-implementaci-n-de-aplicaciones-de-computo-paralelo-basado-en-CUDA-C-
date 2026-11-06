
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <filesystem>

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        cerr << "Error CUDA: " << cudaGetErrorString(err) \
             << " linea " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 1. LEER IMAGEN PGM EN ESCALA DE GRISES

bool leerPGM(const string& ruta,
             vector<unsigned char>& imagen,
             int& ancho, int& alto) {

    ifstream archivo(ruta, ios::binary);

    if (!archivo) {
        cerr << "No se pudo abrir: " << ruta << endl;
        return false;
    }

    string formato;
    int maximo;

    archivo >> formato >> ancho >> alto >> maximo;

    if (!archivo || formato != "P5" ||
        maximo != 255 || ancho <= 0 || alto <= 0) {
        cerr << "Formato PGM no valido." << endl;
        return false;
    }

    archivo.get();

    imagen.resize(static_cast<size_t>(ancho) * alto);

    archivo.read(
        reinterpret_cast<char*>(imagen.data()),
        static_cast<streamsize>(imagen.size())
    );

    return archivo.gcount() ==
           static_cast<streamsize>(imagen.size());
}
// 2. GUARDAR IMAGEN DIRECTAMENTE EN PNG

bool guardarPNG(const string& ruta,
                const vector<unsigned char>& imagen,
                int ancho,
                int alto) {

    int resultado = stbi_write_png(
        ruta.c_str(),
        ancho,
        alto,
        1,
        imagen.data(),
        ancho
    );

    return resultado != 0;
}
// 3. COEFICIENTES DE LOS FILTROS
// filtro 0: Media 3x3
// filtro 1: Sharpen 3x3
// filtro 2: Laplaciano 3x3

__host__ __device__
float coeficiente(int filtro, int dx, int dy) {

    if (filtro == 0) {
        return 1.0f / 9.0f;
    }

    if (filtro == 1) {
        if (dx == 0 && dy == 0) {
            return 5.0f;
        }

        if (abs(dx) + abs(dy) == 1) {
            return -1.0f;
        }

        return 0.0f;
    }

    if (dx == 0 && dy == 0) {
        return -4.0f;
    }

    if (abs(dx) + abs(dy) == 1) {
        return 1.0f;
    }

    return 0.0f;
}
// 4. CONVOLUCION SECUENCIAL EN CPU

void filtroCPU(const unsigned char* entrada,
               float* salida,
               int ancho,
               int alto,
               int filtro) {

    for (int y = 0; y < alto; y++) {

        for (int x = 0; x < ancho; x++) {

            float suma = 0.0f;

            for (int dy = -1; dy <= 1; dy++) {

                for (int dx = -1; dx <= 1; dx++) {

                    int nx = x + dx;
                    int ny = y + dy;

                    // Extension de ceros fuera de la imagen
                    if (nx >= 0 && nx < ancho &&
                        ny >= 0 && ny < alto) {

                        suma += entrada[
                            static_cast<size_t>(ny) * ancho + nx
                        ] * coeficiente(filtro, dx, dy);
                    }
                }
            }

            salida[
                static_cast<size_t>(y) * ancho + x
            ] = suma;
        }
    }
}

// 5. CONVOLUCION PARALELA EN GPU CUDA
// Cada hilo procesa un pixel de salida.

__global__
void filtroGPU(const unsigned char* entrada,
               float* salida,
               int ancho,
               int alto,
               int filtro) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int total = ancho * alto;

    if (i >= total) {
        return;
    }

    int x = i % ancho;
    int y = i / ancho;

    float suma = 0.0f;

    for (int dy = -1; dy <= 1; dy++) {

        for (int dx = -1; dx <= 1; dx++) {

            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < ancho &&
                ny >= 0 && ny < alto) {

                suma += entrada[
                    static_cast<size_t>(ny) * ancho + nx
                ] * coeficiente(filtro, dx, dy);
            }
        }
    }

    salida[i] = suma;
}
// 6. CONVERTIR RESULTADOS A PIXELES PNG

vector<unsigned char> visualizar(
    const vector<float>& datos,
    int filtro
) {

    vector<unsigned char> salida(datos.size());

    for (size_t i = 0; i < datos.size(); i++) {

        float valor = datos[i];

        // Desplazamiento visual del Laplaciano
        if (filtro == 2) {
            valor += 128.0f;
        }

        valor = max(0.0f, min(255.0f, valor));

        salida[i] = static_cast<unsigned char>(
            roundf(valor)
        );
    }

    return salida;
}

// 7. PROGRAMA PRINCIPAL

int main() {

    const string rutaImagen =
        "imagenes/imagen_2000.pgm";

    // Crear carpeta si no existe
    filesystem::create_directories("resultados");

    vector<unsigned char> imagen;

    int ancho = 0;
    int alto = 0;

    if (!leerPGM(rutaImagen, imagen, ancho, alto)) {
        return 1;
    }

    cout << "====================================" << endl;
    cout << " PROCESAMIENTO DE IMAGENES CON CUDA " << endl;
    cout << "====================================" << endl;

    cout << "Imagen: " << ancho << " x " << alto << endl;

    cudaDeviceProp propiedades;
    CUDA_CHECK(cudaGetDeviceProperties(&propiedades, 0));

    cout << "GPU: " << propiedades.name << endl;

    size_t total = imagen.size();

    size_t bytesEntrada =
        total * sizeof(unsigned char);

    size_t bytesSalida =
        total * sizeof(float);

    unsigned char* d_entrada = nullptr;
    float* d_salida = nullptr;

    CUDA_CHECK(cudaMalloc(
        (void**)&d_entrada,
        bytesEntrada
    ));

    CUDA_CHECK(cudaMalloc(
        (void**)&d_salida,
        bytesSalida
    ));

    CUDA_CHECK(cudaMemcpy(
        d_entrada,
        imagen.data(),
        bytesEntrada,
        cudaMemcpyHostToDevice
    ));

    const char* nombres[] = {
        "Media_3x3",
        "Sharpen_3x3",
        "Laplaciano_3x3"
    };

    const double tolerancia = 1e-4;

    for (int filtro = 0; filtro < 3; filtro++) {

        vector<float> cpu(total);
        vector<float> gpu(total);

        // Procesamiento CPU
        filtroCPU(
            imagen.data(),
            cpu.data(),
            ancho,
            alto,
            filtro
        );

        // Procesamiento GPU
        int hilos = 256;

        int bloques =
            (static_cast<int>(total) + hilos - 1) / hilos;

        filtroGPU<<<bloques, hilos>>>(
            d_entrada,
            d_salida,
            ancho,
            alto,
            filtro
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            gpu.data(),
            d_salida,
            bytesSalida,
            cudaMemcpyDeviceToHost
        ));

        // Comparacion numerica CPU vs GPU
        double errorMax = 0.0;
        double errorSuma = 0.0;

        for (size_t i = 0; i < total; i++) {

            double error = fabs(
                static_cast<double>(cpu[i]) -
                static_cast<double>(gpu[i])
            );

            errorMax = max(errorMax, error);
            errorSuma += error;
        }

        double errorMedio = errorSuma / total;

        cout << "\n------------------------------------" << endl;
        cout << "Filtro: " << nombres[filtro] << endl;
        cout << "Error maximo: " << errorMax << endl;
        cout << "Error medio: " << errorMedio << endl;

        if (errorMax <= tolerancia) {
            cout << "Validacion CPU/GPU: CORRECTA" << endl;
        } else {
            cerr << "Validacion CPU/GPU: ERROR" << endl;
            CUDA_CHECK(cudaFree(d_entrada));
            CUDA_CHECK(cudaFree(d_salida));
            return 1;
        }

        // Guardar resultados directamente en PNG
        string rutaCPU =
            string("resultados/") +
            nombres[filtro] + "_CPU.png";

        string rutaGPU =
            string("resultados/") +
            nombres[filtro] + "_GPU.png";

        bool guardadoCPU = guardarPNG(
            rutaCPU,
            visualizar(cpu, filtro),
            ancho,
            alto
        );

        bool guardadoGPU = guardarPNG(
            rutaGPU,
            visualizar(gpu, filtro),
            ancho,
            alto
        );

        if (!guardadoCPU || !guardadoGPU) {

            cerr << "ERROR al guardar PNG." << endl;

            CUDA_CHECK(cudaFree(d_entrada));
            CUDA_CHECK(cudaFree(d_salida));

            return 1;
        }

        cout << "Guardado CPU: " << rutaCPU << endl;
        cout << "Guardado GPU: " << rutaGPU << endl;
    }

    CUDA_CHECK(cudaFree(d_entrada));
    CUDA_CHECK(cudaFree(d_salida));

    cout << "\n====================================" << endl;
    cout << " PROCESAMIENTO FINALIZADO " << endl;
    cout << "====================================" << endl;

    return 0;
}

