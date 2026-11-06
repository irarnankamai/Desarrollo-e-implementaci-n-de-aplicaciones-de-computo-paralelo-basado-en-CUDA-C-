
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <numeric>
#include <cstdlib>
#include <limits>

using namespace std;
namespace fs = std::filesystem;

// CONFIGURACION

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        cerr << "CUDA ERROR: " \
             << cudaGetErrorString(error) \
             << " linea " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int REPETICIONES = 5;
constexpr double TOLERANCIA = 1e-4;

struct Prueba {
    int resolucion;
    int filtro;
    int mascara;
    int hilos;
};

struct Medicion {
    double cpu;
    double h2d;
    double kernel;
    double d2h;
    double total;
};

// 1. LEER IMAGEN PGM BINARIA

bool leerTokenPGM(istream& archivo, string& token) {
    token.clear();
    char c;

    while (archivo.get(c)) {
        if (c == '#') {
            archivo.ignore(
                numeric_limits<streamsize>::max(),
                '\n'
            );
            continue;
        }

        if (isspace(static_cast<unsigned char>(c))) {
            continue;
        }

        token += c;
        break;
    }

    if (token.empty()) return false;

    while (archivo.get(c)) {
        if (isspace(static_cast<unsigned char>(c))) {
            break;
        }

        if (c == '#') {
            archivo.ignore(
                numeric_limits<streamsize>::max(),
                '\n'
            );
            break;
        }

        token += c;
    }

    return true;
}

bool leerPGM(
    const string& ruta,
    vector<unsigned char>& imagen,
    int& ancho,
    int& alto
) {
    ifstream archivo(ruta, ios::binary);

    if (!archivo) return false;

    string formato, w, h, maximo;

    if (!leerTokenPGM(archivo, formato) ||
        !leerTokenPGM(archivo, w) ||
        !leerTokenPGM(archivo, h) ||
        !leerTokenPGM(archivo, maximo)) {
        return false;
    }

    try {
        ancho = stoi(w);
        alto = stoi(h);

        if (formato != "P5" ||
            stoi(maximo) != 255 ||
            ancho <= 0 ||
            alto <= 0) {
            return false;
        }
    }
    catch (...) {
        return false;
    }

    size_t n =
        static_cast<size_t>(ancho) * alto;

    imagen.resize(n);

    archivo.read(
        reinterpret_cast<char*>(imagen.data()),
        static_cast<streamsize>(n)
    );

    return archivo.gcount() ==
           static_cast<streamsize>(n);
}

// 2. CALCULO OPTIMIZADO CON ENTEROS

// 0 = Media
// 1 = Sharpen
// 2 = Laplaciano
//
// Se acumula en int y se convierte a float
// una sola vez al terminar cada pixel.

__host__ __device__
inline float calcularPixelOptimizado(
    const unsigned char* entrada,
    int x,
    int y,
    int ancho,
    int alto,
    int filtro,
    int k
) {
    int suma = 0;

    // Media: suma entera exacta de los pixeles
    if (filtro == 0) {
        int radio = k / 2;

        for (int dy = -radio; dy <= radio; ++dy) {
            int ny = y + dy;

            if (ny < 0 || ny >= alto) continue;

            for (int dx = -radio; dx <= radio; ++dx) {
                int nx = x + dx;

                if (nx < 0 || nx >= ancho) continue;

                size_t indice =
                    static_cast<size_t>(ny) * ancho + nx;

                suma += static_cast<int>(
                    entrada[indice]
                );
            }
        }

        // Division unica con precision double.
        // Conserva el redondeo final a float.
        return static_cast<float>(
            static_cast<double>(suma) /
            static_cast<double>(k * k)
        );
    }

    // Sharpen y Laplaciano:
    // solo 5 posiciones tienen coeficiente no cero.

    size_t centro =
        static_cast<size_t>(y) * ancho + x;

    int valorCentro =
        static_cast<int>(entrada[centro]);

    int vecinos = 0;

    if (x > 0) {
        vecinos += entrada[centro - 1];
    }

    if (x + 1 < ancho) {
        vecinos += entrada[centro + 1];
    }

    if (y > 0) {
        vecinos += entrada[centro - ancho];
    }

    if (y + 1 < alto) {
        vecinos += entrada[centro + ancho];
    }

    if (filtro == 1) {
        suma = 5 * valorCentro - vecinos;
    }
    else {
        suma = vecinos - 4 * valorCentro;
    }

    return static_cast<float>(suma);
}

// 3. CONVOLUCION CPU

void convolucionCPU(
    const unsigned char* entrada,
    float* salida,
    int ancho,
    int alto,
    int filtro,
    int k
) {
    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {

            size_t indice =
                static_cast<size_t>(y) * ancho + x;

            salida[indice] = calcularPixelOptimizado(
                entrada,
                x,
                y,
                ancho,
                alto,
                filtro,
                k
            );
        }
    }
}

// 4. CONVOLUCION GPU CUDA

__global__ void convolucionGPU(
    const unsigned char* entrada,
    float* salida,
    int ancho,
    int alto,
    int filtro,
    int k
) {
    size_t i =
        static_cast<size_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;

    size_t total =
        static_cast<size_t>(ancho) * alto;

    if (i >= total) return;

    int x = static_cast<int>(i % ancho);
    int y = static_cast<int>(i / ancho);

    salida[i] = calcularPixelOptimizado(
        entrada,
        x,
        y,
        ancho,
        alto,
        filtro,
        k
    );
}

// 5. FUNCIONES AUXILIARES

string nombreFiltro(int filtro) {
    if (filtro == 0) return "Media";
    if (filtro == 1) return "Sharpen";
    return "Laplaciano";
}

double mediana(vector<double> valores) {
    if (valores.empty()) return 0.0;

    sort(valores.begin(), valores.end());

    size_t n = valores.size();

    if (n % 2 != 0) {
        return valores[n / 2];
    }

    return (
        valores[n / 2 - 1] +
        valores[n / 2]
    ) / 2.0;
}

double desviacion(const vector<double>& valores) {
    if (valores.empty()) return 0.0;

    double promedio = accumulate(
        valores.begin(),
        valores.end(),
        0.0
    ) / valores.size();

    double suma = 0.0;

    for (double valor : valores) {
        double diferencia = valor - promedio;
        suma += diferencia * diferencia;
    }

    return sqrt(suma / valores.size());
}

double medirEvento(
    cudaEvent_t inicio,
    cudaEvent_t fin
) {
    float tiempo = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &tiempo,
        inicio,
        fin
    ));

    return static_cast<double>(tiempo);
}

// 6. PROGRAMA PRINCIPAL

int main() {

    fs::create_directories("resultados");

    int cantidadGPU = 0;

    CUDA_CHECK(cudaGetDeviceCount(&cantidadGPU));

    if (cantidadGPU == 0) {
        cerr << "No se detecto GPU CUDA.\n";
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp gpuInfo{};

    CUDA_CHECK(cudaGetDeviceProperties(
        &gpuInfo,
        0
    ));

    cout << "====================================\n";
    cout << " EXPERIMENTOS CUDA OPTIMIZADOS\n";
    cout << "====================================\n";

    cout << "GPU: " << gpuInfo.name << "\n";
    cout << "Repeticiones: " << REPETICIONES << "\n";
    cout << "Precision: acumulacion entera\n";

    // CONFIGURACIONES DE EXPERIMENTOS

    vector<Prueba> pruebas = {

        {4000, 0, 3, 256},
        {4000, 1, 3, 256},
        {4000, 2, 3, 256},

        {2000, 0, 9, 256},
        {4000, 0, 9, 256},
        {6000, 0, 9, 256},

        {4000, 0, 21, 256},

        {4000, 0, 9, 128},
        {4000, 0, 9, 512}
    };

    ofstream csv(
        "resultados/mediciones_optimizadas.csv"
    );

    ofstream resumen(
        "resultados/resumen_optimizado.csv"
    );

    if (!csv || !resumen) {
        cerr << "Error creando archivos CSV.\n";
        return 1;
    }

    csv
        << "resolucion,filtro,mascara,hilos,repeticion,"
        << "cpu_ms,h2d_ms,kernel_ms,d2h_ms,total_gpu_ms\n";

    resumen
        << "resolucion,filtro,mascara,hilos,"
        << "cpu_mediana_ms,cpu_desv_ms,"
        << "h2d_mediana_ms,kernel_mediana_ms,"
        << "kernel_desv_ms,d2h_mediana_ms,"
        << "gpu_total_mediana_ms,gpu_total_desv_ms,"
        << "speedup_kernel,speedup_total,"
        << "error_maximo,error_medio,validacion\n";

    csv << fixed << setprecision(6);
    resumen << fixed << setprecision(6);

    int pruebasCorrectas = 0;
    int pruebasConDiferencias = 0;

    // EJECUTAR PRUEBAS

    for (size_t p = 0; p < pruebas.size(); ++p) {

        const Prueba& t = pruebas[p];

        cout << "\n====================================\n";
        cout << "PRUEBA " << p + 1
             << " DE " << pruebas.size() << "\n";

        cout << t.resolucion
             << "x" << t.resolucion
             << " | " << nombreFiltro(t.filtro)
             << " " << t.mascara
             << "x" << t.mascara
             << " | " << t.hilos
             << " hilos\n";

        if (t.hilos > gpuInfo.maxThreadsPerBlock) {
            cerr << "Cantidad de hilos no admitida.\n";
            return 1;
        }

        string ruta =
            "imagenes/imagen_" +
            to_string(t.resolucion) +
            ".pgm";

        vector<unsigned char> imagen;

        int ancho = 0;
        int alto = 0;

        if (!leerPGM(
            ruta,
            imagen,
            ancho,
            alto
        )) {
            cerr << "No se pudo leer: "
                 << ruta << "\n";
            return 1;
        }

        if (ancho != t.resolucion ||
            alto != t.resolucion) {
            cerr << "Resolucion incorrecta.\n";
            return 1;
        }

        size_t n = imagen.size();

        size_t bytesEntrada =
            n * sizeof(unsigned char);

        size_t bytesSalida =
            n * sizeof(float);

        vector<float> salidaCPU(n);
        vector<float> salidaGPU(n);

        unsigned char* d_entrada = nullptr;
        float* d_salida = nullptr;

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_entrada),
            bytesEntrada
        ));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_salida),
            bytesSalida
        ));

        int bloques = static_cast<int>(
            (n + t.hilos - 1) / t.hilos
        );

        cudaEvent_t inicio;
        cudaEvent_t fin;

        CUDA_CHECK(cudaEventCreate(&inicio));
        CUDA_CHECK(cudaEventCreate(&fin));

        // CALENTAMIENTO CPU

        convolucionCPU(
            imagen.data(),
            salidaCPU.data(),
            ancho,
            alto,
            t.filtro,
            t.mascara
        );

        // CALENTAMIENTO GPU

        CUDA_CHECK(cudaMemcpy(
            d_entrada,
            imagen.data(),
            bytesEntrada,
            cudaMemcpyHostToDevice
        ));

        convolucionGPU<<<bloques, t.hilos>>>(
            d_entrada,
            d_salida,
            ancho,
            alto,
            t.filtro,
            t.mascara
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            salidaGPU.data(),
            d_salida,
            bytesSalida,
            cudaMemcpyDeviceToHost
        ));

        // VALIDACION NUMERICA

        double errorMax = 0.0;
        double errorSuma = 0.0;
        bool finitos = true;

        for (size_t i = 0; i < n; ++i) {

            if (!isfinite(salidaCPU[i]) ||
                !isfinite(salidaGPU[i])) {
                finitos = false;
                break;
            }

            double error = fabs(
                static_cast<double>(salidaCPU[i]) -
                static_cast<double>(salidaGPU[i])
            );

            errorMax = max(errorMax, error);
            errorSuma += error;
        }

        if (!finitos) {
            errorMax =
                numeric_limits<double>::infinity();
        }

        double errorMedio = finitos
            ? errorSuma / n
            : numeric_limits<double>::infinity();

        bool valido =
            finitos && errorMax <= TOLERANCIA;

        cout << scientific << setprecision(10);

        cout << "Error maximo: "
             << errorMax << "\n";

        cout << "Error medio: "
             << errorMedio << "\n";

        cout << "Tolerancia: "
             << TOLERANCIA << "\n";

        cout << fixed << setprecision(3);

        cout << "Validacion: "
             << (valido ? "CORRECTA" : "ERROR")
             << "\n";

        if (valido) {
            pruebasCorrectas++;
        }
        else {
            pruebasConDiferencias++;

            cout << "ADVERTENCIA: diferencia superior "
                 << "a la tolerancia.\n";
        }

        // CINCO REPETICIONES

        vector<Medicion> datos;

        for (int r = 1; r <= REPETICIONES; ++r) {

            Medicion m{};

            // TIEMPO CPU

            auto cpuInicio =
                chrono::steady_clock::now();

            convolucionCPU(
                imagen.data(),
                salidaCPU.data(),
                ancho,
                alto,
                t.filtro,
                t.mascara
            );

            auto cpuFin =
                chrono::steady_clock::now();

            m.cpu =
                chrono::duration<double, milli>(
                    cpuFin - cpuInicio
                ).count();

            // TRANSFERENCIA CPU -> GPU

            CUDA_CHECK(cudaEventRecord(inicio));

            CUDA_CHECK(cudaMemcpy(
                d_entrada,
                imagen.data(),
                bytesEntrada,
                cudaMemcpyHostToDevice
            ));

            CUDA_CHECK(cudaEventRecord(fin));
            CUDA_CHECK(cudaEventSynchronize(fin));

            m.h2d = medirEvento(inicio, fin);

            // TIEMPO KERNEL

            CUDA_CHECK(cudaEventRecord(inicio));

            convolucionGPU<<<bloques, t.hilos>>>(
                d_entrada,
                d_salida,
                ancho,
                alto,
                t.filtro,
                t.mascara
            );

            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventRecord(fin));
            CUDA_CHECK(cudaEventSynchronize(fin));

            m.kernel = medirEvento(inicio, fin);

            // TRANSFERENCIA GPU -> CPU

            CUDA_CHECK(cudaEventRecord(inicio));

            CUDA_CHECK(cudaMemcpy(
                salidaGPU.data(),
                d_salida,
                bytesSalida,
                cudaMemcpyDeviceToHost
            ));

            CUDA_CHECK(cudaEventRecord(fin));
            CUDA_CHECK(cudaEventSynchronize(fin));

            m.d2h = medirEvento(inicio, fin);

            // Total de etapas medidas
            m.total = m.h2d + m.kernel + m.d2h;

            datos.push_back(m);

            csv
                << t.resolucion << ","
                << nombreFiltro(t.filtro) << ","
                << t.mascara << ","
                << t.hilos << ","
                << r << ","
                << m.cpu << ","
                << m.h2d << ","
                << m.kernel << ","
                << m.d2h << ","
                << m.total << "\n";

            csv.flush();

            cout << "Repeticion " << r
                 << " | CPU: " << m.cpu
                 << " ms | Kernel: " << m.kernel
                 << " ms | GPU total: " << m.total
                 << " ms\n";
        }

        // ESTADISTICAS

        vector<double> tiemposCPU;
        vector<double> tiemposH2D;
        vector<double> tiemposKernel;
        vector<double> tiemposD2H;
        vector<double> tiemposTotal;

        for (const auto& m : datos) {
            tiemposCPU.push_back(m.cpu);
            tiemposH2D.push_back(m.h2d);
            tiemposKernel.push_back(m.kernel);
            tiemposD2H.push_back(m.d2h);
            tiemposTotal.push_back(m.total);
        }

        double cpuMediana =
            mediana(tiemposCPU);

        double kernelMediana =
            mediana(tiemposKernel);

        double gpuTotalMediana =
            mediana(tiemposTotal);

        double speedupKernel =
            kernelMediana > 0.0
            ? cpuMediana / kernelMediana
            : 0.0;

        double speedupTotal =
            gpuTotalMediana > 0.0
            ? cpuMediana / gpuTotalMediana
            : 0.0;

        // GUARDAR RESUMEN

        resumen
            << t.resolucion << ","
            << nombreFiltro(t.filtro) << ","
            << t.mascara << ","
            << t.hilos << ","
            << cpuMediana << ","
            << desviacion(tiemposCPU) << ","
            << mediana(tiemposH2D) << ","
            << kernelMediana << ","
            << desviacion(tiemposKernel) << ","
            << mediana(tiemposD2H) << ","
            << gpuTotalMediana << ","
            << desviacion(tiemposTotal) << ","
            << speedupKernel << ","
            << speedupTotal << ",";

        resumen
            << scientific << setprecision(10)
            << errorMax << ","
            << errorMedio << ","
            << (valido ? "CORRECTA" : "ERROR")
            << "\n";

        resumen << fixed << setprecision(6);
        resumen.flush();

        cout << "\nRESULTADOS ESTADISTICOS\n";

        cout << "Mediana CPU: "
             << cpuMediana << " ms\n";

        cout << "Mediana kernel: "
             << kernelMediana << " ms\n";

        cout << "Mediana GPU total: "
             << gpuTotalMediana << " ms\n";

        cout << "Speedup kernel: "
             << speedupKernel << "x\n";

        cout << "Speedup total: "
             << speedupTotal << "x\n";

        // LIBERAR RECURSOS

        CUDA_CHECK(cudaEventDestroy(inicio));
        CUDA_CHECK(cudaEventDestroy(fin));

        CUDA_CHECK(cudaFree(d_entrada));
        CUDA_CHECK(cudaFree(d_salida));
    }

    // RESUMEN FINAL

    cout << "\n====================================\n";
    cout << " EXPERIMENTOS FINALIZADOS\n";
    cout << "====================================\n";

    cout << "Pruebas ejecutadas: "
         << pruebas.size() << "\n";

    cout << "Validaciones correctas: "
         << pruebasCorrectas << "\n";

    cout << "Validaciones con diferencias: "
         << pruebasConDiferencias << "\n";

    cout << "\nArchivos generados:\n";
    cout << "resultados/mediciones_optimizadas.csv\n";
    cout << "resultados/resumen_optimizado.csv\n";

    return pruebasConDiferencias == 0 ? 0 : 2;
}
