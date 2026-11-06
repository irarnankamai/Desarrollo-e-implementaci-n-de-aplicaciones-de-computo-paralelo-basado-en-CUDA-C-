
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <cstdint>

using namespace std;
namespace fs = std::filesystem;

// CONFIGURACION

const string IMAGEN_ORIGINAL = "imagenes/6x6orig.jpg";

const vector<int> RESOLUCIONES = {
    2000,
    4000,
    6000
};

// CONVERSION DE UN PIXEL RGB A GRIS

double obtenerGris(
    const unsigned char* imagen,
    int ancho,
    int x,
    int y
) {
    size_t indice =
        (static_cast<size_t>(y) * ancho + x) * 3;

    double r = imagen[indice];
    double g = imagen[indice + 1];
    double b = imagen[indice + 2];

    return 0.299 * r + 0.587 * g + 0.114 * b;
}
// REDIMENSIONAMIENTO BILINEAL

bool generarPGM(
    const unsigned char* imagen,
    int ancho,
    int alto,
    int inicioX,
    int inicioY,
    int lado,
    int tamano
) {
    if (!imagen || tamano <= 0 || lado <= 0) {
        cerr << "ERROR: parametros invalidos.\n";
        return false;
    }

    size_t cantidad =
        static_cast<size_t>(tamano) * tamano;

    vector<unsigned char> resultado(cantidad);

    // Las tres resoluciones usan exactamente
    // el mismo recorte de la imagen original.

    for (int y = 0; y < tamano; ++y) {

        double sy = inicioY +
            ((y + 0.5) * lado / tamano) - 0.5;

        sy = clamp(sy, 0.0, double(alto - 1));

        int y0 = static_cast<int>(floor(sy));
        int y1 = min(y0 + 1, alto - 1);

        double fy = sy - y0;

        for (int x = 0; x < tamano; ++x) {

            double sx = inicioX +
                ((x + 0.5) * lado / tamano) - 0.5;

            sx = clamp(sx, 0.0, double(ancho - 1));

            int x0 = static_cast<int>(floor(sx));
            int x1 = min(x0 + 1, ancho - 1);

            double fx = sx - x0;

            double p00 = obtenerGris(
                imagen, ancho, x0, y0
            );

            double p10 = obtenerGris(
                imagen, ancho, x1, y0
            );

            double p01 = obtenerGris(
                imagen, ancho, x0, y1
            );

            double p11 = obtenerGris(
                imagen, ancho, x1, y1
            );

            double gris =
                (1.0 - fx) * (1.0 - fy) * p00 +
                fx * (1.0 - fy) * p10 +
                (1.0 - fx) * fy * p01 +
                fx * fy * p11;

            size_t indice =
                static_cast<size_t>(y) * tamano + x;

            resultado[indice] =
                static_cast<unsigned char>(
                    clamp(round(gris), 0.0, 255.0)
                );
        }
    }

    string ruta =
        "imagenes/imagen_" +
        to_string(tamano) + ".pgm";

    ofstream archivo(ruta, ios::binary);

    if (!archivo) {
        cerr << "ERROR creando: " << ruta << "\n";
        return false;
    }

    archivo << "P5\n";
    archivo << tamano << " " << tamano << "\n";
    archivo << "255\n";

    archivo.write(
        reinterpret_cast<const char*>(resultado.data()),
        static_cast<streamsize>(resultado.size())
    );

    bool correcto = archivo.good();
    archivo.close();

    if (!correcto) {
        cerr << "ERROR escribiendo: " << ruta << "\n";
        return false;
    }

    cout << "GENERADA: " << ruta << "\n";
    cout << "Resolucion: "
         << tamano << " x " << tamano << "\n";
    cout << "Pixeles: " << cantidad << "\n\n";

    return true;
}

// PROGRAMA PRINCIPAL

int main() {

    cout << "====================================\n";
    cout << " PREPARACION DE IMAGENES PARA CUDA\n";
    cout << "====================================\n\n";

    fs::create_directories("imagenes");

    int ancho = 0;
    int alto = 0;
    int canales = 0;

    unsigned char* imagen = stbi_load(
        IMAGEN_ORIGINAL.c_str(),
        &ancho,
        &alto,
        &canales,
        3
    );

    if (!imagen) {
        cerr << "ERROR al leer: "
             << IMAGEN_ORIGINAL << "\n";
        cerr << "Detalle: "
             << stbi_failure_reason() << "\n";
        return 1;
    }

    cout << "Imagen original: "
         << IMAGEN_ORIGINAL << "\n";

    cout << "Dimensiones: "
         << ancho << " x " << alto << "\n";

    if (ancho <= 0 || alto <= 0) {
        cerr << "Dimensiones invalidas.\n";
        stbi_image_free(imagen);
        return 1;
    }

    // Recorte cuadrado central unico.
    int lado = min(ancho, alto);

    int inicioX = (ancho - lado) / 2;
    int inicioY = (alto - lado) / 2;

    cout << "Recorte cuadrado comun: "
         << lado << " x " << lado << "\n";

    cout << "Origen del recorte: ("
         << inicioX << ", "
         << inicioY << ")\n\n";

    bool todasCorrectas = true;

    for (int tamano : RESOLUCIONES) {

        cout << "------------------------------------\n";
        cout << "Generando "
             << tamano << " x " << tamano << "\n";

        bool correcto = generarPGM(
            imagen,
            ancho,
            alto,
            inicioX,
            inicioY,
            lado,
            tamano
        );

        if (!correcto) {
            todasCorrectas = false;
            break;
        }
    }

    stbi_image_free(imagen);

    cout << "====================================\n";

    if (todasCorrectas) {
        cout << "TODAS LAS IMAGENES ESTAN LISTAS.\n";
        cout << "Origen comun: "
             << IMAGEN_ORIGINAL << "\n";
        return 0;
    }

    cerr << "ERROR PREPARANDO IMAGENES.\n";
    return 1;
}
