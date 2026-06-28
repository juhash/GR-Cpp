#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

// Unidades geométricas: G = c = 1, e fixamos M = 1.
const double M = 1.0;

// O momento angular L conserva-se -> é uma constante, não evolui.
double L = 0.0;

// Partícula no plano da órbita.
struct State {
    double r;     // distância radial r
    double vr;    // velocidade radial  dr/dtau
    double phi;   // ângulo
};

// Dado um estado, devolve as suas derivadas em ordem a tau.
// É aqui que vive a física.
State derivatives(State s) {
    double r = s.r;
    State d;
    d.r   = s.vr;                                                  // dr/dtau
    d.vr  = -M/(r*r) + (L*L)/(r*r*r) - 3.0*M*(L*L)/(r*r*r*r);      // a equação da órbita
    d.phi = L/(r*r);                                              // dphi/dtau
    return d;
}

// RK4 em C++ ! - super importante
State rk4Step(State s, double h) {
    State k1 = derivatives(s);
    State s2 = { s.r + 0.5*h*k1.r, s.vr + 0.5*h*k1.vr, s.phi + 0.5*h*k1.phi };

    State k2 = derivatives(s2);
    State s3 = { s.r + 0.5*h*k2.r, s.vr + 0.5*h*k2.vr, s.phi + 0.5*h*k2.phi };

    State k3 = derivatives(s3);
    State s4 = { s.r + h*k3.r, s.vr + h*k3.vr, s.phi + h*k3.phi };

    State k4 = derivatives(s4);

    State out;
    out.r   = s.r   + (h/6.0)*(k1.r   + 2*k2.r   + 2*k3.r   + k4.r);
    out.vr  = s.vr  + (h/6.0)*(k1.vr  + 2*k2.vr  + 2*k3.vr  + k4.vr);
    out.phi = s.phi + (h/6.0)*(k1.phi + 2*k2.phi + 2*k3.phi + k4.phi);
    return out;
}


// printing E^2 (energia ao quadrado) para confirmar que tudo corre bem
double energySquared(State s) {
    double r = s.r;
    double Veff = (1.0 - 2.0*M/r) * (1.0 + (L*L)/(r*r));
    return s.vr*s.vr + Veff;
}

int main() {
    // Condições iniciais: órbita circular a r0
    double r0 = 10.0;
    L = std::sqrt(M * r0 * r0 / (r0 - 3.0*M));   // L p/ órbita circular

    State s = { r0, 0.0, 0.0 };   // começa em r0, sem velocidade radial, ângulo 0

    double h = 0.01;              // timestep on tau
    int passos = 50000;

    // ficheiro output
    std::ofstream file("orbita.csv");
    file << std::setprecision(10);
    file << "tau,r,phi,x,y,E2\n";

    for (int i = 0; i < passos; i++) {
        double tau = i * h;

        // converter (r, phi) para coordenadas cartesianas p plot
        double x = s.r * std::cos(s.phi);
        double y = s.r * std::sin(s.phi);

        // guardar 1 em cada 10 pontos, para o ficheiro não ficar gigante
        if (i % 10 == 0) {
            file << tau << "," << s.r << "," << s.phi << ","
                 << x << "," << y << "," << energySquared(s) << "\n";
        }

        s = rk4Step(s, h);   // avança um passo

        if (s.r <= 2.0 * M) {   // cruzou o horizonte -> paramos
            std::cout << "A particula caiu no horizonte em tau = " << tau << "\n";
            break;
        }
    }

    file.close();
    std::cout << "Pronto. Trajetoria escrita em orbita.csv\n";
    std::cout << "ISCO em r = " << 6.0*M << "  |  Horizonte em r = " << 2.0*M << "\n";
    return 0;
}