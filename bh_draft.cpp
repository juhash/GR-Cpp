#include <GL/glew.h>      // tem de vir ANTES do glfw
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>
#include <cmath>
using glm::vec3;
using glm::vec2;

//constantes
const double G = 1.0;
const double c = 1.0;

// Escala unica: 1 unidade de r em unidades geometrizadas (G=c=1) equivale a
// kWorldScale unidades NDC (-1 a 1). Usada tanto para calcular o tamanho do BH
// no ecra como para posicionar a particula que vai cair - e a ligacao entre a
// fisica e o ecra que faltava antes (so o tamanho do BH tinha escala; a posicao
// dos raios era arbitraria em NDC, desligada de r_s). Com mass=1.0 -> r_s=2.0,
// kWorldScale=0.05 da um raio em NDC de 0.1 (~ igual ao tamanho visual anterior).
const double kWorldScale = 0.05;

// OS SHADERS (em GLSL, como texto)

// Shader dos pontos: reutilizado tanto para o buraco negro como para o fotao.
// Antes de cada draw call mudamos uColor e uPointSize consoante o que estamos a desenhar.
const char* pointVertexSrc = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
uniform float uPointSize;   // deixou de estar fixo (60.0) no shader, agora vem de fora
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    gl_PointSize = uPointSize;
}
)";

const char* pointFragmentSrc = R"(
#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main() {
    // gl_PointCoord vai de (0,0) a (1,1) dentro do ponto; o centro e (0.5,0.5)
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;   // fora do raio 0.5 -> deita o pixel fora
    FragColor = vec4(uColor, 1.0);
}
)";

// Shader das linhas: usado agora so para o rasto do fotao.
// Cada vertice tem a sua propria transparencia (aAlpha), para o rasto ir desaparecendo
// da ponta mais antiga para a mais recente - o GPU interpola isto ao longo da linha.
const char* lineVertexSrc = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in float aAlpha;
out float vAlpha;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vAlpha = aAlpha;
}
)";

const char* lineFragmentSrc = R"(
#version 330 core
in float vAlpha;
out vec4 FragColor;
uniform vec3 uColor;
void main() { FragColor = vec4(uColor, vAlpha); }
)";

// funcao auxiliar: compila um shader e avisa se falhar
GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    int ok; glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[512]; glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "Erro shader:\n" << log << "\n"; }
    return shader;
}

// funcao auxiliar: liga um programa e avisa se falhar
// (faltava esta verificacao - um shader pode compilar sozinho e o link falhar na mesma)
GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    int ok; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) { char log[512]; glGetProgramInfoLog(prog, 512, nullptr, log);
        std::cerr << "Erro a ligar programa:\n" << log << "\n"; }
    return prog;
}

struct BlackHole {
    vec3   position;
    float  mass;
    double r_s, r_photon, r_isco;

    BlackHole(vec3 pos, float m) : position(pos), mass(m) {
        r_s      = 2.0 * G * mass / (c * c);
        r_photon = 1.5 * r_s;
        r_isco   = 3.0 * r_s;
    }

    void draw(GLuint /*prog*/, GLuint vao, GLint uColorLoc, GLint uSizeLoc, float displayRadiusPx) {
        glUniform3f(uColorLoc, 1.0f, 0.0f, 0.0f);   // vermelho
        glUniform1f(uSizeLoc, displayRadiusPx);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, 1);              // UM ponto, arredondado no shader
    }
};

struct Ray {
    vec3   posCart;
    double r, theta, phi;
    double dr, dtheta, dphi;
    double E, L;
};

struct Particle {
    vec3   posCart;
    double r, theta, phi;
    double dr, dtheta, dphi;
    double E, L;
    bool   active = true;
};

// Agrupa tudo o que pertence a UMA particula desenhada no ecra: o estado fisico
// (Particle), o seu proprio historico de posicoes (trail) e os seus proprios
// buffers GPU. Sem isto, teriamos de multiplicar cada variavel por 5 a mao -
// com isto, e so' criar um std::vector<ParticleVisual> do tamanho que quisermos.
struct ParticleVisual {
    Particle particle;
    double phi0;        // angulo inicial fixo desta particula (define a sua "linha" radial)
    vec3   color;
    std::vector<vec2>  trail;
    std::vector<float> trailData;
    float  respawnTimer = 0.0f;
    GLuint vao = 0, vbo = 0;
    GLuint trailVAO = 0, trailVBO = 0;
};

// --- FISICA: geodesica temporal (partial massiva) na metrica de Schwarzschild ---
//
// (dr/dtau)^2 = E^2 - (1 - r_s/r)(1 + L^2/r^2)
//
// E, L sao as quantidades conservadas por unidade de massa (energia e momento angular).
// tau e o tempo PROPRIO da particula (nao o tempo coordenado t medido a distancia) -
// e o que torna isto bem comportado numericamente ate' ao horizonte: dr/dtau e finito
// em r=r_s (ao contrario de dr/dt, que tende para 0 ai - por isso um observador distante
// nunca "ve" nada cruzar o horizonte, so cada vez mais lento). Aqui simulamos o que a
// PARTICULA sente, nao o que um observador distante veria.
//
// Com L=0 isto reduz-se a queda puramente radial: dr/dtau = -sqrt(E^2 - (1 - r_s/r)).
double geodesicDrDTau(double r, double r_s, double E, double L) {
    double term = E*E - (1.0 - r_s/r) * (1.0 + (L*L)/(r*r));
    if (term < 0.0) term = 0.0;   // protecao numerica; fisicamente nao deveria ocorrer para r > r_s
    return -std::sqrt(term);      // negativo: r diminui, a particula cai para dentro
}

// um passo de Runge-Kutta 4a ordem para integrar r(tau). RK4 em vez de Euler porque
// e' consideravelmente mais preciso para o mesmo passo de tempo, e vamos precisar
// da mesma funcao para geodesicas com L != 0 mais tarde (orbitas, lente gravitacional).
double rk4StepR(double r, double dtau, double r_s, double E, double L) {
    double k1 = geodesicDrDTau(r,                r_s, E, L);
    double k2 = geodesicDrDTau(r + 0.5*dtau*k1,  r_s, E, L);
    double k3 = geodesicDrDTau(r + 0.5*dtau*k2,  r_s, E, L);
    double k4 = geodesicDrDTau(r + dtau*k3,      r_s, E, L);
    return r + (dtau / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
}

// Constroi uma faixa (ribbon) que afunila da cabeca (ponto mais recente, largura
// headHalfWidthNDC) ate a cauda (ponto mais antigo, largura zero) - como o rasto de
// um cometa, em vez de uma linha de largura constante. Para cada ponto central do
// rasto, gera DOIS vertices (esquerda/direita), deslocados na perpendicular a
// direcao local do movimento. O resultado destina-se a GL_TRIANGLE_STRIP (nao
// GL_LINE_STRIP): a largura por vertice so e' possivel com geometria real, ja que
// glLineWidth() nao e fiavel (nem por vertice) em core profile.
void buildTaperedTrail(const std::vector<vec2>& trail, float headHalfWidthNDC,
                        float maxAlpha, std::vector<float>& out) {
    out.clear();
    const size_t n = trail.size();
    if (n == 0) return;

    for (size_t i = 0; i < n; ++i) {
        vec2 dir;
        if (i + 1 < n)      dir = trail[i + 1] - trail[i];   // direcao para o proximo ponto
        else if (i > 0)     dir = trail[i] - trail[i - 1];   // ultimo ponto: usa o anterior
        else                dir = vec2(1.0f, 0.0f);          // so ha 1 ponto - arbitrario

        float len = glm::length(dir);
        vec2 perp = (len > 1e-6f) ? vec2(-dir.y, dir.x) / len : vec2(0.0f, 1.0f);

        float t = (n > 1) ? static_cast<float>(i) / static_cast<float>(n - 1) : 1.0f;
        float halfWidth = t * headHalfWidthNDC;   // zero na cauda, maximo na cabeca
        float alpha     = t * maxAlpha;

        vec2 left  = trail[i] + perp * halfWidth;
        vec2 right = trail[i] - perp * halfWidth;

        out.push_back(left.x);  out.push_back(left.y);  out.push_back(alpha);
        out.push_back(right.x); out.push_back(right.y); out.push_back(alpha);
    }
}

// MAIN!
int main() {
    if (!glfwInit()) { std::cerr << "Falhou glfwInit\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    const int winWidth = 800, winHeight = 600;
    GLFWwindow* window = glfwCreateWindow(winWidth, winHeight, "Buraco Negro", nullptr, nullptr);
    if (!window) { std::cerr << "Falhou janela\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "Falhou glewInit\n"; return -1; }
    std::cout << "OpenGL versao: " << glGetString(GL_VERSION) << "\n";
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);   // necessario para o rasto desaparecer com transparencia

    // viewport (importante em ecras HiDPI, onde o framebuffer pode ter mais pixeis que a janela)
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);

    // compilar e ligar os shaders
    // programa dos pontos (BH + fotao, partilhado)
    GLuint vs = compileShader(GL_VERTEX_SHADER, pointVertexSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, pointFragmentSrc);
    GLuint pointProg = linkProgram(vs, fs);
    glDeleteShader(vs); glDeleteShader(fs);
    GLint pointColorLoc = glGetUniformLocation(pointProg, "uColor");
    GLint pointSizeLoc  = glGetUniformLocation(pointProg, "uPointSize");

    // programa das linhas (rasto do fotao)
    GLuint lvs = compileShader(GL_VERTEX_SHADER, lineVertexSrc);
    GLuint lfs = compileShader(GL_FRAGMENT_SHADER, lineFragmentSrc);
    GLuint lineProg = linkProgram(lvs, lfs);
    glDeleteShader(lvs); glDeleteShader(lfs);
    GLint lineColorLoc = glGetUniformLocation(lineProg, "uColor");

    BlackHole bh(vec3(0, 0, 0), 1.0f);

    // tamanho do BH no ecra, calculado a partir de r_s (fisica) e nao de um numero
    // decorativo. raio em NDC = r_s * kWorldScale; como 2 unidades NDC (-1 a 1) cobrem
    // fbHeight pixels, 1 unidade NDC = fbHeight/2 pixels, logo o DIAMETRO em pixels
    // (e' o que gl_PointSize espera) e' raioNDC * fbHeight (o *2 do diametro cancela
    // com o /2 da conversao NDC->pixels)
    float bhRadiusPx = static_cast<float>(bh.r_s * kWorldScale * fbHeight);

    // tamanho (diametro, em pixeis) usado para o fotao e para a particula - a mesma
    // constante e reutilizada para a largura maxima do rasto (na cabeca), para que o
    // rasto comece exatamente do tamanho do ponto e nao fique dessincronizado se um
    // dia mudares este valor
    const float kDotSizePx = 6.0f;
    // largura (metade) da cabeca do rasto, em NDC: um pixel vertical = 2/fbHeight NDC,
    // logo um diametro em pixeis corresponde a um RAIO em NDC de diametroPx/fbHeight
    // (a mesma relacao usada acima para bhRadiusPx, so que ao contrario)
    const float headHalfWidthNDC = kDotSizePx / static_cast<float>(fbHeight);

    // VAO do buraco negro: um unico ponto, na posicao do bh (usa bh.position em vez de
    // um valor fixo (0,0) escrito a parte - antes o draw() ignorava position por completo)
    float bhVertex[] = { bh.position.x, bh.position.y };
    GLuint bhVAO, bhVBO;
    glGenVertexArrays(1, &bhVAO);
    glGenBuffers(1, &bhVBO);
    glBindVertexArray(bhVAO);
    glBindBuffer(GL_ARRAY_BUFFER, bhVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(bhVertex), bhVertex, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // --- FOTAO: um ponto que atravessa o ecra a meio, na horizontal, com rasto ---
    const float raySpeed  = 0.6f;    // unidades NDC por segundo
    const float rayStartX = -1.05f;  // comeca mesmo fora do ecra, a esquerda
    const float rayEndX   =  1.05f;  // sai mesmo fora do ecra, a direita
    const float rayY      =  0.0f;   // a meio do ecra - por agora fixo; e o que vai deixar
                                      // de ser constante quando ligares a lente gravitacional
    vec2 rayPos(rayStartX, rayY);

    const size_t maxTrailPoints = 4000;   // capacidade de seguranca do VBO, nao um limite ativo:
                                           // o rasto nao e cortado por contagem de pontos (isso
                                           // dependeria da taxa de refrescamento do ecra), so e
                                           // limpo quando o fotao reinicia a travessia
    std::vector<vec2> trail;
    trail.reserve(maxTrailPoints);

    // VAO/VBO do ponto do fotao (posicao atualizada todos os frames via glBufferSubData)
    GLuint rayVAO, rayVBO;
    glGenVertexArrays(1, &rayVAO);
    glGenBuffers(1, &rayVBO);
    glBindVertexArray(rayVAO);
    glBindBuffer(GL_ARRAY_BUFFER, rayVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vec2), &rayPos, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), (void*)0);
    glEnableVertexAttribArray(0);

    // VAO/VBO do rasto (GL_TRIANGLE_STRIP; cada ponto do rasto vira 2 vertices
    // (esquerda/direita), cada um com x, y, alpha -> 3 floats)
    GLuint trailVAO, trailVBO;
    glGenVertexArrays(1, &trailVAO);
    glGenBuffers(1, &trailVBO);
    glBindVertexArray(trailVAO);
    glBindBuffer(GL_ARRAY_BUFFER, trailVBO);
    glBufferData(GL_ARRAY_BUFFER, maxTrailPoints * 2 * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // buffer reutilizado todos os frames para montar o rasto (2 vertices por ponto),
    // para nao andar a alocar memoria nova a cada iteracao do loop
    std::vector<float> trailData;
    trailData.reserve(maxTrailPoints * 2 * 3);

    // --- PARTICULAS: 5 quedas radiais livres, paralelas, para dentro do buraco negro ---
    // caso mais simples de geodesica: L=0 (sem momento angular - cada uma cai em linha reta
    // ate ao centro, ao longo do seu proprio phi fixo), E=1 (largada em repouso no infinito).
    // r0 escolhido bem fora do horizonte e do r_isco, para dar tempo a ver a queda.

    //incrementos angulares!
    const double particleR0         = 12.0 * bh.r_s;      // fora do r_isco (3*r_s) e do r_photon (1.5*r_s)
    const double particlePhiCenter  = 3.0 * M_PI / 4.0;   // 135 graus - direcao da particula central
    const int    NUM_PARTICLES      = 200; 
    // separacao angular entre particulas vizinhas: pequena mas distinguivel. Com
    // r0*kWorldScale ~ 1.2 NDC de raio inicial, 0.05 rad da' uma separacao lateral de
    // ~0.06 NDC entre vizinhas no arranque - ajusta este valor se quiseres mais ou menos.
    const double PARTICLE_PHI_SPACING = 0.05;

    // A queda REAL (tau real) desde 12*r_s ate ao horizonte demora ~54s de tempo proprio
    // (confirmado numericamente) - fisicamente correto, mas impraticavel de ver. Este
    // multiplicador acelera so' a REPRODUCAO (like um time-lapse): a equacao integrada e'
    // exatamente a mesma, dtau e' so' maior a cada frame. Nao afeta a forma da trajetoria
    // nem o ponto onde desaparece, so' a velocidade a que a vemos acontecer. Como as 100
    // particulas partilham r0, E, L, caem todas ao MESMO ritmo radial e cruzam o horizonte
    // no mesmo instante - so' a posicao lateral (phi) as distingue.
    const double TAU_PLAYBACK_SPEED   = 15.0;
    const float  particleRespawnDelay = 1.2f;   // pausa (segundos) depois de desaparecer

    auto resetParticle = [&](Particle& p, double phi0) {
        p.r = particleR0;
        p.theta = M_PI / 2.0;   // plano equatorial (nao usado no render 2D, mas mantido coerente)
        p.phi = phi0;
        p.dr = 0.0; p.dtheta = 0.0; p.dphi = 0.0;
        p.E = 1.0; p.L = 0.0;
        p.active = true;
    };

    // cores da esquerda (indice 0) para a direita (indice 4); indice 2 = central,
    // com a cor azul clara original
    const vec3 particleColors[5] = {
        vec3(0.75f, 0.55f, 1.0f),   // violeta
        vec3(0.6f,  0.7f,  1.0f),   // azul-violeta
        vec3(0.5f,  0.8f,  1.0f),   // azul claro (central - cor original)
        vec3(0.4f,  0.9f,  0.9f),   // azul-ciano
        vec3(0.4f,  1.0f,  0.7f)    // ciano-esverdeado
    };

    std::vector<ParticleVisual> particles(NUM_PARTICLES);
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        int offset = i - NUM_PARTICLES / 2;   // -2, -1, 0, +1, +2 (0 = central)
        particles[i].phi0  = particlePhiCenter + offset * PARTICLE_PHI_SPACING;
        particles[i].color = particleColors[i];
        resetParticle(particles[i].particle, particles[i].phi0);
        particles[i].trail.reserve(maxTrailPoints);
        particles[i].trailData.reserve(maxTrailPoints * 2 * 3);

        glGenVertexArrays(1, &particles[i].vao);
        glGenBuffers(1, &particles[i].vbo);
        glBindVertexArray(particles[i].vao);
        glBindBuffer(GL_ARRAY_BUFFER, particles[i].vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec2), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), (void*)0);
        glEnableVertexAttribArray(0);

        glGenVertexArrays(1, &particles[i].trailVAO);
        glGenBuffers(1, &particles[i].trailVBO);
        glBindVertexArray(particles[i].trailVAO);
        glBindBuffer(GL_ARRAY_BUFFER, particles[i].trailVBO);
        glBufferData(GL_ARRAY_BUFFER, maxTrailPoints * 2 * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float deltaTime = static_cast<float>(now - lastTime);
        lastTime = now;

        // --- atualizar a posicao do fotao ---
        // por agora e so uma linha reta a velocidade constante; quando a lente
        // gravitacional estiver implementada, e esta parte que vai ser substituida
        // pela integracao da geodesica (dr, dtheta, dphi do struct Ray)
        rayPos.x += raySpeed * deltaTime;
        if (rayPos.x > rayEndX) {
            rayPos = vec2(rayStartX, rayY);
            trail.clear();
        }

        trail.push_back(rayPos);
        if (trail.size() >= maxTrailPoints) {
            trail.erase(trail.begin());   // so entra em accao se algo correr muito mal (seguranca)
        }

        // manda a nova posicao do fotao para a GPU
        glBindBuffer(GL_ARRAY_BUFFER, rayVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec2), &rayPos);

        // rasto do fotao: faixa que afunila da cabeca (largura = kDotSizePx) ate a cauda
        const float TRAIL_MAX_ALPHA = 0.85f;
        buildTaperedTrail(trail, headHalfWidthNDC, TRAIL_MAX_ALPHA, trailData);

        // manda o rasto atualizado para a GPU
        glBindBuffer(GL_ARRAY_BUFFER, trailVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, trailData.size() * sizeof(float), trailData.data());

        // --- atualizar as particulas: integracao da geodesica radial em tempo proprio ---
        // dtau = deltaTime * TAU_PLAYBACK_SPEED: o tempo de frame da simulacao representa
        // o tempo proprio DE CADA PARTICULA, nao o de um observador externo parado (ver
        // nota junto a geodesicDrDTau). As 5 partilham r0, E, L - so o phi difere - por
        // isso r(tau) e' identico para todas: caem ao mesmo ritmo radial, e so' a distancia
        // lateral entre elas muda com r (converge, porque todas as retas radiais se
        // encontram no centro).
        for (auto& pv : particles) {
            Particle& p = pv.particle;
            if (p.active) {
                double dtau = static_cast<double>(deltaTime) * TAU_PLAYBACK_SPEED;
                double newR = rk4StepR(p.r, dtau, bh.r_s, p.E, p.L);
                p.dr = geodesicDrDTau(p.r, bh.r_s, p.E, p.L);
                p.r = newR;

                if (p.r <= bh.r_s) {
                    // cruzou o horizonte de eventos: resultado de r(tau) ter chegado a r_s
                    // na equacao integrada, nao uma condicao arbitraria escolhida a parte
                    p.active = false;
                    pv.respawnTimer = 0.0f;
                } else {
                    p.posCart.x = static_cast<float>(p.r * std::cos(p.phi) * kWorldScale);
                    p.posCart.y = static_cast<float>(p.r * std::sin(p.phi) * kWorldScale);

                    pv.trail.push_back(vec2(p.posCart.x, p.posCart.y));
                    if (pv.trail.size() >= maxTrailPoints) {
                        pv.trail.erase(pv.trail.begin());
                    }

                    glBindBuffer(GL_ARRAY_BUFFER, pv.vbo);
                    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec2), &p.posCart);
                }
            } else {
                // fica desaparecida uma pausa antes de recomecar, para dar tempo a
                // perceber visualmente que foi engolida - nao e' um efeito fisico, e' so
                // para tornar a demonstracao legivel em loop
                pv.respawnTimer += deltaTime;
                if (pv.respawnTimer >= particleRespawnDelay) {
                    resetParticle(p, pv.phi0);
                    pv.trail.clear();
                }
            }

            buildTaperedTrail(pv.trail, headHalfWidthNDC, TRAIL_MAX_ALPHA, pv.trailData);
            glBindBuffer(GL_ARRAY_BUFFER, pv.trailVBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, pv.trailData.size() * sizeof(float), pv.trailData.data());
        }

        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // CAMADA 1: buraco negro
        glUseProgram(pointProg);
        bh.draw(pointProg, bhVAO, pointColorLoc, pointSizeLoc, bhRadiusPx);

        /*
        // CAMADA 2: rasto do fotao (desenhado antes do ponto, para o ponto ficar sempre por cima)
        glUseProgram(lineProg);
        glUniform3f(lineColorLoc, 0.6f, 0.5f, 0.15f);   // amarelo mais apagado que o ponto
        glBindVertexArray(trailVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, static_cast<GLsizei>(trail.size() * 2));

        // CAMADA 3: ponto do fotao - continua a nao ser uma "particula" com volume,
        // mas 3px estava dificil de ver; 6px ainda le-se como um ponto de luz, nao um disco
        glUseProgram(pointProg);
        glUniform3f(pointColorLoc, 1.0f, 0.9f, 0.4f);   // amarelo-luz
        glUniform1f(pointSizeLoc, kDotSizePx);
        glBindVertexArray(rayVAO);
        glDrawArrays(GL_POINTS, 0, 1);
        */

        // CAMADA 4 e 5: rasto e ponto de cada uma das 5 particulas. O desaparecimento
        // continua a vir so' de nao desenhar o ponto quando particle.active == false
        // (o rasto fica visivel mais um pouco, a desvanecer, mesmo depois do ponto sumir)
        for (auto& pv : particles) {
            glUseProgram(lineProg);
            //glUniform3f(lineColorLoc, pv.color.r, pv.color.g, pv.color.b);
            glUniform3f(lineColorLoc, 0.6f, 0.5f, 0.15f);
            glBindVertexArray(pv.trailVAO);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, static_cast<GLsizei>(pv.trail.size() * 2));

            if (pv.particle.active) {
                glUseProgram(pointProg);
                //glUniform3f(pointColorLoc, pv.color.r, pv.color.g, pv.color.b);
                glUniform3f(pointColorLoc, 1.0f, 0.9f, 0.4f);
                glUniform1f(pointSizeLoc, kDotSizePx);
                glBindVertexArray(pv.vao);
                glDrawArrays(GL_POINTS, 0, 1);
            }
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}