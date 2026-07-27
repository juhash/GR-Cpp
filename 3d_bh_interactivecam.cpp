#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
using glm::vec3;
using glm::vec2;
using glm::mat4;

const double G = 1.0;
const double c = 1.0;
const double kWorldScale = 0.05;

const char* pointVertexSrc = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform float uPointSize;
uniform mat4 uViewProj;
// --- CAMERA INTERATIVA: fogNear/fogFar deixam de ser constantes fixas no shader ---
// Antes destas duas uniforms existirem, fogNear/fogFar eram numeros escritos aqui
// dentro (2.2 / 4.5), afinados a olho para UMA distancia de camara fixa. Agora que a
// camara pode fazer zoom (ver OrbitCamera/scrollCallback mais abaixo, e o bloco de
// input dentro do while() em main()), essa distancia muda a cada frame - por isso
// fogNear/fogFar tem de vir de fora, recalculados no CPU a partir da distancia atual
// da camara (orbitCam.distance) e enviados como uniforms, em vez de ficarem hardcoded
// aqui. Sem isto, o efeito de nevoeiro ficaria errado sempre que desses zoom: perto
// de mais ou longe de mais do intervalo fixo original.
uniform float uFogNear;
uniform float uFogFar;
out float vFade;
void main() {
    gl_Position = uViewProj * vec4(aPos, 1.0);
    gl_PointSize = clamp(uPointSize / gl_Position.w, 1.0, 200.0);

    float fadeMin = 0.15;
    float t = clamp((gl_Position.w - uFogNear) / (uFogFar - uFogNear), 0.0, 1.0);
    vFade = mix(1.0, fadeMin, t);
}
)";

const char* pointFragmentSrc = R"(
#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
in float vFade;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    FragColor = vec4(uColor * vFade, 1.0);
}
)";

const char* lineVertexSrc = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aAlpha;
out float vAlpha;
uniform mat4 uViewProj;
// --- CAMERA INTERATIVA: mesma razao do pointVertexSrc (ver comentario la') ---
// uFogNear/uFogFar tem de ser uniforms aqui tambem, porque este shader desenha os
// rastos dos fotoes e precisa do MESMO nevoeiro dependente do zoom que os pontos usam.
uniform float uFogNear;
uniform float uFogFar;
void main() {
    gl_Position = uViewProj * vec4(aPos, 1.0);
    float fadeMin = 0.15;
    float t = clamp((gl_Position.w - uFogNear) / (uFogFar - uFogNear), 0.0, 1.0);
    float depthFade = mix(1.0, fadeMin, t);
    vAlpha = aAlpha * depthFade;
}
)";

const char* lineFragmentSrc = R"(
#version 330 core
in float vAlpha;
out vec4 FragColor;
uniform vec3 uColor;
void main() { FragColor = vec4(uColor, vAlpha); }
)";

GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    int ok; glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[512]; glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "Erro shader:\n" << log << "\n"; }
    return shader;
}

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
        glUniform3f(uColorLoc, 1.0f, 0.0f, 0.0f);
        glUniform1f(uSizeLoc, displayRadiusPx);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, 1);
    }
};

struct Photon {
    vec3   posCart;
    double r, phi;
    double dr;
    double b;
    bool   active = true;
};

struct PhotonVisual {
    Photon photon;
    double phi0;
    double b0;
    vec3   planeE1, planeE2;
    std::vector<vec3>  trail;
    std::vector<float> trailData;
    float  respawnTimer = 0.0f;
    GLuint vao = 0, vbo = 0;
    GLuint trailVAO = 0, trailVBO = 0;
};

double photonInitialDr(double r, double r_s, double b) {
    double term = 1.0 - (b*b / (r*r)) * (1.0 - r_s/r);
    if (term < 0.0) term = 0.0;
    return -std::sqrt(term);
}

double radialAccelPhoton(double r, double r_s, double b) {
    return (b*b) / (r*r*r) - 1.5 * r_s * (b*b) / (r*r*r*r);
}

struct OrbitStep { double r, phi, dr; };

OrbitStep orbitDeriv(const OrbitStep& s, double r_s, double b) {
    OrbitStep d;
    d.r   = s.dr;
    d.phi = b / (s.r * s.r);
    d.dr  = radialAccelPhoton(s.r, r_s, b);
    return d;
}

OrbitStep rk4StepOrbit(const OrbitStep& s, double dlambda, double r_s, double b) {
    OrbitStep k1 = orbitDeriv(s, r_s, b);
    OrbitStep s2{ s.r + 0.5*dlambda*k1.r, s.phi + 0.5*dlambda*k1.phi, s.dr + 0.5*dlambda*k1.dr };
    OrbitStep k2 = orbitDeriv(s2, r_s, b);
    OrbitStep s3{ s.r + 0.5*dlambda*k2.r, s.phi + 0.5*dlambda*k2.phi, s.dr + 0.5*dlambda*k2.dr };
    OrbitStep k3 = orbitDeriv(s3, r_s, b);
    OrbitStep s4{ s.r + dlambda*k3.r, s.phi + dlambda*k3.phi, s.dr + dlambda*k3.dr };
    OrbitStep k4 = orbitDeriv(s4, r_s, b);

    OrbitStep out;
    out.r   = s.r   + (dlambda/6.0) * (k1.r   + 2.0*k2.r   + 2.0*k3.r   + k4.r);
    out.phi = s.phi + (dlambda/6.0) * (k1.phi + 2.0*k2.phi + 2.0*k3.phi + k4.phi);
    out.dr  = s.dr  + (dlambda/6.0) * (k1.dr  + 2.0*k2.dr  + 2.0*k3.dr  + k4.dr);
    return out;
}

void buildTaperedTrail(const std::vector<vec3>& trail, float headHalfWidth,
                        float maxAlpha, std::vector<float>& out) {
    out.clear();
    const size_t n = trail.size();
    if (n == 0) return;
    const vec3 worldUp(0.0f, 1.0f, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        vec3 dir;
        if (i + 1 < n)      dir = trail[i + 1] - trail[i];
        else if (i > 0)     dir = trail[i] - trail[i - 1];
        else                dir = vec3(1.0f, 0.0f, 0.0f);

        float len = glm::length(dir);
        vec3 perp = (len > 1e-6f) ? glm::normalize(glm::cross(dir, worldUp)) : vec3(0.0f, 0.0f, 1.0f);
        if (glm::length(perp) < 1e-6f) perp = glm::normalize(glm::cross(dir, vec3(1.0f, 0.0f, 0.0f)));

        float t = (n > 1) ? static_cast<float>(i) / static_cast<float>(n - 1) : 1.0f;
        float halfWidth = t * headHalfWidth;
        float alpha     = t * maxAlpha;

        vec3 left  = trail[i] + perp * halfWidth;
        vec3 right = trail[i] - perp * halfWidth;

        out.push_back(left.x);  out.push_back(left.y);  out.push_back(left.z);  out.push_back(alpha);
        out.push_back(right.x); out.push_back(right.y); out.push_back(right.z); out.push_back(alpha);
    }
}

// =====================================================================================
// CAMERA INTERATIVA - tudo a partir daqui ate' ao fim do ficheiro e' a parte nova desta
// copia: uma camara que se pode mover com o rato e o teclado, em vez da camara fixa
// (glm::lookAt/glm::perspective calculados uma unica vez) que o ficheiro original tinha.
// =====================================================================================

// Representacao da camara: coordenadas ESFERICAS a' volta de um alvo fixo (a origem,
// onde esta' o buraco negro) - azimuth (rotacao a' volta do eixo Y), elevation (angulo
// acima do plano horizontal) e distance (raio da orbita, ou seja, o zoom).
//
// Decisao deliberada: isto NAO e' uma camara livre (6DOF, WASD + mouselook, tipo jogo em
// primeira pessoa). Fica sempre a olhar para o alvo. Como o objetivo e' "ver o buraco
// negro de varios angulos e distancias", uma camara que anda livremente por aí so'
// tornaria mais facil perder o buraco negro de vista, sem trazer nenhum beneficio.
const float kOrbitMinDistance      = 0.35f;   // zoom maximo (aproximar) - impede a camara
                                               // de atravessar o buraco negro
const float kOrbitMaxDistance      = 6.0f;    // zoom minimo (afastar) - mantem a cena
                                               // dentro do far plane (100.0) com folga
const float kOrbitMaxElevation     = 1.48f;   // ~85 graus. Sem este limite, a elevation
                                               // podia chegar aos +-90 graus (olhar
                                               // exatamente de cima ou de baixo), ponto em
                                               // que a direcao de vista fica paralela ao
                                               // worldUpAxis usado em glm::lookAt - isso
                                               // faz o produto interno usado para
                                               // construir a base da camara colapsar a
                                               // zero, e a matriz resultante degenera
                                               // (imagem gira de forma instavel ou fica
                                               // preta). Ficar sempre um pouco aquem dos
                                               // polos evita esse caso.
const float kMouseOrbitSensitivity = 0.006f;  // radianos por pixel arrastado com o rato
const float kKeyRotateSpeed        = 1.2f;    // radianos por segundo, teclado (setas)
const float kKeyZoomSpeed          = 1.5f;    // fracao de distance por segundo, teclado (+/-)
const float kScrollZoomFactor      = 1.1f;    // multiplicador de distance por "notch" da roda

struct OrbitCamera {
    float azimuth   = 0.0f;
    float elevation = 0.0f;
    float distance  = 1.0f;

    // Converte (azimuth, elevation, distance) na posicao 3D da camara, todos os frames.
    // E' a conversao esferica -> cartesiana standard, com o eixo Y como polar (para
    // condizer com worldUpAxis, o "para cima" do resto da cena) e o angulo azimuth
    // medido a partir do eixo +Z. Esta escolha especifica (e nao, por exemplo, medir a
    // partir do eixo +X) e' o que permite reproduzir exatamente a antiga camara fixa
    // (0, 1.8, 2.6) quando azimuth=0 - ver os valores iniciais em main().
    vec3 position() const {
        float ce = std::cos(elevation);
        return vec3(distance * ce * std::sin(azimuth),
                     distance * std::sin(elevation),
                     distance * ce * std::cos(azimuth));
    }

    // Mantem azimuth/elevation/distance dentro de limites seguros (ver os comentarios
    // junto a`s constantes kOrbit* acima para o porque de cada limite). Chamada sempre
    // que algum input (teclado, rato ou scroll) pode ter alterado estes valores.
    void clampRanges() {
        if (elevation >  kOrbitMaxElevation) elevation =  kOrbitMaxElevation;
        if (elevation < -kOrbitMaxElevation) elevation = -kOrbitMaxElevation;
        if (distance < kOrbitMinDistance) distance = kOrbitMinDistance;
        if (distance > kOrbitMaxDistance) distance = kOrbitMaxDistance;
    }
};

// Callback de scroll da roda do rato. Esta e' a UNICA parte do input da camara que
// precisa mesmo de ser um callback, em vez de ser perguntada a cada frame (por
// "polling", como o botao do rato e o teclado sao mais abaixo): o GLFW nao guarda
// nenhum estado tipo "quanto se rodou a roda ate agora" que se possa consultar a
// qualquer momento - a unica forma de saber que a roda rodou e' registar uma funcao
// que o GLFW chama no momento em que isso acontece.
//
// Como esta funcao tem de ter exatamente a assinatura que o GLFW espera (e' uma funcao
// C livre, chamada pela biblioteca), nao pode ser uma lambda com capturas - por isso nao
// ha' forma direta de lhe dar acesso a` variavel orbitCam que vive dentro de main(). A
// solucao e' glfwSetWindowUserPointer/glfwGetWindowUserPointer: em main(), guardamos o
// ENDERECO de orbitCam "dentro" da janela GLFW; aqui, vamos busca-lo de volta.
void scrollCallback(GLFWwindow* window, double /*xoffset*/, double yoffset) {
    auto* cam = static_cast<OrbitCamera*>(glfwGetWindowUserPointer(window));
    if (!cam) return;
    // yoffset e' tipicamente +1 ou -1 por "notch" da roda. Multiplicar (em vez de somar)
    // por um fator > 1 faz o zoom sentir-se com velocidade constante RELATIVA (cada
    // notch aproxima/afasta a MESMA percentagem da distancia atual), o que parece mais
    // natural do que uma soma/subtracao fixa - perto do buraco negro cada notch move
    // pouco em unidades absolutas, longe move muito, tal como se espera de um zoom.
    cam->distance *= std::pow(kScrollZoomFactor, static_cast<float>(-yoffset));
    cam->clampRanges();
}

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
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);

    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);

    GLuint vs = compileShader(GL_VERTEX_SHADER, pointVertexSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, pointFragmentSrc);
    GLuint pointProg = linkProgram(vs, fs);
    glDeleteShader(vs); glDeleteShader(fs);
    GLint pointColorLoc = glGetUniformLocation(pointProg, "uColor");
    GLint pointSizeLoc  = glGetUniformLocation(pointProg, "uPointSize");
    GLint pointViewProjLoc = glGetUniformLocation(pointProg, "uViewProj");
    // CAMERA INTERATIVA: locations das duas novas uniforms de nevoeiro (ver pointVertexSrc)
    GLint pointFogNearLoc  = glGetUniformLocation(pointProg, "uFogNear");
    GLint pointFogFarLoc   = glGetUniformLocation(pointProg, "uFogFar");

    GLuint lvs = compileShader(GL_VERTEX_SHADER, lineVertexSrc);
    GLuint lfs = compileShader(GL_FRAGMENT_SHADER, lineFragmentSrc);
    GLuint lineProg = linkProgram(lvs, lfs);
    glDeleteShader(lvs); glDeleteShader(lfs);
    GLint lineColorLoc = glGetUniformLocation(lineProg, "uColor");
    GLint lineViewProjLoc = glGetUniformLocation(lineProg, "uViewProj");
    // CAMERA INTERATIVA: idem, para o shader das linhas (rasto dos fotoes)
    GLint lineFogNearLoc  = glGetUniformLocation(lineProg, "uFogNear");
    GLint lineFogFarLoc   = glGetUniformLocation(lineProg, "uFogFar");

    // --- CAMERA INTERATIVA: substitui a antiga camara fixa ---
    // No ficheiro original, a camara era um `glm::vec3 cameraPos(0, 1.8, 2.6)` fixo,
    // usado uma unica vez para calcular viewProj antes do while(...). Aqui, cameraPos
    // deixa de existir como constante - passa a ser DERIVADO, todos os frames, do estado
    // de orbitCam (ver a funcao position() na struct OrbitCamera acima).
    //
    // Os valores iniciais de distance/elevation/azimuth abaixo sao escolhidos para
    // reproduzir EXATAMENTE essa posicao antiga (0, 1.8, 2.6):
    //   distance  = |cameraPos| = sqrt(1.8^2 + 2.6^2)
    //   elevation = atan2(1.8, 2.6)               (angulo cuja tangente e' altura/profundidade)
    //   azimuth   = 0, porque cameraPos ja' estava no plano XZ com x=0
    // Ou seja: a primeira imagem desenhada no ecra e' identica a` versao sem camara
    // interativa - so' a partir do primeiro input (rato/teclado/scroll) e' que a vista
    // comeca a mudar.
    OrbitCamera orbitCam;
    orbitCam.distance  = std::sqrt(1.8f*1.8f + 2.6f*2.6f);
    orbitCam.elevation = std::atan2(1.8f, 2.6f);
    orbitCam.azimuth   = 0.0f;
    const OrbitCamera kDefaultCamera = orbitCam;   // guardado para a tecla de reset (R)

    const glm::vec3 cameraTarget(0.0f, 0.0f, 0.0f);
    const glm::vec3 worldUpAxis(0.0f, 1.0f, 0.0f);

    // A projecao (FOV/aspect/near/far) continua a nao depender da camara - so' fazemos
    // zoom por DISTANCIA (aproximar/afastar o ponto de vista), nao por FOV, por isso
    // projMatrix continua a so' precisar de ser calculada uma vez aqui fora do loop.
    mat4 projMatrix = glm::perspective(glm::radians(45.0f),
                                        static_cast<float>(fbWidth) / static_cast<float>(fbHeight),
                                        0.05f, 100.0f);

    // Liga orbitCam a` janela GLFW, para o scrollCallback (funcao C livre, sem capturas)
    // conseguir chegar-lhe - ver a explicacao completa junto a` definicao de
    // scrollCallback acima. Isto tem de ser feito ANTES de comecar a receber eventos
    // (glfwPollEvents la' em baixo), caso contrario um scroll muito cedo seria ignorado.
    glfwSetWindowUserPointer(window, &orbitCam);
    glfwSetScrollCallback(window, scrollCallback);

    // Estado do arrasto do rato (botao esquerdo), usado dentro do while(...) para saber
    // quanto o cursor se moveu desde o frame anterior enquanto o botao esta' premido.
    // Precisa de viver FORA do loop porque cada frame so' conhece a posicao "atual" do
    // cursor - sem guardar a posicao do frame anterior nao havia como calcular a
    // diferenca (delta) que faz a camara orbitar.
    bool   mouseWasDown = false;
    double lastCursorX = 0.0, lastCursorY = 0.0;

    BlackHole bh(vec3(0, 0, 0), 1.0f);

    float bhRadiusPx = static_cast<float>(bh.r_s * kWorldScale * fbHeight);

    const float kDotSizePx        = 2.5f;
    const float headHalfWidthWorld = 0.006f;

    float bhVertex[] = { bh.position.x, bh.position.y, bh.position.z };
    GLuint bhVAO, bhVBO;
    glGenVertexArrays(1, &bhVAO);
    glGenBuffers(1, &bhVBO);
    glBindVertexArray(bhVAO);
    glBindBuffer(GL_ARRAY_BUFFER, bhVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(bhVertex), bhVertex, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    const size_t maxTrailPoints = 4000;

    const double photonR0    = 15.0 * bh.r_s;
    const int    NUM_PHOTONS = 120;

    const double bCrit = 1.5 * std::sqrt(3.0) * bh.r_s;

    const double LAMBDA_PLAYBACK_SPEED = 15.0;
    const float  photonRespawnDelay    = 1.2f;
    const int    ORBIT_SUBSTEPS        = 12;

    const vec3 photonColor(1.0f, 0.9f, 0.35f);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    auto randomUnitVector = [&]() -> vec3 {
        double z = 2.0 * uni01(rng) - 1.0;
        double theta = 2.0 * M_PI * uni01(rng);
        double s = std::sqrt(std::max(0.0, 1.0 - z*z));
        return vec3(static_cast<float>(s * std::cos(theta)),
                    static_cast<float>(z),
                    static_cast<float>(s * std::sin(theta)));
    };

    auto buildPlaneBasis = [&](const vec3& n, vec3& e1, vec3& e2) {
        vec3 helper = (std::abs(n.y) < 0.9f) ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
        vec3 base1 = glm::normalize(glm::cross(helper, n));
        vec3 base2 = glm::cross(n, base1);
        float psi = static_cast<float>(2.0 * M_PI * uni01(rng));
        e1 = base1 * std::cos(psi) + base2 * std::sin(psi);
        e2 = -base1 * std::sin(psi) + base2 * std::cos(psi);
    };

    auto sampleImpactParameter = [&](int i) -> double {
        double frac = static_cast<double>(i) / static_cast<double>(NUM_PHOTONS);
        double t = uni01(rng);
        if (frac < 0.45) {
            double lo = 0.10 * bCrit, hi = 0.97 * bCrit;
            return lo + t * (hi - lo);
        } else if (frac < 0.65) {
            double lo = 0.985 * bCrit, hi = 1.02 * bCrit;
            return lo + t * (hi - lo);
        } else {
            double lo = 1.05 * bCrit, hi = 3.5 * bCrit;
            return lo + t * (hi - lo);
        }
    };

    auto resetPhoton = [&](Photon& ph, double phi0, double b0) {
        ph.r   = photonR0;
        ph.phi = phi0;
        ph.b   = b0;
        ph.dr  = photonInitialDr(ph.r, bh.r_s, ph.b);
        ph.active = true;
    };

    std::vector<PhotonVisual> photons(NUM_PHOTONS);
    for (int i = 0; i < NUM_PHOTONS; ++i) {
        photons[i].phi0 = 0.0;
        photons[i].b0   = sampleImpactParameter(i);
        vec3 n = randomUnitVector();
        buildPlaneBasis(n, photons[i].planeE1, photons[i].planeE2);
        resetPhoton(photons[i].photon, photons[i].phi0, photons[i].b0);
        photons[i].trail.reserve(maxTrailPoints);
        photons[i].trailData.reserve(maxTrailPoints * 2 * 4);

        glGenVertexArrays(1, &photons[i].vao);
        glGenBuffers(1, &photons[i].vbo);
        glBindVertexArray(photons[i].vao);
        glBindBuffer(GL_ARRAY_BUFFER, photons[i].vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec3), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)0);
        glEnableVertexAttribArray(0);

        glGenVertexArrays(1, &photons[i].trailVAO);
        glGenBuffers(1, &photons[i].trailVBO);
        glBindVertexArray(photons[i].trailVAO);
        glBindBuffer(GL_ARRAY_BUFFER, photons[i].trailVBO);
        glBufferData(GL_ARRAY_BUFFER, maxTrailPoints * 2 * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }

    std::vector<float> gridData;
    {
        const float gridExtent = 1.5f;
        const float gridStep   = 0.25f;
        const float gridY      = -0.02f;
        const float gridAlpha  = 0.12f;
        for (float x = -gridExtent; x <= gridExtent + 1e-4f; x += gridStep) {
            gridData.insert(gridData.end(), { x, gridY, -gridExtent, gridAlpha,  x, gridY, gridExtent, gridAlpha });
        }
        for (float z = -gridExtent; z <= gridExtent + 1e-4f; z += gridStep) {
            gridData.insert(gridData.end(), { -gridExtent, gridY, z, gridAlpha,  gridExtent, gridY, z, gridAlpha });
        }
    }
    GLuint gridVAO, gridVBO;
    glGenVertexArrays(1, &gridVAO);
    glGenBuffers(1, &gridVBO);
    glBindVertexArray(gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferData(GL_ARRAY_BUFFER, gridData.size() * sizeof(float), gridData.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    const GLsizei gridVertexCount = static_cast<GLsizei>(gridData.size() / 4);

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float deltaTime = static_cast<float>(now - lastTime);
        lastTime = now;

        // =================================================================
        // CAMERA INTERATIVA - bloco de input, uma vez por frame
        // =================================================================

        // Teclado, por POLLING: perguntamos o estado atual de cada tecla a cada frame
        // (glfwGetKey devolve GLFW_PRESS/GLFW_RELEASE na hora), em vez de precisar de um
        // callback - ao contrario do scroll (ver scrollCallback acima), o GLFW guarda o
        // estado das teclas e deixa consulta-lo a qualquer momento.
        // Setas: rodam a camara (azimuth/elevation) a uma velocidade angular constante,
        // multiplicada por deltaTime para nao depender da framerate.
        if (glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS) orbitCam.azimuth   -= kKeyRotateSpeed * deltaTime;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) orbitCam.azimuth   += kKeyRotateSpeed * deltaTime;
        if (glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS) orbitCam.elevation += kKeyRotateSpeed * deltaTime;
        if (glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS) orbitCam.elevation -= kKeyRotateSpeed * deltaTime;
        // +/- (tecla GLFW_KEY_EQUAL parilha com "+") ou Page Up/Down: zoom por teclado,
        // alternativa ao scroll do rato (por exemplo, para quem nao tenha roda).
        if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_PAGE_UP) == GLFW_PRESS)
            orbitCam.distance *= (1.0f - kKeyZoomSpeed * deltaTime);
        if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
            orbitCam.distance *= (1.0f + kKeyZoomSpeed * deltaTime);
        // R: repoe a camara nos valores iniciais (kDefaultCamera), guardados logo a
        // seguir a` criacao de orbitCam mais acima - uma forma rapida de "desfazer"
        // qualquer orbita/zoom e voltar ao enquadramento original.
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) orbitCam = kDefaultCamera;

        // Rato: arrastar com o botao ESQUERDO premido orbita a camara (tal como o botao
        // do meio no Blender, ou o arrasto no SketchUp) - tambem por polling
        // (glfwGetCursorPos/glfwGetMouseButton dao' logo o estado atual, nao ha'
        // informacao aqui que só' um callback pudesse fornecer).
        //
        // O CUIDADO importante esta' na transicao solto -> premido: nesse primeiro
        // frame do arrasto, so' guardamos a posicao atual do cursor SEM aplicar nenhuma
        // rotacao. Se aplicassemos logo, o delta seria calculado contra a posicao do
        // cursor de ANTES de premir o botao (que pode estar em qualquer lado do ecra,
        // de um movimento de rato que nao era suposto orbitar nada) - isso causaria um
        // "salto" brusco da camara logo no instante do clique. Guardando a posicao sem
        // mexer na camara nesse frame, o delta so' comeca a ser aplicado a partir do
        // SEGUNDO frame do arrasto em diante, que ja' e' um movimento real com o botao
        // ja' premido.
        double curCursorX, curCursorY;
        glfwGetCursorPos(window, &curCursorX, &curCursorY);
        bool mouseIsDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        if (mouseIsDown && mouseWasDown) {
            orbitCam.azimuth   -= static_cast<float>(curCursorX - lastCursorX) * kMouseOrbitSensitivity;
            orbitCam.elevation += static_cast<float>(curCursorY - lastCursorY) * kMouseOrbitSensitivity;
        }
        mouseWasDown = mouseIsDown;
        lastCursorX  = curCursorX;
        lastCursorY  = curCursorY;

        // Aplicado UMA vez por frame, depois de todas as fontes de input (teclado E
        // rato) terem tido oportunidade de alterar azimuth/elevation/distance - garante
        // que nenhuma combinacao de inputs no mesmo frame consegue ultrapassar os
        // limites definidos em OrbitCamera::clampRanges().
        orbitCam.clampRanges();

        // A camara deixou de ser estatica, por isso viewMatrix/viewProj (que antes eram
        // calculadas UMA VEZ, antes do while(...), e enviadas uma unica vez para os
        // shaders) tem agora de ser recalculadas E reenviadas em TODOS os frames, a
        // partir da posicao atual de orbitCam.
        mat4 viewMatrix = glm::lookAt(orbitCam.position(), cameraTarget, worldUpAxis);
        mat4 viewProj   = projMatrix * viewMatrix;

        // fogNear/fogFar (as novas uniforms uFogNear/uFogFar dos shaders) seguem o zoom
        // atual: sao recalculadas a partir de orbitCam.distance a cada frame, em vez de
        // serem as constantes fixas (2.2 / 4.5) que o shader tinha originalmente. Os
        // fatores 0.7/1.6 sao um ponto de partida escolhido a olho, para que o nevoeiro
        // continue a fazer sentido tanto de perto como de longe.
        float fogNear = orbitCam.distance * 0.7f;
        float fogFar  = orbitCam.distance * 1.6f;

        glUseProgram(pointProg);
        glUniformMatrix4fv(pointViewProjLoc, 1, GL_FALSE, &viewProj[0][0]);
        glUniform1f(pointFogNearLoc, fogNear);
        glUniform1f(pointFogFarLoc, fogFar);
        glUseProgram(lineProg);
        glUniformMatrix4fv(lineViewProjLoc, 1, GL_FALSE, &viewProj[0][0]);
        glUniform1f(lineFogNearLoc, fogNear);
        glUniform1f(lineFogFarLoc, fogFar);

        // =================================================================
        // fim do bloco de camera - daqui para baixo e' fisica/render, inalterado
        // =================================================================

        for (auto& pv : photons) {
            Photon& ph = pv.photon;
            if (ph.active) {
                double dlambda = static_cast<double>(deltaTime) * LAMBDA_PLAYBACK_SPEED;
                double subDlambda = dlambda / static_cast<double>(ORBIT_SUBSTEPS);

                bool captured = false, escaped = false;
                for (int step = 0; step < ORBIT_SUBSTEPS; ++step) {
                    OrbitStep s{ ph.r, ph.phi, ph.dr };
                    OrbitStep next = rk4StepOrbit(s, subDlambda, bh.r_s, ph.b);
                    ph.r = next.r; ph.phi = next.phi; ph.dr = next.dr;

                    if (ph.r <= bh.r_s) { captured = true; break; }
                    if (ph.r >= photonR0 && ph.dr > 0.0) { escaped = true; break; }
                }

                if (captured) {
                    ph.active = false;
                    pv.respawnTimer = 0.0f;
                } else if (escaped) {
                    ph.active = false;
                    pv.respawnTimer = 0.0f;
                } else {
                    float rf   = static_cast<float>(ph.r * kWorldScale);
                    float cphi = static_cast<float>(std::cos(ph.phi));
                    float sphi = static_cast<float>(std::sin(ph.phi));
                    ph.posCart = rf * (cphi * pv.planeE1 + sphi * pv.planeE2);

                    pv.trail.push_back(ph.posCart);
                    if (pv.trail.size() >= maxTrailPoints) {
                        pv.trail.erase(pv.trail.begin());
                    }

                    glBindBuffer(GL_ARRAY_BUFFER, pv.vbo);
                    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec3), &ph.posCart);
                }
            } else {
                pv.respawnTimer += deltaTime;
                if (pv.respawnTimer >= photonRespawnDelay) {
                    resetPhoton(ph, pv.phi0, pv.b0);
                    pv.trail.clear();
                }
            }

            buildTaperedTrail(pv.trail, headHalfWidthWorld, 0.85f, pv.trailData);
            glBindBuffer(GL_ARRAY_BUFFER, pv.trailVBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, pv.trailData.size() * sizeof(float), pv.trailData.data());
        }

        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(lineProg);
        glUniform3f(lineColorLoc, 0.3f, 0.3f, 0.38f);
        glBindVertexArray(gridVAO);
        glDrawArrays(GL_LINES, 0, gridVertexCount);

        glUseProgram(pointProg);
        bh.draw(pointProg, bhVAO, pointColorLoc, pointSizeLoc, bhRadiusPx);

        for (auto& pv : photons) {
            glUseProgram(lineProg);
            glUniform3f(lineColorLoc, photonColor.r, photonColor.g, photonColor.b);
            glBindVertexArray(pv.trailVAO);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, static_cast<GLsizei>(pv.trail.size() * 2));

            if (pv.photon.active) {
                glUseProgram(pointProg);
                glUniform3f(pointColorLoc, photonColor.r, photonColor.g, photonColor.b);
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
