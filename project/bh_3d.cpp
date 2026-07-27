#include <GL/glew.h>      // tem de vir ANTES do glfw
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>   // NOVO (3D): glm::perspective, glm::lookAt
#include <iostream>
#include <vector>
#include <cmath>
#include <random>         // NOVO (D): orientacao 3D aleatoria do plano orbital de cada fotao
using glm::vec3;
using glm::vec2;
using glm::mat4;

//constantes
const double G = 1.0;
const double c = 1.0;

// Escala unica: 1 unidade de r em unidades geometrizadas (G=c=1) equivale a
// kWorldScale unidades NDC (-1 a 1). Usada tanto para calcular o tamanho do BH
// no ecra como para posicionar a particula que vai cair - e a ligacao entre a
// fisica e o ecra que faltava antes (so o tamanho do BH tinha escala; a posicao
// dos raios era arbitraria em NDC, desligada de r_s). Com mass=1.0 -> r_s=2.0,
// kWorldScale=0.05 da um raio em NDC de 0.1 (~ igual ao tamanho visual anterior).
// NOVO (3D): "unidades NDC" acima ja nao e' literal - kWorldScale continua a
// converter r (fisica) para as MESMAS unidades numericas de antes, mas agora essas
// unidades sao coordenadas de MUNDO 3D, que so' chegam a NDC depois de passar pela
// camara/projecao (uViewProj). Mantemos o valor porque a camara foi colocada a uma
// distancia que reproduz aproximadamente o mesmo enquadramento visual de antes.
const double kWorldScale = 0.05;

// OS SHADERS (em GLSL, como texto)

// Shader dos pontos: reutilizado tanto para o buraco negro como para os fotoes.
// Antes de cada draw call mudamos uColor e uPointSize consoante o que estamos a desenhar.
const char* pointVertexSrc = R"(
#version 330 core
layout (location = 0) in vec3 aPos;   // NOVO (3D): era vec2, posicao agora e' no mundo 3D, nao em NDC
uniform float uPointSize;   // deixou de estar fixo (60.0) no shader, agora vem de fora
uniform mat4 uViewProj;     // NOVO (3D): camara (view) * projecao, substitui o mapeamento direto para NDC
out float vFade;   // NOVO (C): atenuacao de brilho com a distancia a camara
void main() {
    gl_Position = uViewProj * vec4(aPos, 1.0);
    // NOVO (3D): sem isto, um ponto longe da camara ficaria do mesmo tamanho em pixeis
    // que um ponto perto - gl_Position.w e' a profundidade em espaco de vista (perspectiva),
    // por isso dividir por ele encolhe o ponto com a distancia. O clamp evita valores
    // absurdos quando w e' muito pequeno (perto do plano da camara).
    gl_PointSize = clamp(uPointSize / gl_Position.w, 1.0, 200.0);

    // NOVO (C): "nevoeiro" linear simples - gl_Position.w e' a distancia em espaco de
    // vista (a mesma usada acima para o tamanho do ponto); perto da camara (fogNear) fica
    // brilho total, longe (fogFar) esbate para fadeMin. So' uma pista visual de profundidade,
    // nao fisica real - valores de partida, ajustar a olho consoante a posicao da camara.
    float fogNear = 2.2;
    float fogFar  = 4.5;
    float fadeMin = 0.15;
    float t = clamp((gl_Position.w - fogNear) / (fogFar - fogNear), 0.0, 1.0);
    vFade = mix(1.0, fadeMin, t);
}
)";

const char* pointFragmentSrc = R"(
#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
in float vFade;   // NOVO (C)
void main() {
    // gl_PointCoord vai de (0,0) a (1,1) dentro do ponto; o centro e (0.5,0.5)
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;   // fora do raio 0.5 -> deita o pixel fora
    FragColor = vec4(uColor * vFade, 1.0);   // NOVO (C): escurece com a distancia a camara
}
)";

// Shader das linhas: usado para o rasto de cada fotao.
// Cada vertice tem a sua propria transparencia (aAlpha), para o rasto ir desaparecendo
// da ponta mais antiga para a mais recente - o GPU interpola isto ao longo da linha.
const char* lineVertexSrc = R"(
#version 330 core
layout (location = 0) in vec3 aPos;   // NOVO (3D): era vec2
layout (location = 1) in float aAlpha;
out float vAlpha;
uniform mat4 uViewProj;   // NOVO (3D): mesma camara/projecao do shader dos pontos
void main() {
    gl_Position = uViewProj * vec4(aPos, 1.0);
    // NOVO (C): mesmo "nevoeiro" linear do shader dos pontos, mas aplicado ao ALPHA em
    // vez da cor (o rasto ja usa alpha para desvanecer a cauda - aqui multiplicam-se os
    // dois efeitos: cauda mais transparente E tudo mais transparente ao longe).
    float fogNear = 2.2;
    float fogFar  = 4.5;
    float fadeMin = 0.15;
    float t = clamp((gl_Position.w - fogNear) / (fogFar - fogNear), 0.0, 1.0);
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

// NOVO (D): ja nao existe um struct Particle separado para materia do disco - depois de
// confirmar com o utilizador, TUDO o que se ve nesta cena passou a ser luz (fotoes),
// nao materia caindo. O struct Ray "por implementar" que existia aqui (nunca chegou a
// ser instanciado; o fotao antigo era so' uma posicao vec3 a mover-se em linha reta, sem
// gravidade nenhuma) foi removido e substituido por este, que segue mesmo a geodesica
// nula da metrica de Schwarzschild.
struct Photon {
    vec3   posCart;
    double r, phi;   // "theta" deixa de existir como variavel de estado: cada fotao vive
                      // no SEU proprio plano orbital (ver PhotonVisual::planeE1/planeE2),
                      // nao no plano equatorial fixo (theta=pi/2) que a materia usava -
                      // por simetria esferica de Schwarzschild, qualquer plano que passe
                      // pelo centro e' uma geodesica igualmente valida.
    double dr;
    double b;         // parametro de impacto (L/E); ver comentario junto a radialAccelPhoton
    bool   active = true;
};

// Agrupa tudo o que pertence a UM fotao desenhado no ecra: o estado fisico (Photon), o
// seu historico de posicoes (trail), a orientacao 3D do seu plano orbital, e os seus
// proprios buffers GPU.
struct PhotonVisual {
    Photon photon;
    double phi0;             // fase angular de partida dentro do PROPRIO plano (arbitraria
                              // - a variedade real vem da orientacao do plano, nao disto)
    double b0;                // parametro de impacto fixo (o respawn repoe o mesmo fotao)
    vec3   planeE1, planeE2;  // NOVO (D): base ortonormal do plano orbital deste fotao em
                               // 3D; (r,phi) em coordenadas polares DENTRO do plano mapeiam
                               // para 3D via posCart = r*(cos(phi)*planeE1 + sin(phi)*planeE2)
    std::vector<vec3>  trail;
    std::vector<float> trailData;
    float  respawnTimer = 0.0f;
    GLuint vao = 0, vbo = 0;
    GLuint trailVAO = 0, trailVBO = 0;
};

// --- FISICA: geodesica NULA (fotao) na metrica de Schwarzschild ---
//
// Fotoes nao tem massa: a equacao de energia perde o termo "+1" que a versao massiva
// tinha (esse termo vem de normalizar a 4-velocidade de uma particula massiva, u.u=-1;
// para luz, k.k=0 - a normalizacao e' outra). Com E, L as quantidades conservadas por
// unidade de parametro AFIM (fotoes nao tem tempo proprio, por isso lambda em vez de
// tau), fica:
//
//   (dr/dlambda)^2 = E^2 - (L^2/r^2)(1 - r_s/r)
//
// A trajetoria de um fotao so' depende da RAZAO b = L/E ("parametro de impacto"), nunca
// de E e L em separado - por isso fixamos E=1 sempre e chamamos a L resultante de b: e'
// o UNICO numero que distingue um fotao de outro.
double photonInitialDr(double r, double r_s, double b) {
    double term = 1.0 - (b*b / (r*r)) * (1.0 - r_s/r);
    if (term < 0.0) term = 0.0;   // protecao numerica
    return -std::sqrt(term);      // negativo: vem "de fora", r comeca a diminuir
}

// Mesma tecnica usada na versao massiva (derivar a equacao de energia em ordem a r, ver
// historico junto a radialAccel/OrbitStep para o racional completo): d^2r/dlambda^2 =
// -0.5 * dV_eff/dr, com V_eff(r) = (b^2/r^2)(1 - r_s/r) = b^2/r^2 - r_s*b^2/r^3. Repara
// que FALTA aqui o termo -M/r^2 (gravidade Newtoniana) que a versao massiva tinha: um
// fotao nao "cai" por atracao Newtoniana simples, so' sente a parte puramente
// relativista (o termo em r^-4). E' esse termo que cria a esfera de fotoes em
// r_photon=1.5*r_s=3M: abaixo dela nao existe nenhuma orbita circular de luz possivel,
// so' captura ou fuga.
double radialAccelPhoton(double r, double r_s, double b) {
    return (b*b) / (r*r*r) - 1.5 * r_s * (b*b) / (r*r*r*r);
}

// Estado orbital completo: posicao radial e angular, e a velocidade radial (dr/dlambda)
// como variavel de estado explicita - e' o que permite mudar de sinal num
// pericentro/apocentro (ou dar varias voltas perto da esfera de fotoes) sem ambiguidade.
struct OrbitStep { double r, phi, dr; };

OrbitStep orbitDeriv(const OrbitStep& s, double r_s, double b) {
    OrbitStep d;
    d.r   = s.dr;                          // dr/dlambda = a propria velocidade radial
    d.phi = b / (s.r * s.r);                // dphi/dlambda = L/r^2 = b/r^2 (E=1)
    d.dr  = radialAccelPhoton(s.r, r_s, b); // d(dr/dlambda)/dlambda = aceleracao radial
    return d;
}

// RK4 para o sistema acoplado (r, phi, dr) - 4 avaliacoes da derivada, pesos classicos
// 1-2-2-1, com phi integrado com a mesma precisao de 4a ordem que r.
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

// Constroi uma faixa (ribbon) que afunila da cabeca (ponto mais recente, largura
// headHalfWidthWorld) ate a cauda (ponto mais antigo, largura zero) - como o rasto de
// um cometa, em vez de uma linha de largura constante. Para cada ponto central do
// rasto, gera DOIS vertices (esquerda/direita), deslocados na perpendicular a
// direcao local do movimento. O resultado destina-se a GL_TRIANGLE_STRIP (nao
// GL_LINE_STRIP): a largura por vertice so e' possivel com geometria real, ja que
// glLineWidth() nao e fiavel (nem por vertice) em core profile.
// NOVO (3D): trail e' agora vec3 (mundo 3D), nao vec2 (NDC). A perpendicular 2D
// (-dir.y, dir.x) so fazia sentido em espaco de ecra; em 3D precisamos de um eixo de
// referencia para definir "para os lados". worldUp e' so' uma referencia GLOBAL para
// orientar a fita (nao o plano orbital de cada fotao, que agora pode ser qualquer um) -
// funciona para qualquer direcao excepto quando dir e' quase paralelo a worldUp, caso em
// que o cross cai para o fallback (0,0,1) mais abaixo.
void buildTaperedTrail(const std::vector<vec3>& trail, float headHalfWidth,
                        float maxAlpha, std::vector<float>& out) {
    out.clear();
    const size_t n = trail.size();
    if (n == 0) return;
    const vec3 worldUp(0.0f, 1.0f, 0.0f);   // referencia global para "para os lados" da fita

    for (size_t i = 0; i < n; ++i) {
        vec3 dir;
        if (i + 1 < n)      dir = trail[i + 1] - trail[i];   // direcao para o proximo ponto
        else if (i > 0)     dir = trail[i] - trail[i - 1];   // ultimo ponto: usa o anterior
        else                dir = vec3(1.0f, 0.0f, 0.0f);    // so ha 1 ponto - arbitrario

        float len = glm::length(dir);
        vec3 perp = (len > 1e-6f) ? glm::normalize(glm::cross(dir, worldUp)) : vec3(0.0f, 0.0f, 1.0f);
        if (glm::length(perp) < 1e-6f) perp = glm::normalize(glm::cross(dir, vec3(1.0f, 0.0f, 0.0f)));

        float t = (n > 1) ? static_cast<float>(i) / static_cast<float>(n - 1) : 1.0f;
        float halfWidth = t * headHalfWidth;   // zero na cauda, maximo na cabeca
        float alpha     = t * maxAlpha;

        vec3 left  = trail[i] + perp * halfWidth;
        vec3 right = trail[i] - perp * halfWidth;

        // 4 floats por vertice (x, y, z, alpha)
        out.push_back(left.x);  out.push_back(left.y);  out.push_back(left.z);  out.push_back(alpha);
        out.push_back(right.x); out.push_back(right.y); out.push_back(right.z); out.push_back(alpha);
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
    // NOVO (3D): sem depth test, o que se desenha por cima e' so' o que foi desenhado
    // por ULTIMO (ordem do codigo), nao o que esta' mais perto da camara. Com a camara em
    // perspectiva e fotoes espalhados em planos orbitais 3D aleatorios (NOVO (D)), isto e'
    // o que garante que um fotao que passe ENTRE a camara e o buraco negro aparece por
    // cima dele no ecra, e um que passe do lado de la fica escondido atras - profundidade
    // real, nao so' ordem de desenho.
    glEnable(GL_DEPTH_TEST);

    // viewport (importante em ecras HiDPI, onde o framebuffer pode ter mais pixeis que a janela)
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);

    // compilar e ligar os shaders
    // programa dos pontos (BH + fotoes, partilhado)
    GLuint vs = compileShader(GL_VERTEX_SHADER, pointVertexSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, pointFragmentSrc);
    GLuint pointProg = linkProgram(vs, fs);
    glDeleteShader(vs); glDeleteShader(fs);
    GLint pointColorLoc = glGetUniformLocation(pointProg, "uColor");
    GLint pointSizeLoc  = glGetUniformLocation(pointProg, "uPointSize");
    GLint pointViewProjLoc = glGetUniformLocation(pointProg, "uViewProj");   // NOVO (3D)

    // programa das linhas (rasto dos fotoes)
    GLuint lvs = compileShader(GL_VERTEX_SHADER, lineVertexSrc);
    GLuint lfs = compileShader(GL_FRAGMENT_SHADER, lineFragmentSrc);
    GLuint lineProg = linkProgram(lvs, lfs);
    glDeleteShader(lvs); glDeleteShader(lfs);
    GLint lineColorLoc = glGetUniformLocation(lineProg, "uColor");
    GLint lineViewProjLoc = glGetUniformLocation(lineProg, "uViewProj");   // NOVO (3D)

    // NOVO (3D): camara e projecao. Colocada ligeiramente acima e afastada, a olhar para
    // a origem, para que a cena apareca em perspectiva com profundidade real.
    const glm::vec3 cameraPos(0.0f, 1.8f, 2.6f);
    const glm::vec3 cameraTarget(0.0f, 0.0f, 0.0f);
    const glm::vec3 worldUpAxis(0.0f, 1.0f, 0.0f);
    mat4 viewMatrix = glm::lookAt(cameraPos, cameraTarget, worldUpAxis);
    // aspect ratio recalculado depois de termos fbWidth/fbHeight (ja obtidos acima);
    // near/far em unidades do mundo (mesmas unidades de kWorldScale)
    mat4 projMatrix = glm::perspective(glm::radians(45.0f),
                                        static_cast<float>(fbWidth) / static_cast<float>(fbHeight),
                                        0.05f, 100.0f);
    mat4 viewProj = projMatrix * viewMatrix;

    // a camara e' estatica por agora, por isso a matriz so precisa de ser enviada
    // uma vez a cada programa, nao a cada frame (quando a camara interativa for
    // implementada, isto passa para dentro do while(...))
    glUseProgram(pointProg);
    glUniformMatrix4fv(pointViewProjLoc, 1, GL_FALSE, &viewProj[0][0]);
    glUseProgram(lineProg);
    glUniformMatrix4fv(lineViewProjLoc, 1, GL_FALSE, &viewProj[0][0]);

    BlackHole bh(vec3(0, 0, 0), 1.0f);

    // tamanho do BH no ecra, calculado a partir de r_s (fisica) e nao de um numero
    // decorativo. raio em NDC = r_s * kWorldScale; como 2 unidades NDC (-1 a 1) cobrem
    // fbHeight pixels, 1 unidade NDC = fbHeight/2 pixels, logo o DIAMETRO em pixels
    // (e' o que gl_PointSize espera) e' raioNDC * fbHeight (o *2 do diametro cancela
    // com o /2 da conversao NDC->pixels)
    float bhRadiusPx = static_cast<float>(bh.r_s * kWorldScale * fbHeight);

    // NOVO (D): fotoes sao luz, nao cometas - o ponto e o rasto ficam bem mais finos do
    // que a materia do disco ficava antes (era 6px/0.015 mundo; um fotao nao tem volume,
    // so' faz sentido ler-se como um tracinho de luz fino).
    const float kDotSizePx        = 2.5f;
    const float headHalfWidthWorld = 0.006f;

    // VAO do buraco negro: um unico ponto, na posicao do bh
    float bhVertex[] = { bh.position.x, bh.position.y, bh.position.z };
    GLuint bhVAO, bhVBO;
    glGenVertexArrays(1, &bhVAO);
    glGenBuffers(1, &bhVBO);
    glBindVertexArray(bhVAO);
    glBindBuffer(GL_ARRAY_BUFFER, bhVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(bhVertex), bhVertex, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    const size_t maxTrailPoints = 4000;   // capacidade de seguranca do VBO por fotao, nao
                                           // um limite ativo - o rasto so e' limpo quando o
                                           // fotao e' capturado/escapa e reinicia

    // --- FOTOES: luz a passar por varias direccoes 3D a volta do buraco negro ---
    //
    // NOVO (D): antes disto havia dois sistemas separados e desligados um do outro - um
    // "fotao" decorativo (reta horizontal, sem gravidade, desenho comentado/desligado) e
    // 200 particulas de MATERIA (geodesica temporal, com massa) a cair/orbitar sempre no
    // mesmo plano equatorial (Y=0). Confirmado com o utilizador: esta cena passa a
    // mostrar so' luz - cada "particula" agora e' um FOTAO real, integrado pela geodesica
    // NULA (ver photonInitialDr/radialAccelPhoton acima), e cada um vive no seu PROPRIO
    // plano orbital em 3D (nao so' no plano XZ) - por simetria esferica de Schwarzschild,
    // qualquer plano que passe pela origem e' uma geodesica igualmente valida. E' esta
    // diversidade de planos (nao so' de posicao dentro de um unico plano) que da'
    // profundidade 3D real a cena, incluindo fotoes cujo trajeto passa mesmo ENTRE a
    // camara e o buraco negro.
    const double photonR0    = 15.0 * bh.r_s;   // distancia de partida ("no infinito" o suficiente)
    const int    NUM_PHOTONS = 120;

    // Parametro de impacto CRITICO: b_crit = 3*sqrt(3)*M = 1.5*sqrt(3)*r_s (~2.598 r_s).
    // E' o valor exato que separa fotoes CAPTURADOS (b < b_crit) de fotoes DESVIADOS QUE
    // ESCAPAM (b > b_crit). Em b = b_crit existe uma orbita circular instavel - a "esfera
    // de fotoes", r = r_photon = 1.5*r_s - por isso fotoes com b muito perto deste valor
    // dao varias voltas a rodar antes de "decidirem" para que lado vao (efeito "whirl",
    // aqui o analogo para luz do mesmo fenomeno que a materia ja mostrava perto do MBO).
    const double bCrit = 1.5 * std::sqrt(3.0) * bh.r_s;

    const double TAU_PLAYBACK_SPEED   = 15.0;    // fotoes nao tem tempo proprio (tau real);
                                                  // isto acelera so' a REPRODUCAO do
                                                  // parametro afim - ajustar a olho
    const float  photonRespawnDelay    = 1.2f;   // pausa (segundos) depois de desaparecer
    const int    ORBIT_SUBSTEPS        = 12;     // ver justificacao original junto ao RK4:
                                                  // orbitas perto do valor critico mudam phi
                                                  // depressa, um so' passo grande seria grosseiro

    const vec3 photonColor(1.0f, 0.9f, 0.35f);   // amarelo-luz - TODOS os fotoes com a MESMA
                                                  // cor (pedido do utilizador: sao todos luz,
                                                  // nao ha "familias" para distinguir por cor
                                                  // como havia com materia/momento angular)

    std::mt19937 rng(1234);   // seed fixo: a cena e' reprodutivel entre corridas
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    // Vetor unitario uniformemente distribuido na esfera (Y como eixo polar, para
    // condizer com worldUpAxis) - usado como NORMAL do plano orbital de cada fotao.
    auto randomUnitVector = [&]() -> vec3 {
        double z = 2.0 * uni01(rng) - 1.0;
        double theta = 2.0 * M_PI * uni01(rng);
        double s = std::sqrt(std::max(0.0, 1.0 - z*z));
        return vec3(static_cast<float>(s * std::cos(theta)),
                    static_cast<float>(z),
                    static_cast<float>(s * std::sin(theta)));
    };

    // Base ortonormal (e1, e2) perpendicular a n, com uma rotacao extra aleatoria (psi)
    // em torno de n - sem essa rotacao, e1 viria sempre da mesma formula geometrica
    // (cross com um eixo de referencia fixo) e os fotoes tenderiam a comecar todos do
    // mesmo "lado" relativo ao seu proprio plano, perdendo variedade.
    auto buildPlaneBasis = [&](const vec3& n, vec3& e1, vec3& e2) {
        vec3 helper = (std::abs(n.y) < 0.9f) ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
        vec3 base1 = glm::normalize(glm::cross(helper, n));
        vec3 base2 = glm::cross(n, base1);
        float psi = static_cast<float>(2.0 * M_PI * uni01(rng));
        e1 = base1 * std::cos(psi) + base2 * std::sin(psi);
        e2 = -base1 * std::sin(psi) + base2 * std::cos(psi);
    };

    // Distribui os fotoes pelos 3 regimes fisicos possiveis: capturados, "whirl" (banda
    // estreita a volta do valor critico) e desviados-mas-escapam. As fronteiras sao
    // proporcoes fixas de NUM_PHOTONS - nao ha significado fisico nos 45%/65%, so'
    // controlam quantos fotoes de cada tipo se veem na cena.
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
        photons[i].phi0 = 0.0;   // referencia arbitraria dentro do plano - a variedade
                                  // real vem da orientacao aleatoria do proprio plano
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

    // NOVO (C): grelha de referencia, estatica, no plano equatorial - puramente visual.
    // Sem isto, uma imagem parada nao da nenhuma pista de profundidade nenhuma; com
    // linhas a convergir em perspectiva, o olho le a cena como 3D mesmo sem a camara
    // se mexer (util enquanto nao houver camara interativa). Y fica ligeiramente
    // abaixo de 0 (nao exatamente no mesmo plano dos rastos) para evitar z-fighting -
    // duas superficies exatamente coplanares "tremem" no depth buffer.
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

        // --- atualizar cada fotao: integracao da geodesica NULA em parametro afim ---
        for (auto& pv : photons) {
            Photon& ph = pv.photon;
            if (ph.active) {
                double dlambda = static_cast<double>(deltaTime) * TAU_PLAYBACK_SPEED;
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
                    // cruzou o horizonte de eventos
                    ph.active = false;
                    pv.respawnTimer = 0.0f;
                } else if (escaped) {
                    // foi desviado mas b > b_crit: nunca foi capturado, volta a
                    // ultrapassar r0 a subir e afasta-se para sempre
                    ph.active = false;
                    pv.respawnTimer = 0.0f;
                } else {
                    // NOVO (D): a posicao 3D deste fotao ja nao vem de (r cos phi, 0,
                    // r sin phi) fixo no plano XZ - vem de projetar (r, phi), que vivem
                    // no plano PROPRIO deste fotao, na base 3D (planeE1, planeE2) desse
                    // plano. Fotoes com planos muito inclinados passam por cima/baixo do
                    // disco equatorial, ou mesmo entre a camara e o buraco negro - e'
                    // isto que da profundidade 3D real a cena.
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
                // fica desaparecido uma pausa antes de recomecar, para dar tempo a
                // perceber visualmente que foi capturado/escapou - nao e' um efeito
                // fisico, e' so' para tornar a demonstracao legivel em loop
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
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   // NOVO (3D): limpar tambem o depth buffer

        // CAMADA 0: grelha de referencia - desenhada primeiro; o depth test trata de a
        // esconder atras do que estiver mais perto da camara, independentemente da
        // ordem de desenho
        glUseProgram(lineProg);
        glUniform3f(lineColorLoc, 0.3f, 0.3f, 0.38f);
        glBindVertexArray(gridVAO);
        glDrawArrays(GL_LINES, 0, gridVertexCount);

        // CAMADA 1: buraco negro
        glUseProgram(pointProg);
        bh.draw(pointProg, bhVAO, pointColorLoc, pointSizeLoc, bhRadiusPx);

        // CAMADA 2 e 3: rasto e ponto de cada fotao. Todos com a MESMA cor (photonColor)
        // - sao todos luz, o que os distingue e' so' a trajetoria (capturado, whirl
        // perto da esfera de fotoes, ou desviado-e-escapado), nao uma familia de cor.
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
