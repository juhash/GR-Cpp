#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

using glm::vec3;
using glm::mat4;

const double G = 1.0;
const double c = 1.0;
const double kWorldScale = 0.035;

const char* pointVertexSrc = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uMVP;
uniform float uPointSize;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;
}
)";

const char* pointFragmentSrc = R"(
#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    FragColor = vec4(uColor, 1.0);
}
)";

const char* lineVertexSrc = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

const char* lineFragmentSrc = R"(
#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
uniform float uAlpha;
void main() {
    FragColor = vec4(uColor, uAlpha);
}
)";

GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, log);
        std::cerr << "Shader compile error:\n" << log << "\n";
    }
    return shader;
}

GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(prog, 1024, nullptr, log);
        std::cerr << "Program link error:\n" << log << "\n";
    }
    return prog;
}

struct BlackHole {
    vec3 position;
    double mass;
    double r_s;

    BlackHole(vec3 pos, double m) : position(pos), mass(m) {
        r_s = 2.0 * G * mass / (c * c);
    }
};

struct State3D {
    double r;
    double theta;
    double phi;
};

struct Particle {
    State3D state;
    double E;
    double L;
    double Lz;
    bool active = true;
    vec3 posCart;
};

struct ParticleVisual {
    Particle particle;
    State3D initialState;
    vec3 color;
    std::vector<vec3> trail;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint trailVAO = 0;
    GLuint trailVBO = 0;
};

vec3 sphericalToCartesian(double r, double theta, double phi, double scale) {
    const double st = std::sin(theta);
    const double ct = std::cos(theta);
    const double cp = std::cos(phi);
    const double sp = std::sin(phi);
    return vec3(
        static_cast<float>(scale * r * st * cp),
        static_cast<float>(scale * r * ct),
        static_cast<float>(scale * r * st * sp)
    );
}

double radialDrDtau(double r, double r_s, double E, double L) {
    const double term = E * E - (1.0 - r_s / r) * (1.0 + (L * L) / (r * r));
    return -std::sqrt(std::max(0.0, term));
}

State3D rhsSchwarzschild(const State3D& s, double r_s, double E, double L, double Lz) {
    const double r = s.r;
    const double theta = s.theta;
    const double sinTheta = std::sin(theta);
    const double sin2 = sinTheta * sinTheta;
    const double r2 = r * r;

    const double dr = radialDrDtau(r, r_s, E, L);

    // Conserved angular momentum magnitude L and z-component Lz satisfy
    // L^2 = r^4 (theta_dot^2 + sin^2(theta) phi_dot^2)
    // and Lz = r^2 sin^2(theta) phi_dot.
    // This gives the physically consistent angular evolution.
    const double angularTerm = (L * L - (Lz * Lz) / sin2) / (r2 * r2);
    const double dtheta = std::sqrt(std::max(0.0, angularTerm));
    const double dphi = Lz / (r2 * sin2);

    return {dr, dtheta, dphi};
}

State3D rk4StepState(const State3D& s, double dtau, double r_s, double E, double L, double Lz) {
    const auto k1 = rhsSchwarzschild(s, r_s, E, L, Lz);
    const State3D s2{ s.r + 0.5 * dtau * k1.r, s.theta + 0.5 * dtau * k1.theta, s.phi + 0.5 * dtau * k1.phi };
    const auto k2 = rhsSchwarzschild(s2, r_s, E, L, Lz);
    const State3D s3{ s.r + 0.5 * dtau * k2.r, s.theta + 0.5 * dtau * k2.theta, s.phi + 0.5 * dtau * k2.phi };
    const auto k3 = rhsSchwarzschild(s3, r_s, E, L, Lz);
    const State3D s4{ s.r + dtau * k3.r, s.theta + dtau * k3.theta, s.phi + dtau * k3.phi };
    const auto k4 = rhsSchwarzschild(s4, r_s, E, L, Lz);

    return {
        s.r + (dtau / 6.0) * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r),
        s.theta + (dtau / 6.0) * (k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta),
        s.phi + (dtau / 6.0) * (k1.phi + 2.0 * k2.phi + 2.0 * k3.phi + k4.phi)
    };
}

int main() {
    if (!glfwInit()) {
        std::cerr << "glfwInit failed\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    const int winWidth = 1000;
    const int winHeight = 800;
    GLFWwindow* window = glfwCreateWindow(winWidth, winHeight, "Schwarzschild 3D", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "glewInit failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    int fbWidth = 0, fbHeight = 0;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);

    GLuint vsPoint = compileShader(GL_VERTEX_SHADER, pointVertexSrc);
    GLuint fsPoint = compileShader(GL_FRAGMENT_SHADER, pointFragmentSrc);
    GLuint pointProg = linkProgram(vsPoint, fsPoint);
    glDeleteShader(vsPoint);
    glDeleteShader(fsPoint);

    GLuint vsLine = compileShader(GL_VERTEX_SHADER, lineVertexSrc);
    GLuint fsLine = compileShader(GL_FRAGMENT_SHADER, lineFragmentSrc);
    GLuint lineProg = linkProgram(vsLine, fsLine);
    glDeleteShader(vsLine);
    glDeleteShader(fsLine);

    GLint uMVPPoint = glGetUniformLocation(pointProg, "uMVP");
    GLint uColorPoint = glGetUniformLocation(pointProg, "uColor");
    GLint uPointSizePoint = glGetUniformLocation(pointProg, "uPointSize");

    GLint uMVPLine = glGetUniformLocation(lineProg, "uMVP");
    GLint uColorLine = glGetUniformLocation(lineProg, "uColor");
    GLint uAlphaLine = glGetUniformLocation(lineProg, "uAlpha");

    BlackHole bh(vec3(0.0f), 1.0);
    const double r_s = bh.r_s;

    const int NUM_PARTICLES = 8;
    const double particleR0 = 10.0 * r_s;
    const double TAU_PLAYBACK_SPEED = 12.0;
    const float particlePointSizePx = 4.0f;
    const float trailAlpha = 0.45f;
    const size_t maxTrailPoints = 2500;

    const std::vector<double> particleLValues = {
        2.4 * r_s,
        2.6 * r_s,
        2.8 * r_s,
        3.0 * r_s,
        3.2 * r_s,
        3.4 * r_s,
        3.6 * r_s,
        3.8 * r_s
    };

    std::vector<ParticleVisual> particles(NUM_PARTICLES);

    GLuint bhVAO = 0;
    GLuint bhVBO = 0;
    glGenVertexArrays(1, &bhVAO);
    glGenBuffers(1, &bhVBO);
    glBindVertexArray(bhVAO);
    glBindBuffer(GL_ARRAY_BUFFER, bhVBO);
    const vec3 bhPoint = vec3(0.0f, 0.0f, 0.0f);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vec3), &bhPoint, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)0);
    glEnableVertexAttribArray(0);

    for (int i = 0; i < NUM_PARTICLES; ++i) {
        ParticleVisual& pv = particles[i];
        const double phi0 = 0.2 * (i - (NUM_PARTICLES - 1) / 2.0);
        const double theta0 = 1.15 + 0.03 * i;
        const double L0 = particleLValues[i];
        const double Lz0 = 0.82 * L0;

        pv.initialState = {particleR0, theta0, phi0};
        pv.particle.state = pv.initialState;
        pv.particle.E = 1.0;
        pv.particle.L = L0;
        pv.particle.Lz = Lz0;
        pv.particle.active = true;
        pv.particle.posCart = sphericalToCartesian(pv.particle.state.r, pv.particle.state.theta, pv.particle.state.phi, kWorldScale);
        pv.color = vec3(0.65f + 0.02f * i, 0.78f, 1.0f);
        pv.trail.reserve(maxTrailPoints);

        glGenVertexArrays(1, &pv.vao);
        glGenBuffers(1, &pv.vbo);
        glBindVertexArray(pv.vao);
        glBindBuffer(GL_ARRAY_BUFFER, pv.vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec3), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)0);
        glEnableVertexAttribArray(0);

        glGenVertexArrays(1, &pv.trailVAO);
        glGenBuffers(1, &pv.trailVBO);
        glBindVertexArray(pv.trailVAO);
        glBindBuffer(GL_ARRAY_BUFFER, pv.trailVBO);
        glBufferData(GL_ARRAY_BUFFER, maxTrailPoints * sizeof(vec3), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)0);
        glEnableVertexAttribArray(0);
    }

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        const double now = glfwGetTime();
        const float deltaTime = static_cast<float>(now - lastTime);
        lastTime = now;

        for (auto& pv : particles) {
            Particle& p = pv.particle;
            if (p.active) {
                const double dtau = static_cast<double>(deltaTime) * TAU_PLAYBACK_SPEED;
                const State3D nextState = rk4StepState(p.state, dtau, r_s, p.E, p.L, p.Lz);
                p.state = nextState;

                if (p.state.r <= r_s) {
                    p.active = false;
                } else {
                    p.posCart = sphericalToCartesian(p.state.r, p.state.theta, p.state.phi, kWorldScale);
                    pv.trail.push_back(p.posCart);
                    if (pv.trail.size() > maxTrailPoints) {
                        pv.trail.erase(pv.trail.begin());
                    }

                    glBindBuffer(GL_ARRAY_BUFFER, pv.vbo);
                    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec3), &p.posCart);
                }
            }

            glBindBuffer(GL_ARRAY_BUFFER, pv.trailVBO);
            glBufferData(GL_ARRAY_BUFFER, pv.trail.size() * sizeof(vec3), pv.trail.empty() ? nullptr : pv.trail.data(), GL_DYNAMIC_DRAW);
        }

        glClearColor(0.01f, 0.01f, 0.03f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);

        const float aspect = static_cast<float>(fbWidth) / static_cast<float>(fbHeight);
        const mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 200.0f);
        const mat4 view = glm::lookAt(vec3(0.0f, 0.5f, 6.5f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
        const mat4 mvp = projection * view;

        glUseProgram(pointProg);
        glUniformMatrix4fv(uMVPPoint, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform3f(uColorPoint, 1.0f, 0.15f, 0.15f);
        glUniform1f(uPointSizePoint, 20.0f);

        glBindVertexArray(0);
        glBindVertexArray(0);

        glUseProgram(lineProg);
        glUniformMatrix4fv(uMVPLine, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform3f(uColorLine, 0.75f, 0.85f, 1.0f);
        glUniform1f(uAlphaLine, trailAlpha);

        for (const auto& pv : particles) {
            if (!pv.trail.empty()) {
                glBindVertexArray(pv.trailVAO);
                glBindBuffer(GL_ARRAY_BUFFER, pv.trailVBO);
                glBufferData(GL_ARRAY_BUFFER, pv.trail.size() * sizeof(vec3), pv.trail.data(), GL_DYNAMIC_DRAW);
                glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(pv.trail.size()));
            }
        }

        glUseProgram(pointProg);
        glUniformMatrix4fv(uMVPPoint, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform3f(uColorPoint, 0.8f, 0.9f, 1.0f);
        glUniform1f(uPointSizePoint, particlePointSizePx);

        for (const auto& pv : particles) {
            if (pv.particle.active) {
                glBindVertexArray(pv.vao);
                glBindBuffer(GL_ARRAY_BUFFER, pv.vbo);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec3), &pv.particle.posCart);
                glDrawArrays(GL_POINTS, 0, 1);
            }
        }

        glUseProgram(pointProg);
        glUniformMatrix4fv(uMVPPoint, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform3f(uColorPoint, 1.0f, 0.0f, 0.0f);
        glUniform1f(uPointSizePoint, 32.0f);
        glBindVertexArray(bhVAO);
        glDrawArrays(GL_POINTS, 0, 1);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
