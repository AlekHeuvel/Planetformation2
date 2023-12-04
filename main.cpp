#include <iostream>
#include "barnes_hut.h"
#include "Eigen/Dense"
#include <chrono>
#include <random>
#include <fstream>
#include <omp.h>
#include <array>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wextra-tokens"
using namespace std;
using namespace Eigen;

double ROOT_SIZE = 150e9;
const Vector3d NORMAL_VECTOR(0, 0, 1);
double solarMass = 1.989e30;
const int n_particles = 1000;
vector<Vector3d>  force = {};
int counter;

Vector3d get_velocity_circular_orbit(const Vector3d& position) {
    double norm = position.norm();
    double velocityMagnitude = sqrt(solarMass * G / pow(norm, 3));
    Vector3d circularVelocity = velocityMagnitude * NORMAL_VECTOR.cross(position);
    return circularVelocity;
}

void saveParticlesToCSV(const vector<Particle>& particles, const string& filename, int timestep) {
    ofstream file(filename, ios::app); // Open in append mode

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    // Write timestep and particle data
    for (const auto& particle : particles) {
        file << timestep << ","
             << particle.position[0] << "," << particle.position[1] << "," << particle.position[2] << ","
             << particle.velocity[0] << "," << particle.velocity[1] << "," << particle.velocity[2] << ","
             << particle.mass << "\n";
    }

    file.close();
}

int main() {
    // empty data.csv
    ofstream file("data.csv");
    file.close();

    uniform_real_distribution<double> unif(-ROOT_SIZE, ROOT_SIZE); // the range of the random numbers -300e9 to 300e9
    default_random_engine re;

    vector<Particle> particles = {};

    // Creating sun
    Vector3d sunPos = Vector3d::Zero();
    Vector3d sunVel = Vector3d::Zero();
    particles.push_back(Particle(sunPos, sunVel, solarMass, 7e8));

    // Creating rest of the particles
    for (int i = 0; i < n_particles-1; i++) {
        Vector3d pos = VectorXd::Random(DIM) * ROOT_SIZE/2 / 2;
        Vector3d vel = get_velocity_circular_orbit(pos);
        particles.push_back(Particle(pos, vel, 6e24, 6e6));
    }

    // Setting up timing and time steps
    double t = 0;
    auto start = chrono::high_resolution_clock::now();
    Vector3d ROOT_LOWER = {-ROOT_SIZE, -ROOT_SIZE, -ROOT_SIZE};
    Vector3d ROOT_UPPER = {ROOT_SIZE, ROOT_SIZE, ROOT_SIZE};
    // Simulation loop

    Node root(ROOT_LOWER, ROOT_UPPER);
    // for particle in particles add to root
    for (const Particle &particle: particles) {
        root.addParticle(particle);
    }
    root.setMass();
    root.setCentreOfMass();

    for (const Particle &particle: particles) {
        force.push_back(root.getForceWithParticle(particle));
//            cout << root.getForceWithParticle(particle) << endl;
    }
    for (int j = 0; j < particles.size(); j++) {
        particles[j].velocity += force[j] / particles[j].mass * dt;
    }
    omp_set_num_threads(10);
    for (int i = 0; i < 10; i++) {
        // Full-step position update
        for (Particle &particle: particles) {
            particle.position += particle.velocity * dt;
        }

        // Recalculate forces

        Node new_root(ROOT_LOWER, ROOT_UPPER);
        for (Particle &particle: particles) {
            new_root.addParticle(particle);
        }
        new_root.setMass();
        new_root.setCentreOfMass();
        #pragma omp parallel for
        for (int j = 0; j < particles.size(); j++) {
            force[j] = new_root.getForceWithParticle(particles[j]);
        }

        // Final half-step velocity update
        for (int j = 0; j < particles.size(); j++) {
            particles[j].velocity += force[j] / particles[j].mass * dt;
        }

        root.collisionDetection();
        t += dt;


        if(i % 1 == 0) {
            cout << i << endl;
            saveParticlesToCSV(particles, "data.csv", i);

        }

    }

    auto end = chrono::high_resolution_clock::now();
    cout << "Time taken: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
}


#pragma clang diagnostic pop