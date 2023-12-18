#include <iostream>
#include <random>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <cmath>
#include <set>
#include "Eigen/Dense"
#include "barnes_hut.h"

using namespace std;
using namespace Eigen;

// Define all variables
const int n_particles = 0;
const double solarMass = 1.989e30;
const double earthMass = 5.972e24;
const double AU = 149.6e9;
const double ROOT_SIZE = 10 * AU;
const double inner_radius = 0.5 * AU; // Define the inner radius of the disk
const double outer_radius = 4 * AU; // Define the outer radius of the disk

vector<Particle> p_list = {};

std::random_device rd;  // Obtain a random number from hardware
std::mt19937 gen(rd()); // Seed the generator
std::uniform_real_distribution<> dis_eccentricity(0.0, 0.01); // Eccentricity range
std::uniform_real_distribution<> dis_angle(0.0, 2 * PI); // Angle range
std::uniform_real_distribution<> embryo_mass(0.01*earthMass, 0.11*earthMass);

const Vector3d NORMAL_VECTOR(0, 0, 1);

Vector3d get_velocity_elliptical_orbit(const Vector3d& position, double eccentricity, double mass) {
    double semiMajorAxis = position.norm() / (1 - eccentricity);
    semiMajorAxis = 0.2 * AU;
    double velocityMagnitude = sqrt(G * (solarMass + mass) * (2 / position.norm() - 1 / semiMajorAxis)) / 2;
    Vector3d velocity = velocityMagnitude * NORMAL_VECTOR.cross(position).normalized();

    return velocity;
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

bool isTooClose(const Vector3d& pos, const vector<Particle>& particles, double minDistance) {
    for (const auto& particle : particles) {
        if ((particle.position - pos).norm() < minDistance) {
            return true;
        }
    }
    return false;
}

double kineticEnergy(vector<Particle>& particles) {
    double E = 0;
    for (auto& particle : particles) {
        E += 0.5 * particle.mass * particle.velocity.squaredNorm();
    }
    return E;
}

double potentialEnergy(vector<Particle>& particles) {
    double E = 0;
    for (int i = 0; i < particles.size(); i++) {
        for (int j = i+1; j < particles.size(); j++) {
            E += -(G * particles[i].mass * particles[j].mass)/(particles[i].position-particles[j].position).norm();
        }
    }
    return E;
}

double totalEnergy(vector<Particle>& particles) {
    return kineticEnergy(particles) + potentialEnergy(particles);
}

Vector3d angularMomentum(vector<Particle>& particles) {
    Vector3d L = Vector3d::Zero();
    for (auto& particle : particles) {
        L += particle.mass * particle.position.cross(particle.velocity);
    }
    return L;
}

Particle createParticle() {
    Vector3d pos, vel;
    double mass, radius;
    bool positionOK;

    do {
        // Generate a random number for the power law distribution
        double rand_num = static_cast<double>(rand()) / RAND_MAX;
        double dr = 1e9;
        double semiMajorAxis = inner_radius +
                dr * (pow((rand_num*(sqrt(outer_radius) - sqrt(inner_radius)) + sqrt(inner_radius)),2) - inner_radius)/dr;

        // Generate a random angle for uniform distribution around the disk
        double angle = dis_angle(gen);

        // Calculate position in Cartesian coordinates
        pos = Vector3d(semiMajorAxis * cos(angle), semiMajorAxis * sin(angle), 0);

        mass = embryo_mass(gen);
        radius = pow(3/(4 * PI * density) * mass, 1.0/3.0);

        // Check if the position is too close to other p_list
        positionOK = !isTooClose(pos, p_list, radius * 10); // Using 10 * radius as minimum distance
    } while (!positionOK);

    // Generate a random eccentricity for each particle
    double eccentricity = dis_eccentricity(gen);

    // Calculate velocity for elliptical orbit
    vel = get_velocity_elliptical_orbit(pos, eccentricity, mass);

    return Particle(pos, vel, {}, mass, radius);
    }

int main() {
    // empty data.csv
    ofstream file("data.csv");
    file.close();

    // Create p_list
    for (int i = 0; i < n_particles; i++) {
        p_list.push_back(createParticle());
    }

    // Creating binary star system
    double semi_major = 0.2 * AU;
    double eccentricity = 0.5;
    double dis_com_focal_point = semi_major * (1 + eccentricity);
    Vector3d star1Pos = Vector3d(dis_com_focal_point * cos(0), dis_com_focal_point * sin(0), 0);
    Vector3d star1Vel = get_velocity_elliptical_orbit(star1Pos, 0.5, solarMass);
//    Vector3d star1Vel = Vector3d::Zero();
    Vector3d star2Pos = Vector3d(dis_com_focal_point * cos(PI), dis_com_focal_point * sin(PI), 0);
    Vector3d star2Vel = get_velocity_elliptical_orbit(star2Pos, 0.5, solarMass);
//    Vector3d star2Vel = Vector3d::Zero();
    p_list.push_back(Particle(star1Pos, star1Vel, {}, solarMass, 7e8));
    p_list.push_back(Particle(star2Pos, star2Vel, {}, solarMass, 7e8));

    // Setting up timing and time steps
    double t = 0;
    auto start = chrono::high_resolution_clock::now();

    Vector3d ROOT_LOWER = {-ROOT_SIZE, -ROOT_SIZE, -ROOT_SIZE};
    Vector3d ROOT_UPPER = {ROOT_SIZE, ROOT_SIZE, ROOT_SIZE};
    Node root(ROOT_LOWER, ROOT_UPPER);
    root.rebuildTree(p_list);

    for (const Particle &particle: p_list) {
        particle.force = root.getForceWithParticle(particle);
    }
    for (Particle &particle : p_list) {
        particle.velocity += particle.force / particle.mass * dt;
    }

    omp_set_num_threads(10);
    // Simulation loop
    for (int i = 0; i < 50000; i++) {

        // Initial half-step velocity update
        #pragma omp parallel for
        for (auto & particle : p_list) {
            particle.velocity += particle.force / particle.mass * 0.5 * dt;
        }

        // Full-step position update
        #pragma omp parallel for
        for (Particle &particle: p_list) {
            particle.position += particle.velocity * dt;
        }

        root.rebuildTree(p_list);

        #pragma omp parallel for
        for (auto & particle : p_list) {
            particle.force = root.getForceWithParticle(particle);
        }

        // Final half-step velocity update
        #pragma omp parallel for
        for (auto & particle : p_list) {
            particle.velocity += particle.force / particle.mass * 0.5 * dt;
        }

        // Recalculate forces
        root.rebuildTree(p_list);
        #pragma omp parallel for
        for (Particle& particle : p_list) {
            particle.force = root.getForceWithParticle(particle);
        }

        // Final half-step velocity update
        for (auto & particle : p_list) {
            particle.velocity += particle.force / particle.mass * 0.5 * dt;
        }

//        p_list = root.collideParticles();
        t += dt;

        saveParticlesToCSV(p_list, "data.csv", i);
        if(i % 50 == 0) {
            cout << i << endl;
            cout << "Total energy: " << totalEnergy(p_list) << endl;
            cout << "Angular momentum: " << angularMomentum(p_list) << endl;

            auto end = chrono::high_resolution_clock::now();
            cout << "Time taken: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        }
    }

}
