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
const double solarMass = 1.989e30;
const double earthMass = 5.972e24;
const int n_embryos = 300;
const double embryo_mass_total =15.0 * earthMass;
const double embryo_mass = embryo_mass_total / n_embryos;
const int n_planetesimals = 10;
const double planetesimal_mass_total = 0.0222 * earthMass;
const double planetesimal_mass = planetesimal_mass_total / n_planetesimals;

const double AU = 149.6e9;
const double ROOT_SIZE = 10 * AU;
const double inner_radius = 1 * AU; // Define the inner radius of the disk
const double outer_radius = 5 * AU; // Define the outer radius of the disk

vector<Particle> p_list = {};

std::random_device rd;  // Obtain a random number from hardware
std::mt19937 gen(rd()); // Seed the generator
std::uniform_real_distribution<> dis_angle(0.0, 2 * PI); // Angle range

const Vector3d NORMAL_VECTOR(0, 0, 1);

Vector3d get_velocity_elliptical_orbit_sun(const Vector3d& position, double semiMajorAxis) {
    double velocityMagnitude = sqrt(G * (2.0*solarMass) * (1.0 / (2.0*position.norm()) - 1.0 / (4.0*semiMajorAxis)));
    Vector3d velocity = velocityMagnitude * NORMAL_VECTOR.cross(position).normalized();
    return velocity;
}

Vector3d get_velocity_elliptical_orbit(const Vector3d& position, double semiMajorAxis) {
    double radius = position.norm();
    double velocityMagnitude = sqrt(G * 2.0*solarMass *((2.0/ radius)-(1/semiMajorAxis)));

    // Velocity is perpendicular to the position vector (circular orbit)
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

Particle createParticle(const int i) {
    Vector3d pos, vel;
    double mass, radius;
    bool positionOK;
    double semiMajorAxis;

    do {
        // Generate a random number for the power law distribution
        double rand_num = static_cast<double>(rand()) / RAND_MAX;
        semiMajorAxis = inner_radius +
                        (pow((rand_num*(sqrt(outer_radius) - sqrt(inner_radius)) + sqrt(inner_radius)),2) - inner_radius);

        // Generate a random angle for uniform distribution around the disk
        double angle = dis_angle(gen);

        // Calculate position in Cartesian coordinates
        pos = Vector3d(semiMajorAxis * cos(angle), semiMajorAxis * sin(angle), 0);

        if (i == 0){
            mass = embryo_mass;
        }
        else {
            mass = planetesimal_mass;
        }

        radius = pow(3/(4 * PI * density) * mass, 1.0/3.0);

        vel = get_velocity_elliptical_orbit(pos,semiMajorAxis);
        cout << pos.norm() << endl;
        cout << vel.norm() << endl;

        // Check if the position is too close to other p_list
        positionOK = !isTooClose(pos, p_list, radius * 10); // Using 10 * radius as minimum distance
    } while (!positionOK);



    return Particle(pos, vel, {}, mass, radius);
}

int main() {
    // empty data.csv
    ofstream file("data.csv");
    file.close();

    // Create embryos
    for (int i = 0; i < n_embryos; i++) {
        p_list.push_back(createParticle(0));
    }
    // Create planetesimals
    //for (int i = 0; i < n_planetesimals; i++) {
    //    p_list.push_back(createParticle(1));
    //}

    // Creating binary star system
    double semi_major = 0.2 * AU;
    double eccentricity = 0.5;
    double dis_com_focal_point = semi_major * (1 + eccentricity);
    Vector3d star1Pos = Vector3d(dis_com_focal_point, 0, 0);
    Vector3d star1Vel = get_velocity_elliptical_orbit_sun(star1Pos, semi_major);
    Vector3d star2Pos = Vector3d(-1 * dis_com_focal_point, 0, 0);
    Vector3d star2Vel = get_velocity_elliptical_orbit_sun(star2Pos, semi_major);
    p_list.push_back(Particle(star1Pos, star1Vel, {}, solarMass, 7e8));
    p_list.push_back(Particle(star2Pos, star2Vel, {}, solarMass, 7e8));

    // Setting up timing and time steps
    double t = 0;
    auto start = chrono::high_resolution_clock::now();

    Vector3d ROOT_LOWER = {-ROOT_SIZE, -ROOT_SIZE, -ROOT_SIZE};
    Vector3d ROOT_UPPER = {ROOT_SIZE, ROOT_SIZE, ROOT_SIZE};
    Node root(ROOT_LOWER, ROOT_UPPER);
    root.rebuildTree(p_list);

    // Initial force calculation
    for (const Particle &particle: p_list) {
        particle.force = root.getForceWithParticle(particle);
    }

    omp_set_num_threads(10);
    // Simulation loop
    for (int i = 0; i < 50000; i++) {
        // Initial half-step velocity update
#pragma omp parallel for
        for (auto &particle : p_list) {
            particle.velocity += particle.force / particle.mass * 0.5 * dt;
        }

        // Full-step position update
#pragma omp parallel for
        for (Particle &particle : p_list) {
            particle.position += particle.velocity * dt;
        }

        // Recalculate forces after position update
        root.rebuildTree(p_list);
#pragma omp parallel for
        for (Particle &particle : p_list) {
            particle.force = root.getForceWithParticle(particle);
        }

        // Final half-step velocity update
#pragma omp parallel for
        for (auto &particle : p_list) {
            particle.velocity += particle.force / particle.mass * 0.5 * dt;
        }

        if (i > 500){
            p_list = root.collideParticles();
        }


        // Time increment
        t += dt;

        saveParticlesToCSV(p_list, "data.csv", i);
        if(i % 250 == 0) {
            cout << i << endl;
            cout << p_list.size() << endl;
            //cout << "Total energy: " << totalEnergy(p_list) << endl;
            //cout << "Angular momentum: " << angularMomentum(p_list) << endl;

            auto end = chrono::high_resolution_clock::now();
            cout << "Time taken: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        }
    }

}