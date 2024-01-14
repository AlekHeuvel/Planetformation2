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
const int nEmbryos = 10;

//const double embryoMass = embryo_mass_total / nEmbryos;
//const double planetesimalMass = planetesimal_mass_total / nPlanetesimals;
const int nPlanetesimals = 300;
const double planetesimalMass = 0.0222 * earthMass;
const double embryoMass = 15 * planetesimalMass;

const double inclination = 45 * PI / 180;
const double AU = 149.6e9;

const double ROOT_SIZE = 25 * AU;
const double semiMajorAxisSun = 0.25 * AU;
const double inner_radius = 2 * AU; // Define the inner radius of the disk
const double outer_radius = 3 * AU; // Define the outer radius of the disk

vector<Particle> p_list = {};

std::random_device rd;  // Obtain a random number from hardware
std::mt19937 gen(rd()); // Seed the generator
std::uniform_real_distribution<> dis_angle(0.0, 2 * PI); // Angle range

const Vector3d NORMAL_VECTOR(0, 0, 1);

Vector3d rotateAroundXAxis(const Vector3d& original, double angle) {
    double cosA = cos(angle);
    double sinA = sin(angle);

    // Rotate around x-axis
    return Vector3d(
            original.x(), // x doesn't change
            original.y() * cosA - original.z() * sinA, // rotate y and z
            original.y() * sinA + original.z() * cosA
    );
}

Vector3d getVelocityEllipticalOrbitSun(const Vector3d& position, double semiMajorAxis) {
    double velocityMagnitude = sqrt(G * (2.0*solarMass) * (1.0 / (2.0*position.norm()) - 1.0 / (4.0*semiMajorAxis)));
    Vector3d velocity = velocityMagnitude * NORMAL_VECTOR.cross(position).normalized();
    return velocity;
}

Vector3d getVelocityEllipticalOrbit(const Vector3d& position, double semiMajorAxis) {
    double radius = position.norm();
    double velocityMagnitude = sqrt(G * 2.0*solarMass *((2.0/ radius)-(1/semiMajorAxis)));

    // Velocity is perpendicular to the position vector (circular orbit)
    Vector3d velocity = velocityMagnitude * NORMAL_VECTOR.cross(position).normalized();
    return velocity;
}

Vector3d getVelocityCircularOrbit(const Vector3d& position) {
    double radius = position.norm();
    double velocityMagnitude = sqrt(G * 2*solarMass /radius);

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

Vector3d totalMomentum(vector<Particle>& particles) {
    Vector3d totalMomentum = Vector3d::Zero();
    for (const auto& particle : p_list) {
        totalMomentum += particle.mass * particle.velocity;
    }
    return totalMomentum;
}

Particle createParticle(double mass) {
    Vector3d pos, vel;
    double radius;
    double semiMajorAxis, angle;
    double rand_num;

    do {
        // Generate a random num
        // ber for the power law distribution
        rand_num = static_cast<double>(rand()) / RAND_MAX;
        semiMajorAxis = inner_radius +
                        (pow((rand_num*(sqrt(outer_radius) - sqrt(inner_radius)) + sqrt(inner_radius)),2) - inner_radius);

        // Generate a random angle for uniform distribution around the disk
        angle = dis_angle(gen);

        // Calculate position in Cartesian coordinates
        pos = Vector3d(semiMajorAxis * cos(angle), semiMajorAxis * sin(angle), 0);

        radius = pow(3/(4 * PI * density) * mass, 1.0/3.0);

    } while (isTooClose(pos, p_list, radius * 10));

    vel = getVelocityCircularOrbit(pos);

    return Particle(pos, vel, {}, mass, radius);
}

void adjustMomentum() {
    Vector3d totalMomentum = Vector3d::Zero();
    double totalMass = 0.0;

    // Calculate total momentum and mass
    for (auto &particle: p_list) {
        totalMomentum += particle.mass * particle.velocity;
        totalMass += particle.mass;
    }

    // Calculate the velocity correction
    Vector3d velocityCorrection = totalMomentum / totalMass;

    // Apply the correction to all particles
    for (auto &particle: p_list) {
        particle.velocity -= velocityCorrection;
    }

    // Debugging: Check total momentum after adjustment
    totalMomentum = Vector3d::Zero();
    for (const auto &particle: p_list) {
        totalMomentum += particle.mass * particle.velocity;
    }
    cout << "Total Momentum after adjustment: " << totalMomentum.transpose() << endl;
}


int main() {
    // empty data.csv
    ofstream file("data.csv");
    file.close();

    // Create embryos
    for (int i = 0; i < nEmbryos; i++) {
        p_list.push_back(createParticle(embryoMass));
    }
    //Create planetesimals
    for (int i = 0; i < nPlanetesimals; i++) {
        p_list.push_back(createParticle(planetesimalMass));
    }

    for (auto &particle : p_list) {
        particle.position = rotateAroundXAxis(particle.position, inclination);
        particle.velocity = rotateAroundXAxis(particle.velocity, inclination);
    }

    // Creating binary star system
    double eccentricity = 0.5;
    double dis_com_focal_point = semiMajorAxisSun * (1 + eccentricity);
    Vector3d star1Pos = Vector3d(dis_com_focal_point, 0, 0);
    Vector3d star1Vel = getVelocityEllipticalOrbitSun(star1Pos, semiMajorAxisSun);
    Vector3d star2Pos = Vector3d(-1 * dis_com_focal_point, 0, 0);
    Vector3d star2Vel = getVelocityEllipticalOrbitSun(star2Pos, semiMajorAxisSun);
    p_list.push_back(Particle(star1Pos, star1Vel, {}, solarMass, 7e8));
    p_list.push_back(Particle(star2Pos, star2Vel, {}, solarMass, 7e8));

    // Creating single star system

//    Vector3d starPos = Vector3d::Zero();
//    Vector3d starVel = Vector3d::Zero();
//    p_list.push_back(Particle(starPos, starVel, {}, solarMass, 7e8));

    adjustMomentum();

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

    for (auto &particle : p_list) {
        particle.velocity += particle.force / particle.mass * 0.5 * dt;

    }

    omp_set_num_threads(8);
    // Simulation loop
    for (int i = 0; i < 50000; i++) {


#pragma omp parallel for
        for (auto &particle : p_list) {
            particle.velocity += particle.force / particle.mass * 0.5 * dt;
            particle.position += particle.velocity * dt;
        }

        root.rebuildTree(p_list);
        p_list = root.collideParticles();

        // Recalculate forces after position update
        root.rebuildTree(p_list);
#pragma omp parallel for
        for (Particle &particle : p_list) {
            particle.force = root.getForceWithParticle(particle);
            particle.velocity += particle.force / particle.mass * 0.5 * dt;
        }

        // Time increment
        t += dt;

        saveParticlesToCSV(p_list, "data.csv", i);
        if(i % 250 == 0) {

            auto end = chrono::high_resolution_clock::now();
            cout << "Time taken: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        }
    }
}
