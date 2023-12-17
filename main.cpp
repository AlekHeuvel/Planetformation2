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

const double solarMass = 1.989e30;
const double earthMass = 5.972e24;
const double AU = 149.6e9;
const int n_particles = 133;
const double ROOT_SIZE = 10 * AU;
vector<Vector3d> force = {};

std::random_device rd;  // Obtain a random number from hardware
std::mt19937 gen(rd()); // Seed the generator
std::uniform_real_distribution<> dis_eccentricity(0.0, 0.01); // Eccentricity range
std::uniform_real_distribution<> dis_angle(0.0, 2 * M_PI); // Angle range
std::uniform_real_distribution<> embryo_mass(0.01*earthMass, 0.11*earthMass);

const Vector3d NORMAL_VECTOR(0, 0, 1);

Vector3d get_velocity_elliptical_orbit(const Vector3d& position, double eccentricity) {
    double rand_num = static_cast<double>(rand()) / RAND_MAX;
    double semiMajorAxis = position.norm() / (1 - eccentricity);
    double velocityMagnitude = sqrt(G * solarMass * (2 / position.norm() - 1 / semiMajorAxis));
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

int main() {
    // empty data.csv
    ofstream file("data.csv");
    file.close();

    vector<Particle> particles = {};

    const double inner_radius = 0.5 * AU; // Define the inner radius of the disk
    const double outer_radius = 4 * AU; // Define the outer radius of the disk

    for (int i = 0; i < n_particles; i++) {
        Vector3d pos, vel;
        double mass, radius;
        bool positionOK;

        do {
            // Generate a random number for the power law distribution
            double rand_num = static_cast<double>(rand()) / RAND_MAX;
            //double semiMajorAxis = pow((pow(outer_radius, exponent + 1) - pow(inner_radius, exponent + 1)) * rand_num + pow(inner_radius, exponent + 1), 1.0 / (exponent + 1));
            double dr = 1e9;
            double semiMajorAxis = inner_radius + dr *(pow((rand_num*(sqrt(outer_radius) - sqrt(inner_radius))+sqrt(inner_radius)),2) - inner_radius)/dr;
            // Generate a random angle for uniform distribution around the disk
            double angle = dis_angle(gen);

            // Calculate position in Cartesian coordinates
            pos = Vector3d(semiMajorAxis * cos(angle), semiMajorAxis * sin(angle), 0);

            // Generate a random eccentricity for each particle
            double eccentricity = dis_eccentricity(gen);

            // Calculate velocity for elliptical orbit
            vel = get_velocity_elliptical_orbit(pos, eccentricity);

            mass = embryo_mass(gen);
            radius = pow(3/(4 * M_PI * density) * mass, 1.0/3.0);

            // Check if the position is too close to other particles
            positionOK = !isTooClose(pos, particles, radius*10); // Using 10 * radius as minimum distance
        } while (!positionOK);

        // Add particle to the list with the correct velocity
        particles.push_back(Particle(pos, vel,{}, mass, radius));
    }

    // Creating sun
    Vector3d sunPos = Vector3d::Zero();
    Vector3d sunVel = Vector3d::Zero();
    particles.push_back(Particle(sunPos, sunVel,{}, solarMass, 7e8));

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
    }
    for (int j = 0; j < particles.size(); j++) {
        particles[j].velocity += force[j] / particles[j].mass * 0.5*dt;
    }
    omp_set_nested(1);
    omp_set_num_threads(10);
    for (int i = 0; i < 100000000000; i++) {
        // Initial half-step velocity update
        #pragma omp parallel for
        for (int j = 0; j < particles.size(); j++) {
            particles[j].velocity += force[j] / particles[j].mass * 0.5*dt;
        }

        // Full-step position update
        #pragma omp parallel for
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
        #pragma omp parallel for
        for (int j = 0; j < particles.size(); j++) {
            particles[j].velocity += force[j] / particles[j].mass * 0.5*dt;
        }

        // Additional steps such as collision handling or tree building
        new_root.buildTree(particles, new_root);
        if (i>100000){
            particles = new_root.collideParticles();
        }


        // Time increment
        t += dt;



        if(i % 1000 == 0) {
            cout << i << endl;
            cout << particles.size() << endl;
            saveParticlesToCSV(particles, "data.csv", i);
            //double E;
            //E = 0;
            //for (int j = 0; j < particles.size(); j++) {
            //    E += 0.5 * particles[j].mass * particles[j].velocity.norm();
            //    for (int k = j+1; k < particles.size(); k++) {
            //        E += -(G * particles[j].mass * particles[k].mass)/sqrt((particles[j].position-particles[k].position).norm());
            //    }
            //}
            //cout << E << endl;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "Time taken: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
}
