#include "Eigen/Dense"
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;
using namespace Eigen;

const int dt = 4e5;
const int DIM = 3;
const double G = 6.67408e-11;
const double THETA = 0.5; // Radians
const double density = 3e3; //bulk density kg/m^3
const double PI = 3.14159265358979323846264338;

struct Particle {
    mutable Vector3d position;
    mutable Vector3d velocity;
    mutable Vector3d force;
    double mass;
    double radius;
    bool operator==(const Particle& other) const = default;
};

class Node {
public:
    Node(Vector3d lowerBound, Vector3d upperBound, int idx = 0)
            : lowerBound(lowerBound), upperBound(upperBound),
              size(pow(2, 1.0/3) * (upperBound[0] - lowerBound[0])),
              mass(0), centreOfMass(Vector3d::Zero()), particles({}), children({}) {}

    void rebuildTree(vector<Particle>& p_list) {
        // Add all p_list to the new tree
        particles.clear();
        children.clear();

        for (const Particle& particle : p_list) {
            addParticle(particle);
        }

        // Set the mass and center of mass for the new tree
        setMass();
        setCentreOfMass();
    }

    Node& createChild(Vector3d lower, Vector3d upper) {
        Node child(std::move(lower), std::move(upper));
        children.emplace_back(child);
        return children.back();
    }

    Node& getChild(Vector3d point) {
        for (Node& child : children) {
            if (child.isInside(point)) {
                return child;
            }
        }
        Vector3d mean = (lowerBound + upperBound) / 2;
        Vector3d newLower, newUpper;
        for (int i = 0; i < DIM; ++i) {
            if (point[i] < mean[i]) {
                newLower[i] = lowerBound[i];
                newUpper[i] = mean[i];
            } else {
                newLower[i] = mean[i];
                newUpper[i] = upperBound[i];
            }
        }
        return createChild(newLower, newUpper);
    }

    bool isInside(const Vector3d& point) const {
        for (int i = 0; i < DIM; ++i) {
            if (point[i] < lowerBound[i] || point[i] > upperBound[i]) {
                return false;
            }
        }
        return true;
    }

    void addParticle(const Particle& particle) {
        particles.push_back(particle);

        // When there is more than one particle in a node, we split the node
        if (particles.size() >= 2) {
            // Since the first particle to enter the node won't generate a child node, we need to add it to a child
            // when a second particle is created
            if (particles.size() == 2) {
                Particle first_particle = particles[0];
                Node& child = getChild(first_particle.position);
                child.addParticle(first_particle);
            }
            Node& child = getChild(particle.position);
            child.addParticle(particle);
        }
    }

    void setMass() {
        mass = 0;
        for (const Particle& particle : particles) {
            mass += particle.mass;
        }
        for (Node& child : children) {
            child.setMass();
        }
    }

    void setCentreOfMass() {
        centreOfMass.setZero();
        for (const Particle& particle : particles) {
            centreOfMass += particle.mass * particle.position;
        }
        centreOfMass /= mass;
        for (Node& child : children) {
            child.setCentreOfMass();
        }
    }

    double getAngleWithParticle(const Particle& particle) const {
        double distance = (particle.position - centreOfMass).norm();
        return size / distance;
    }

    Vector3d getForceWithParticle(const Particle& particle) {
        double angle = getAngleWithParticle(particle);
        bool inNode = find(particles.begin(), particles.end(), particle) != particles.end();
        if (inNode || (angle > THETA && !children.empty())) {
            Vector3d force = Vector3d::Zero();
            for (Node& child : children) {
                force += child.getForceWithParticle(particle);
            }
            return force;
        } else {
            Vector3d r = particle.position - centreOfMass;
            double distance = r.norm();

            // Softening factor (epsilon), choose an appropriate value
            const double epsilon = 1e8; // Example value, adjust as needed

            // Modified gravitational force equation with softening
            return -G * mass * particle.mass * r / (pow(distance * distance + epsilon * epsilon, 1.5));
        }
    }

    static Particle mergeParticles(const Particle& p1, const Particle& p2) {
        double totalMass = p1.mass + p2.mass;
        Vector3d newPosition = ((p1.position+p1.velocity*dt) * p1.mass + (p2.position+p2.velocity*dt) * p2.mass) / totalMass;
        Vector3d newVelocity = ((p1.force + p2.force) * dt + p1.velocity * p1.mass + p2.velocity * p2.mass) / totalMass;

        // Assuming radius is proportional to the cube root of mass
        double newRadius = pow(3/(4 * PI * density) * totalMass, 1.0/3.0);

        return Particle(newPosition, newVelocity, {}, totalMass, newRadius);
    }

    vector<Particle> collideParticles() {
        vector<Particle> newParticles;

        if (all_of(children.begin(), children.end(), [](Node& child) { return child.particles.size() != 1; })) {
            for (Node& child : children) {
                vector<Particle> childParticles = child.collideParticles();
                newParticles.insert(newParticles.end(), childParticles.begin(), childParticles.end());
            }
        } else {
            // Copy particles to a temporary list for safe iteration and modification
            vector<Particle> tempParticles = particles;

            while (!tempParticles.empty()) {
                Particle currentParticle = tempParticles.back();
                tempParticles.pop_back();
                bool collided = false;

                for (auto it = tempParticles.begin(); it != tempParticles.end(); ) {
                    if (haveCollided(currentParticle, *it)) {
                        currentParticle = mergeParticles(currentParticle, *it);
                        it = tempParticles.erase(it); // Erase and get next iterator
                        collided = true;
                    } else {
                        ++it;
                    }
                }

                if (collided) {
                    // If collision occurred, add the merged particle
                    newParticles.push_back(currentParticle);
                } else {
                    // If no collision, add the original particle
                    newParticles.push_back(currentParticle);
                }
            }
        }

        return newParticles;
    }


    static bool haveCollided(const Particle& particle1, const Particle& particle2) {
        Vector3d dx = particle1.position - particle2.position;
        Vector3d dv = particle1.velocity - particle2.velocity;

        double dxNormSquared = dx.squaredNorm();
        double dvNormSquared = dv.squaredNorm();

        // Using the sum of radii directly in the calculation
        double radiiSumSquared = pow(particle1.radius + particle2.radius, 2);

        // Calculate the closest approach
        double d = (dxNormSquared*dvNormSquared - pow(dx.dot(dv), 2)) / dvNormSquared;
        if (d < 0) {
            return false;
        }

        // Check if the distance at the closest approach is less than the sum of the radii
        if (d < radiiSumSquared) {
            // Additional check for the p_list moving towards each other
            if (-dx.dot(dv) > 0) {
                // ?
                if (-dx.dot(dv)/dvNormSquared < dt){
                    cout << "collision" << endl;
                    return true;
                }
            }
        }
        return false;
    }

private:
    int idx;
    Vector3d lowerBound;
    Vector3d upperBound;
    double size;
    double mass;
    Vector3d centreOfMass;
    vector<Particle> particles;
    vector<Node> children;
};