#include "Eigen/Dense"
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;
using namespace Eigen;

const int dt = 1e5;
const int DIM = 3;
const double G = 6.67408e-11;
const double THETA = 1; // Radians

struct Particle {
    mutable Vector3d position;
    mutable Vector3d velocity;
    const double mass;
    const double radius;
    bool operator==(const Particle& other) const = default;
};

class Node {
public:
    Node(Vector3d lowerBound, Vector3d upperBound)
            : lowerBound(lowerBound), upperBound(upperBound),
              size(pow(2, 1.0/3) * (upperBound[0] - lowerBound[0])),
              mass(0), centreOfMass(Vector3d::Zero()), particles({}), children({}) {}

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
            return -G * mass * particle.mass * r / pow(distance, 3);
        }
    }

    void collisionDetection() {
        for (Node& child : children) {
            if (child.particles.size() > 1) {
                for (int i = 0; i < particles.size(); i++) {
                    for (int j = i; j < particles.size(); j++) {
                        particleCollision(particles[i], particles[j]);
                        }
                    }
                break;
                }
            else {
                child.collisionDetection();
            }
        }
    }

    static bool particleCollision(Particle& particle1, Particle& particle2) {
        Vector3d dx = particle1.position - particle2.position;
        Vector3d dv = particle1.velocity - particle2.velocity;

        double dxNormSquared = dx.squaredNorm();
        double dvNormSquared = dv.squaredNorm();

        // Using the sum of radii directly in the calculation
        double radiiSum = particle1.radius + particle2.radius;
        double radiiSumSquared = radiiSum * radiiSum;

        double dxdv = dx.dot(dv);
        // Directly using multiplied terms in the discriminant calculation
        double discriminant = dxdv * dxdv - dxNormSquared * (dvNormSquared * radiiSumSquared - dxNormSquared);

        if (discriminant < 0) {
            return false; // No collision possible
        }
        double discriminantRoot = sqrt(discriminant);
        double denominator = dvNormSquared * radiiSumSquared - 2 * dxdv;

        if (denominator == 0) {
            return false; // Avoid division by zero
        }

        double t12 = -discriminantRoot / denominator;

        if (t12 <= 0 || t12 >= dt) {
            return false; // No collision within the time step
        }

        // Collision detected
        cout << "Collision detected at t = " << t12 << endl;
        return true;
        // Handle collision here}
}


private:
    Vector3d lowerBound;
    Vector3d upperBound;
    double size;
    double mass;
    Vector3d centreOfMass;
    vector<Particle> particles;
    vector<Node> children;
};
