//
// Created by Jack on 2024-02-29.
//

#ifndef FIREWORKS_PARTICLE_HPP
#define FIREWORKS_PARTICLE_HPP

#include <utility>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Particle {

public:

    explicit Particle(pair<int, int> pairInput);
    ~Particle();

    void applyAttraction(vector<Particle> *particle_array);
    bool isDead() const;

    //setter and getter
    void setPos(pair<float, float> pairInput);
    void setVel(pair<float, float> pairInput);
    void setAcc(pair<float, float> pairInput);
    void setLifespan(double val);

    pair<float, float> getPos();
    pair<float, float> getVel();
    pair<float, float> getAcc();
    int getLifespan();





private:
    pair<float, float> position;
    pair<float, float> velocity;
    pair<float, float> acceleration;
    double lifespan;

    float genFlt(float min, float max);


};


#endif //FIREWORKS_PARTICLE_HPP
