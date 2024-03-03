//
// Created by Jack on 2024-02-29.
//

#include "Particle.hpp"


Particle::Particle(pair<int, int> pairInput) {
    this->position = pairInput;
    this->velocity = make_pair(genFlt(-1,1),genFlt(-2,2));
    this->acceleration = make_pair(0,0);
    this->lifespan = 255.0;
}

Particle::~Particle() {

}



bool Particle::isDead() const {
    return (lifespan < 0.0);
}

float Particle::genFlt(float min_value, float max_value) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(min_value, max_value);

    return dis(gen);
}



void Particle::setPos(pair<float, float> pairInput) {
    this->position = pairInput;
}
void Particle::setVel(pair<float, float> pairInput) {
    this->velocity = pairInput;
}
void Particle::setAcc(pair<float, float> pairInput) {
    this->acceleration = pairInput;
}
pair<float, float> Particle::getPos() {
    return this->position;
}
pair<float, float> Particle::getVel() {
    return this->velocity;
}
pair<float, float> Particle::getAcc() {
    return this->acceleration;
}
void Particle::setLifespan(double val) {
    this->lifespan = val;
}
int Particle::getLifespan() {
    return this->lifespan;
}
