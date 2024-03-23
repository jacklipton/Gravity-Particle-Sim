// Particle.cuh
#ifndef PARTICLE_CUH
#define PARTICLE_CUH

//float2 is a struct that contains two floats built in for cuda
struct Particle_CUDA {
    float2 pos;
    float2 vel;
    float2 acc;
    double lifespan;
};

__global__ void updateParticles(Particle_CUDA* particles, int numParticles);

#endif // PARTICLE_CUH