// Particle.cuh
#ifndef PARTICLE_CUH
#define PARTICLE_CUH

//float2 is a struct that contains two floats built in for cuda
struct Particle_CUDA {
    float2 position;
    float2 velocity;
    float2 acceleration;
    float lifespan;
};

__global__ void addArrays(float* a, float* b, float* c, int size);

#endif // PARTICLE_CUH