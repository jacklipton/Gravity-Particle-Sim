// CUDA kernel to add two arrays element-wise
#include "Particle_CUDA.cuh"


__global__ void updateParticles(Particle_CUDA* particles, int numParticles) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numParticles) {
        Particle_CUDA& p = particles[tid];
        float2 totalForce = make_float2(0, 0);
        for (int i = 0; i < numParticles; i++) {
            if (i != tid) {
                Particle_CUDA& other = particles[i];
                float2 force = make_float2(other.pos.x - p.pos.x, other.pos.y - p.pos.y);
                float distSq = force.x * force.x + force.y * force.y;
                distSq = max(1.0f, min(1000.0f, distSq));
                float strength = 0.05f * (1.0f / distSq);
                force.x *= strength;
                force.y *= strength;
                totalForce.x += force.x;
                totalForce.y += force.y;
            }
        }
        p.acc.x += totalForce.x;
        p.acc.y += totalForce.y;
        p.vel.x += p.acc.x;
        p.vel.y += p.acc.y;
        p.pos.x += p.vel.x;
        p.pos.y += p.vel.y;
        p.lifespan -= 0.5f;
    }
}