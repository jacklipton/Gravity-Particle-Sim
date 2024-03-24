#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>
#include <vector>
#include <utility>
#include <iostream>

#include <cuda_runtime.h>
#include <chrono>

#include "Particle.hpp"
#include "Particle_CUDA.cuh"

using namespace std;



vector<Particle> particles;



void update(SDL_Renderer *rndr);
void applyAttraction(Particle *prtcle);

int main(int argc, char* args []) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("Gravitational Fireworks", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_SHOWN);
    if (window == nullptr) {
        SDL_Log("Failed to create window: %s", SDL_GetError());
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == nullptr) {
        SDL_Log("Failed to create renderer: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    bool running = true;
    bool press = false;
    SDL_Event event;

    Uint32 startTime = SDL_GetTicks(); // Get the initial time

    while (running) {
        // Close window with any input
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
                break;
            }
            else if(event.type == SDL_MOUSEBUTTONDOWN){ press = true; }
            else if(event.type == SDL_MOUSEBUTTONUP){ press = false; }

        }

        // Update at 60 FPS
        Uint32 currentTime = SDL_GetTicks(); // Get the current time
        Uint32 deltaTime = currentTime - startTime; // Calculate time passed since the last frame

        if (deltaTime >= 32) { // 16ms per frame = 60 FPS
            if (press == true){
                int x, y;
                SDL_GetMouseState(&x, &y);
                particles.emplace_back(make_pair(x,y));
                particles.emplace_back(make_pair(x,y));
            }




            if (particles.size() > 0){  

                // cudaEvent_t start, stop;
                // cudaEventCreate(&start);
                // cudaEventCreate(&stop);
    
                // cudaEventRecord(start);
                update(renderer);
                // cudaEventRecord(stop);
                // cudaDeviceSynchronize();

                // float milliseconds = 0;
                // cudaEventElapsedTime(&milliseconds, start, stop);

                // cout << "Elapsed time (GPU): " << milliseconds << " ms" << endl;

            }
            startTime = currentTime; // Reset the start time for the next frame
        }
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

void update(SDL_Renderer *rndr){


    SDL_SetRenderDrawColor(rndr, 0, 0, 0, 255); // Set the background color to purple
    
    unsigned long int numBytes = particles.size() * sizeof(Particle);
    
    Particle_CUDA* d_particles;
    cudaError_t err1 = cudaMalloc((void**)&d_particles, numBytes);

    cudaError_t err2  = cudaMemcpy(d_particles,particles.data(), numBytes, cudaMemcpyHostToDevice);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        cout << "Failed to allocate device memory or copy to device" << endl;
    return;
    }

    
    int blockSize = 256;
    int numBlocks = (particles.size() + blockSize - 1) / blockSize;
    updateParticles<<<numBlocks, blockSize>>>(d_particles, particles.size());

    
    SDL_RenderClear(rndr);
    // for(int i = particles.size()-1; i>= 0; i--){
    //     Particle& p = particles.at(i);
    //     p.setAcc(make_pair(0,0));
    //     applyAttraction(&p);
    //     p.setVel(make_pair(p.getVel().first + p.getAcc().first, p.getVel().second + p.getAcc().second));
    //     p.setPos(make_pair(p.getPos().first + p.getVel().first, p.getPos().second + p.getVel().second)); //deal w/ pairs
    //     p.setLifespan(p.getLifespan() - 0.5);
    //     filledCircleRGBA(rndr,p.getPos().first, p.getPos().second, 8, 255,255,255, p.getLifespan());

    //     if(p.getLifespan() <=0){
    //         particles.erase(particles.begin()+i);
    //     }
    // }

            // Wait for the kernel to finish

    // After the kernel has finished...
    cudaMemcpy(particles.data(), d_particles, particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {
        cout << "Failed to copy device memory to host or to launch/update particles: " << cudaGetErrorString(err3) << endl;
        return;
    }
    


    //cout << particles.at(0).getPos().first << endl;
    cudaFree(d_particles);
    //cout << "Freed device memory" << endl;

    for(int i = particles.size()-1; i>= 0; i--){
        Particle& p = particles.at(i);
        if(p.getLifespan() <=0){
            particles.erase(particles.begin()+i);
        }else{
            filledCircleRGBA(rndr,p.getPos().first, p.getPos().second, 8, 255,255,255, p.getLifespan());
        }

    }

    SDL_RenderPresent(rndr); // Render changes onto the screen
}

// void applyAttraction(Particle *prtcle) {
//     for (Particle& other : particles) {
//         if (&other != prtcle) {
//             pair<float, float> force = make_pair(other.getPos().first - prtcle->getPos().first, other.getPos().second - prtcle->getPos().second);
//             double distSq = (force.first*force.first) + (force.second*force.second);
//             // Constrain the variable between 1 and 1000
//             distSq = (distSq < 1) ? 1 : ((distSq > 1000) ? 1000 : distSq);

//             double strength = 0.05 * (1.0 / distSq);
//             force.first *= strength;
//             force.second *= strength;


//             prtcle->setAcc(make_pair(prtcle->getAcc().first + force.first, prtcle->getAcc().second + force.second));
//         }
//     }
// }