#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>
#include <vector>
#include <utility>
#include <iostream>
#include "Particle.hpp"

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
            }
            update(renderer);
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
    SDL_RenderClear(rndr);
    for(int i = particles.size()-1; i>= 0; i--){
        Particle& p = particles.at(i);
        p.setAcc(make_pair(0,0));
        applyAttraction(&p);
        p.setVel(make_pair(p.getVel().first + p.getAcc().first, p.getVel().second + p.getAcc().second));
        p.setPos(make_pair(p.getPos().first + p.getVel().first, p.getPos().second + p.getVel().second)); //deal w/ pairs
        p.setLifespan(p.getLifespan() - 0.5);
        filledCircleRGBA(rndr,p.getPos().first, p.getPos().second, 8, 255,255,255, p.getLifespan());

        if(p.getLifespan() <=0){
            particles.erase(particles.begin()+i);
        }
    }


    SDL_RenderPresent(rndr); // Render changes onto the screen
}

void applyAttraction(Particle *prtcle) {
    for (Particle& other : particles) {
        if (&other != prtcle) {
            pair<float, float> force = make_pair(other.getPos().first - prtcle->getPos().first, other.getPos().second - prtcle->getPos().second);
            double distSq = (force.first*force.first) + (force.second*force.second);
            // Constrain the variable between 1 and 1000
            distSq = (distSq < 1) ? 1 : ((distSq > 1000) ? 1000 : distSq);

            double strength = 0.05 * (1.0 / distSq);
            force.first *= strength;
            force.second *= strength;


            prtcle->setAcc(make_pair(prtcle->getAcc().first + force.first, prtcle->getAcc().second + force.second));

        }
    }
}