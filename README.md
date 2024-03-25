

<div id="top"></div>

<h3 align="center">Gravitational Fireworks</h3>

  <p align="center">
    An HCI themed art project running using CUDA displaying the gravitational attraction of particles
    <br />
    <a href="https://github.com/jacklipton/Gravitational-Fireworks-jk
      "><strong>Explore the docs »</strong></a>
    <br />
  </p>
</div>





https://github.com/jacklipton/Gravity-Particle-Sim/assets/83594679/2499265a-112a-4cfa-b2d4-bad43b53462b






<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#contributing">Contributing</a></li>
  </ol>
</details>

<br/>


## About The Project

This project was made for a human-computer interaction course at Queen's University. The user is able to create spheres, each with there own individual gravitational pull, by moving their cursor about the window. Particles are free to move and affect each others trajectory, with an attached vector indicating their direction of acceleration.


### Built With

* [C++](https://gcc.gnu.org/)
* [CUDA](https://developer.nvidia.com/cuda-toolkit)
* [CMake](https://cmake.org/)


## Prerequisites
YOU MUST HAVE A NVIDIA GPU

Before you begin, ensure you have met the following requirements:

* Install the CUDA Toolkit from the [nvidia website](https://developer.nvidia.com/cuda-downloads?target_os=Linux)


* Havegnu 9 installed. Ensure GCC/G++ 9.x is set as the default of your shell

```bash
sudo apt update
sudo apt install gcc-9 g++-9
#add as an alternative version capable to be used
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9
#run the command and then set the version to 9
sudo update-alternatives --config gcc
```

* Install CMake

```bash
sudo apt install cmake
```

* Install SDL2 & SDL2_gfx

```bash
sudo apt install libsdl2-dev
sudo apt install libsdl2-gfx-dev
```

## Configuration

 * Clone this repository

```bash
git clone -b CUDA https://github.com/jacklipton/Gravity-Particle-Sim.git
cd Gravity-Particle-Sim
```
 * Modify the CMakeLists.txt line: set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_61) with your GPU architecture
* Create a build folder and move into it

```bash
mkdir build
cd build
```
* Build and run the program

```bash
cmake ..
make
./Fireworks_with_cuda
```

## Contributing

Contributions are welcome! If you have any improvements, bug fixes, or feature suggestions, feel free to fork the repository and submit a pull request.


