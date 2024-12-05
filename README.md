# An N-Body simulation program using all-pair algorithm in CUDA

![](131072-1152.bmp)

# How to use

## Simulation `Nbody.cu`

### Compiling

Run `make.sh` on Linux or `make.bat` on Windows, make sure you've added `nvcc` compiler to your `PATH` environment variable. 

### Running 

First create a folder `datas` in the same directory of `Nbody.cu`, then run `./Nbody.out` on Linux or `./Nbody.exe` on Windows. 

### Do your own configuration 

#### - Particle Number

In `Nbody.cu`, on the top of the code, find `#define GRID_X [particle number] / BLOCK_X`, change the `[particle number]` to the number you want. 

**Notice that particle number must be divisible by block size `BLOCK_X`.** 

#### - Time Step

In `Nbody.cu`, on the top of the code, find `#define dt [time step]`, change the `[time step]` to the number you want. Time unit is `second`. 

#### - Smoothing Distance 

In `Nbody.cu`, on the top of the code, find `#define fix [smooth distance]`, change the `[smooth distance]` to the number you want. Length unit is `meter`. 

#### - Calculating Precision 

In `Nbody.cu`, on the top of the code, find `#define PRECISION [precision]` and `#define PRECISION [precision4]`, change them into `float` and `float` for fp32, or `double` and `double4` for fp64. **FP16 wasn't tested**. 

#### - Steps Per Save 

In `Nbody.cu`, on the top of the code, find `#define rool [steps]`, change the `[steps]` to the number you want. I recommend `5` or `10` per save with _small_ (well, pretty big on the number itself) time step. 

#### - Initial Distribution 

In `Nbody.cu`, change the function `void GenerateRandomPoints()` into whatever you want. Coordinate in unit `meter`. 


## Rendering `rendering.cu` 

### Compiling 

Run `make_rendering.cuda.sh` on Linux or `make_rendering.cuda.bat` on Windows, make sure you've added `nvcc` compiler to your `PATH` environment variable. 

### Running 

Before using it, please check the keyword "Particle Number" under the **next** section "Do your own configuration". 

First create a folder `images` in the same directory of `rendering.cu`, then run `./rendering.cuda.out` on Linux or `./rendering.cuda.exe` on Windows.

### Do your own configuration 

#### - Particle Number

It's **necessary** to match the particle number with your simulation. 

In `rendering.cu`, in function `int main()`, find `char data_file_prefix[32] = "datas/[particle number]";` and `char image_file_prefix[32] = "images/[particle number]";`, change the `[particle number]` to the same as your simulation. If your forgot it, check `datas/[particle number],[step].nbody` for it. 

#### - Resolution 

In `rendering.cu`, in function `int main()`, find `long image_size_width = [image width];` and `long image_size_hight = [image hight];`, change `[image width] x [image hight]` to the resolution you want. 

#### - Rendering Float Point Precision 

It's the calculating precision of floating point number during rendering. 

In `rendering.cu`, on the top of the code, find `#defind PRECISION_RENDERING [precision rendering]`, change the `[precision rendering]` to the fp type that fits you. **FP16 wasn't tested**. 

#### - Camera Configuration 

In `rendering.cu`, in function `int main()`, we have 

`PRECISION_RENDERING image_camera_position[3] = {5e18f, 0, 0};`

`PRECISION_RENDERING image_screen_position[3] = {4.9e18f, 0, 0}; `

`PRECISION_RENDERING image_screen_basis_w[3] = {0, 0.2e18f / image_size_hight * 2, 0}; `

`PRECISION_RENDERING image_screen_basis_h[3] = {0, 0, 0.2e18f / image_size_hight * 2};`

those are camera configurations, documention on them is a Todo. 

#### - Which Frame To Begin With 

In `rendering.cu`, in function `int main()`, find `int image_index_base = [start frame]`, change the `[start frame]` to the number of which frame you want to start with. 
 

## Todos:
- ~~Rewrite the datasaving codes.~~ Finished 2024/12/01
- ~~Rewrite the rendering code. (Maybe in C instead of Python.)~~ Finished 2024/12/01
- ~~Change the rendering method. (Maybe something more mathematical, but not that mathematical.)~~ Finished 2024/12/01
- Rechoose a unit system, then rewrite the whole `Nbody.cu`.
- Do a better rendering algorithm.
- Support continuing calculation from a break point. 
- Support computing on multiple devices. (Since I haven't multiple GPUs in one PC, this might take some times.(sad)) 
- Support distributed computing. (Emmm.) 
