/*
 *  Nbody simulation data afterprocess program.
 *  Rewrite after a big deletion. 
 *  Resbi 27/11/2024
 * 
 *  The CUDA version, no more to say. 
 *  Resbi 30/11/2024
 */


#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <omp.h> 
#include <time.h>
//#include <unistd.h>
#include <io.h>

// Dont use this.
//#include <cuda_fp16.h>

#ifndef __linux__
#include <windows.h>
#endif


#ifdef __linux__ 

  double __getMillisecond() {
    struct timeval time; 
    gettimeofday(&time, NULL); 
    return (double)(time.tv_sec * 1000.0 + time.tv_usec / 1000.0); 
  }

  #define getMillisecond __getMillisecond
#else
  #define getMillisecond GetTickCount
#endif


#ifdef __linux__
  #define __aligned_alloc aligned_alloc
  #define __aligned_free free
#else
  #define __aligned_alloc(__alignment__, __size__) _aligned_malloc(__size__, __alignment__)
  #define __aligned_free _aligned_free
#endif


#define PRECISION double
#define PRECISION_4 double4
#define STAR_RADIUS (PRECISION)2e15

#define TILE_SIZE 128

// Todo: different precision when rendering
#define PRECISION_RENDERING float

#define __aligned_alloc_cuda(__alignment__, __size__) malloc(__size__)
#define __aligned_free_cuda free

// debugging
//#define DEBUG_ALL

#ifdef DEBUG_ALL
#define DEBUG_MEMORY
#define DEBUG_FILE
#endif

void writeBMP(char *image, 
              const char* filename, 
              int H, 
              int W) {
  // Renormalizing size
  int l = W / 4 * 4;
  // Headers
  int bmi[] = { l*H + 54,0,54,40,W,H,1 | 3 * 8 << 16,0,l*H,0,0,100,0 };

  FILE *fp = fopen(filename, "wb");

#ifdef DEBUG_FILE
  if (fp == NULL) {
    printf("[ERR] File open error in writeBMP!\n\twhile opening %s\n", filename);
  }
#endif

  fprintf(fp, "BM");
  fwrite(&bmi, 52, 1, fp);
  // Write image
  fwrite(image, 3, l*H, fp);
  
  fclose(fp);
  //free(fp); 
}

// return a pointer to image
__global__ void render_CUDA(long image_size_width, 
                            long image_size_hight, 
                            long image_size_length, 
                            PRECISION_RENDERING *image_camera_position, 
                            PRECISION_RENDERING *image_screen_position, 
                            PRECISION_RENDERING *image_screen_basis_w, 
                            PRECISION_RENDERING *image_screen_basis_h, 
                            long data_point_number, 
                            PRECISION_4 *data_point_position, 
                            char *image) {

  unsigned long pixel_index = blockIdx.x * blockDim.x + threadIdx.x; 

  //if (pixel_index < image_size_length) {
    PRECISION_RENDERING pixel_accumulation = 0; 
    long pixel_x = pixel_index % image_size_width - image_size_width / 2; 
    long pixel_y = pixel_index / image_size_width - image_size_hight / 2; 

    PRECISION_RENDERING pixel_position[3]; 
    PRECISION_RENDERING pixel_normal[3]; 
    PRECISION_RENDERING pixel_normal_norm_reverse;

    PRECISION_RENDERING pixel_point_relative[3]; 
    PRECISION_RENDERING pixel_point_distance_reverse; 
    PRECISION_RENDERING pixel_innerproduct; 
    PRECISION_RENDERING pixel_point_angleradius; 
    PRECISION_RENDERING pixel_point_angle; 

    pixel_position[0] = image_screen_position[0] + pixel_x * image_screen_basis_w[0] + pixel_y * image_screen_basis_h[0]; 
    pixel_position[1] = image_screen_position[1] + pixel_x * image_screen_basis_w[1] + pixel_y * image_screen_basis_h[1]; 
    pixel_position[2] = image_screen_position[2] + pixel_x * image_screen_basis_w[2] + pixel_y * image_screen_basis_h[2]; 
    
    pixel_normal[0] = pixel_position[0] - image_camera_position[0]; 
    pixel_normal[1] = pixel_position[1] - image_camera_position[1]; 
    pixel_normal[2] = pixel_position[2] - image_camera_position[2]; 

    pixel_normal_norm_reverse = rsqrtf(pixel_normal[0] * pixel_normal[0] +
                                       pixel_normal[1] * pixel_normal[1] + 
                                       pixel_normal[2] * pixel_normal[2]); 

    pixel_normal[0] = pixel_normal[0] * pixel_normal_norm_reverse; 
    pixel_normal[1] = pixel_normal[1] * pixel_normal_norm_reverse; 
    pixel_normal[2] = pixel_normal[2] * pixel_normal_norm_reverse; 

    // I have no need to do tiling here, 
    // since it has no bottleneck on memory accessing, 
    // so I'm not gonna change it, temporarly. 
    // Tile size must match block dim
    __shared__ PRECISION_4 tile_point_position[TILE_SIZE]; 
    for (long tile_index = 0; tile_index < data_point_number / TILE_SIZE; tile_index++) { 
      __syncthreads(); 
      tile_point_position[threadIdx.x].x = data_point_position[tile_index * TILE_SIZE + threadIdx.x].x; 
      tile_point_position[threadIdx.x].y = data_point_position[tile_index * TILE_SIZE + threadIdx.x].y; 
      tile_point_position[threadIdx.x].z = data_point_position[tile_index * TILE_SIZE + threadIdx.x].z; 
      tile_point_position[threadIdx.x].w = data_point_position[tile_index * TILE_SIZE + threadIdx.x].w; 
      __syncthreads(); 

      for (long point_index = 0; point_index < TILE_SIZE; point_index++) {
        pixel_point_relative[0] = tile_point_position[point_index].x - pixel_position[0]; 
        pixel_point_relative[1] = tile_point_position[point_index].y - pixel_position[1]; 
        pixel_point_relative[2] = tile_point_position[point_index].z - pixel_position[2]; 

      
        pixel_point_distance_reverse = rsqrtf(pixel_point_relative[0] * pixel_point_relative[0] + 
                                              pixel_point_relative[1] * pixel_point_relative[1] +
                                              pixel_point_relative[2] * pixel_point_relative[2]);  

        // A bit slower
        //pixel_point_distance_reverse = rnorm3df(pixel_point_relative[0], pixel_point_relative[1], pixel_point_relative[2]); 
      
        pixel_innerproduct = (pixel_point_relative[0] * pixel_normal[0] + 
                              pixel_point_relative[1] * pixel_normal[1] +
                              pixel_point_relative[2] * pixel_normal[2]); 

        if (pixel_innerproduct > 0) {
          // These are what makes it sloooooow
          pixel_point_angleradius = atanf(STAR_RADIUS * pixel_point_distance_reverse); 
          pixel_point_angle = acosf(pixel_innerproduct * pixel_point_distance_reverse); 

          if (pixel_point_angle < 4 * pixel_point_angleradius) {
            //pixel_accumulation += 1; 
            // This is what makes it sloooooow
            pixel_accumulation += 10 * __expf(- pixel_point_angle * pixel_point_angle / (pixel_point_angleradius * pixel_point_angleradius * 2)); 
          }
          /*
          PRECISION pixel_judge = pixel_point_angle * pixel_point_angle / (2 * pixel_point_angleradius); 
          if (pixel_judge < 1) {
            pixel_accumulation += 10 * (1 - pixel_judge); 
          }
          */
        } 
      }
    }

    /*
    if (pixel_accumulation > 1) {
      printf("(%ld) %f\n", pixel_index, pixel_accumulation);
    }
    */
    
    image[pixel_index * 3 + 0] = (char)(255 * atan(pixel_accumulation) * 2 / 3.1416);
    image[pixel_index * 3 + 1] = (char)(255 * atan(pixel_accumulation) * 2 / 3.1416);
    image[pixel_index * 3 + 2] = (char)(255 * atan(pixel_accumulation) * 2 / 3.1416);
    
  //}
}


// return a pointer to data
PRECISION_4 *readData(char *data_file_name, 
                      long *data_point_number) {
  
  FILE *file = fopen(data_file_name, "rb"); 

  // Failed to open! 
  if (file == NULL) {
    *data_point_number = -1;
#ifdef DEBUG_FILE
    printf("[ERR] File open error in readData()!\n\twhile opening %s\n", data_file_name);
#endif
    return NULL; 
  }

  // Read number of points, in ULL
  unsigned long long temp_data_point_number;

#ifdef DEBUG_FILE
  int debug_file_reading_point_number = fread(&temp_data_point_number, 8, 1, file); 
  if ((debug_file_reading_point_number == 0) || (ferror(file))) {
    printf("[ERR] File read error in readData()!\n\twhile reading data_point_number\n"); 
  }
#else
  fread(&temp_data_point_number, 8, 1, file);  
#endif

  *data_point_number = (long)temp_data_point_number; 

  PRECISION_4 *data_point_position = (PRECISION_4 *)__aligned_alloc(32, sizeof(PRECISION_4) * temp_data_point_number); 

#ifdef DEBUG_MEMORY
  if (data_point_position == NULL) {
    printf("[ERR] Memory alloc error in readData()!\n\twhile allocating data_point_position\n"); 
  }
#endif

#ifdef DEBUG_FILE
  int debug_file_reading_point_position = fread(data_point_position, sizeof(PRECISION_4), temp_data_point_number, file); 
  if ((debug_file_reading_point_position == 0) || (ferror(file))) {
    printf("[ERR] File read error in readData()!\n\twhile reading data_point_position\n"); 
  }
#else
  fread(data_point_position, sizeof(PRECISION_4), temp_data_point_number, file); 
#endif

  // Remember to close them!! 
  fclose(file); 
  return data_point_position; 
} 


int main() {
#ifdef DEBUG_ALL
  printf("[DEBUG] Debugging enabled!\n");
#endif

#ifdef DEBUG_MEMORY
  printf("[DEBUG] DEBUG_MEMORY enabled!\n");
#endif

#ifdef DEBUG_FILE
  printf("[DEBUG] DEBUG_FILE enabled!\n");
#endif
  
  char data_file_name[128]; 
  char data_file_prefix[32] = "datas/131072"; 

  char image_file_name[128]; 
  char image_file_prefix[32] = "images/131072"; 

  long image_size_width = 1920 * 2; 
  long image_size_hight = 1080 * 2; 
  long image_size_length = image_size_width * image_size_hight; 

  PRECISION_RENDERING image_camera_position[3] = {5e18f, 0, 0}; 
  PRECISION_RENDERING image_screen_position[3] = {4.9e18f, 0, 0}; 
  PRECISION_RENDERING image_screen_basis_w[3] = {0, 0.2e18f / image_size_hight * 2, 0}; 
  PRECISION_RENDERING image_screen_basis_h[3] = {0, 0, 0.2e18f / image_size_hight * 2}; 

  // CUDA configurations
  int3 cuda_block_size; 
  int3 cuda_grid_size; 

  cuda_block_size.x = TILE_SIZE; 
  cuda_block_size.y = 1; 
  cuda_block_size.z = 1; 

  cuda_grid_size.x = (int)image_size_length / cuda_block_size.x; 
  cuda_grid_size.y = 1; 
  cuda_grid_size.z = 1; 

  dim3 block(cuda_block_size.x, 
             cuda_block_size.y, 
             cuda_block_size.z); 
  
  dim3 grid(cuda_grid_size.x, 
            cuda_grid_size.y, 
            cuda_grid_size.z); 

  printf("camera_position\n\t%f\n\t%f\n\t%f\n", image_camera_position[0], image_camera_position[1], image_camera_position[2]); 
  printf("screen_position\n\t%f\n\t%f\n\t%f\n", image_screen_position[0], image_screen_position[1], image_screen_position[2]); 
  printf("screen_basis_w\n\t%f\n\t%f\n\t%f\n", image_screen_basis_w[0], image_screen_basis_w[1], image_screen_basis_w[2]); 
  printf("screen_basis_h\n\t%f\n\t%f\n\t%f\n", image_screen_basis_h[0], image_screen_basis_h[1], image_screen_basis_h[2]); 
  

  long data_point_number = 0; 
  long data_point_number_next = 0; 

  // image_index = image_index_base + image_index_offset
  // image_index_offset is the counter. 
  int image_index_offset = 0;
  int image_index_base = 418; 
  int image_index = 0; 

  double begin_ms; 
  double end_ms; 
  double delta_ms; 

  while (data_point_number >= 0) { 
    image_index = image_index_base + image_index_offset; 

    PRECISION_RENDERING *cuda_image_camera_position;// = {5e18f, 0, 0}; 
    PRECISION_RENDERING *cuda_image_screen_position;// = {3e18f, 0, 0}; 
    PRECISION_RENDERING *cuda_image_screen_basis_w;// = {0, 3e18f / image_size_hight * 2, 0}; 
    PRECISION_RENDERING *cuda_image_screen_basis_h;// = {0, 0, 3e18f / image_size_hight * 2}; 

    cudaMalloc((void**)&cuda_image_camera_position, 
               sizeof(PRECISION_RENDERING) * 3); 
    cudaMalloc((void**)&cuda_image_screen_position, 
               sizeof(PRECISION_RENDERING) * 3); 
    cudaMalloc((void**)&cuda_image_screen_basis_w, 
               sizeof(PRECISION_RENDERING) * 3); 
    cudaMalloc((void**)&cuda_image_screen_basis_h, 
               sizeof(PRECISION_RENDERING) * 3); 

    PRECISION_4 *data_point_position; 
    PRECISION_4 *data_point_position_next; 
    char *image = (char *)__aligned_alloc(32, sizeof(char) * image_size_length * 3); 
    char *image_previous; 

#ifdef DEBUG_MEMORY
    if (image == NULL) {
      printf("[ERR] Memory alloc error in main()!\n\twhile allocating image\n"); 
    }
#endif

    PRECISION_4 *cuda_data_point_position;
    PRECISION_4 *cuda_data_point_position_next;
    char *cuda_image;
    char *cuda_image_previous; 
    cudaMalloc((void**)&cuda_image, 
               sizeof(char) * image_size_length * 3); 

    begin_ms = getMillisecond(); 

    // The initial frame. 
    if (image_index_offset == 0) {
      sprintf(data_file_name, "%s,%d.nbody", data_file_prefix, image_index); 
      printf("[%d] Reading %s\n", image_index, data_file_name); 

      data_point_position = readData(data_file_name, 
                                     &data_point_number); 

      cudaMalloc((void**)&cuda_data_point_position, 
                 sizeof(PRECISION_4) * data_point_number);


      cudaMemcpy(cuda_data_point_position, 
                 data_point_position, 
                 sizeof(PRECISION_4) * data_point_number, 
                 cudaMemcpyHostToDevice); 
    }

    /*
    for (int i = 0; i < 10; i++) {
      printf("%d\n\t%.2f\n\t%.2f\n\t%.2f\n\t%.2f\n", i, 
                                                     data_point_position[i].x, 
                                                     data_point_position[i].y, 
                                                     data_point_position[i].z, 
                                                     data_point_position[i].w); 
    }
    */

    printf("[%d] Transformming data...\n", image_index); 

    cudaMemcpy(cuda_image_camera_position, 
               image_camera_position, 
               sizeof(PRECISION_RENDERING) * 3, 
               cudaMemcpyHostToDevice); 
  
    cudaMemcpy(cuda_image_screen_position, 
               image_screen_position, 
               sizeof(PRECISION_RENDERING) * 3, 
               cudaMemcpyHostToDevice);  
  
    cudaMemcpy(cuda_image_screen_basis_w, 
               image_screen_basis_w, 
               sizeof(PRECISION_RENDERING) * 3, 
               cudaMemcpyHostToDevice);  
  
    cudaMemcpy(cuda_image_screen_basis_h, 
               image_screen_basis_h, 
               sizeof(PRECISION_RENDERING) * 3, 
               cudaMemcpyHostToDevice); 

    /*
    image = render(image_size_width, 
                   image_size_hight, 
                   image_camera_position, 
                   image_screen_position, 
                   image_screen_basis_w, 
                   image_screen_basis_h, 
                   data_point_number, 
                   data_point_position);
    */

    // Asynchronously read and write files. 
    if (image_index_offset >= 0) {
      printf("[%d] Rendering...\n", image_index);
      
      render_CUDA<<<grid, block>>>(image_size_width, 
                                   image_size_hight, 
                                   image_size_length, 
                                   cuda_image_camera_position, 
                                   cuda_image_screen_position, 
                                   cuda_image_screen_basis_w, 
                                   cuda_image_screen_basis_h, 
                                   data_point_number, 
                                   cuda_data_point_position, 
                                   cuda_image);
     
      // Read data for the next frame
      sprintf(data_file_name, "%s,%d.nbody", data_file_prefix, image_index + 1); 
      printf("[%d] Reading %s\n", image_index + 1, data_file_name); 

      data_point_position_next = readData(data_file_name, 
                                          &data_point_number_next); 

      cudaMalloc((void**)&cuda_data_point_position_next, 
                 sizeof(PRECISION_4) * data_point_number);

      printf("[%d] Transformming data...\n", image_index + 1);

      cudaMemcpyAsync(cuda_data_point_position_next, 
                      data_point_position_next, 
                      sizeof(PRECISION_4) * data_point_number_next, 
                      cudaMemcpyHostToDevice); 
    }

    // Save the previous frame 
    if (image_index_offset > 0) {
      sprintf(image_file_name, "%s-%d.bmp", image_file_prefix, image_index - 1); 
      printf("[%d] Saving image to %s\n", image_index - 1, image_file_name);


      /*
      // Why would this slower??
      cudaMemcpyAsync(image_previous, 
                      cuda_image_previous, 
                      sizeof(char) * image_size_length * 3, 
                      cudaMemcpyDeviceToHost); 
      */
      
      writeBMP(image_previous, 
               image_file_name, 
               image_size_hight, 
               image_size_width); 
    }

    cudaThreadSynchronize();
    cudaDeviceSynchronize();

    cudaMemcpy(image, 
               cuda_image, 
               sizeof(char) * image_size_length * 3, 
               cudaMemcpyDeviceToHost); 
    
    end_ms = getMillisecond();
    delta_ms = end_ms - begin_ms; 

    printf("[%d] Frame rendering finished! Cost %.2f s\n", image_index, delta_ms / 1000); 

    __aligned_free(data_point_position); 

    cudaFree(cuda_image_camera_position); 
    cudaFree(cuda_image_screen_position); 
    cudaFree(cuda_image_screen_basis_w); 
    cudaFree(cuda_image_screen_basis_h); 

    cudaFree(cuda_data_point_position);
    
    if (image_index_offset > 0) {
      __aligned_free(image_previous);  
      cudaFree(cuda_image_previous); 
    }

    data_point_number = data_point_number_next; 
    cuda_data_point_position = cuda_data_point_position_next; 
    data_point_position = data_point_position_next; 

    image_previous = image; 
    cuda_image_previous = cuda_image; 

    image_index_offset++;

#ifdef DEBUG_FILE
    int debug_num_closed = _fcloseall( ); 
   
    if (debug_num_closed > 0) { 
      printf("[ERR] You remains %d files opened!\n", debug_num_closed); 
    }
#endif
  }

  return 0; 
}
