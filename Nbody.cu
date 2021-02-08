#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define LY 9460730472580800 // Light-year
#define G 6.67408e-11

#define BLOCK_Z 1
#define BLOCK_Y 1
#define BLOCK_X 1024 
#define GRID_Z 1
#define GRID_Y 1
#define GRID_X 1024
#define TotalPoint BLOCK_X * BLOCK_Y * BLOCK_Z * GRID_X * GRID_Y * GRID_Z
#define BlackHoles 100
#define rool 40

#define dt 10 * 365 * 24 * 3600.0f
#define half_dt dt/2
#define BodiesPerSave TotalPoint*(TotalPoint/ 1000000000.0)*rool

#define Rx_u 300 * LY
#define Ry_u 300 * LY
#define Rz_u 300 * LY
#define Rx_l -300 * LY
#define Ry_l -300 * LY
#define Rz_l -300 * LY

#define Rvx_u 1e6
#define Rvy_u 1e6
#define Rvz_u 1e6
#define Rvx_l -1e6
#define Rvy_l -1e6
#define Rvz_l -1e6

#define Rm_u 1e30
#define Rm_l 1e10

#define fix 1e3

#define flopf (19*TotalPoint + 15)*1e-9
#define flopS flopf*rool*TotalPoint

void GenerateRandomPoints(float4 *Point, float4 *Point_v) {
  srand(time(NULL));
  //Generate random location for points.
  for (int Tv=0; Tv < TotalPoint; Tv++) {
    Point[Tv].x = rand()/(float)RAND_MAX * (Rx_u - Rx_l) + Rx_l;
    Point[Tv].y = rand()/(float)RAND_MAX * (Ry_u - Ry_l) + Ry_l;
    Point[Tv].z = rand()/(float)RAND_MAX * (Rz_u - Rz_l) + Rz_l;
    
    //Point_Gmdt = Point.w
    Point[Tv].w = rand()/(float)RAND_MAX * (Rm_u + Rm_l) * G * dt;
    
    Point_v[Tv].x = rand()/(float)RAND_MAX * (Rvx_u - Rvx_l) + Rvx_l;
    Point_v[Tv].y = rand()/(float)RAND_MAX * (Rvy_u - Rvy_l) + Rvy_l;
    Point_v[Tv].z = rand()/(float)RAND_MAX * (Rvz_u - Rvz_l) + Rvz_l;
  }
  
  for (int Tv=0; Tv < BlackHoles; Tv++) {
    Point[Tv].w = 1e10 * Rm_u * G * dt;
  }
}

void Save(float4 *Point) {
  FILE *save;
  if ((save=fopen("data.data", "a+")) == NULL) {
    printf("Can't save data.\n");
  }
  
  //Data = [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
  fprintf(save, "[");
  //Print P_xs;
  fprintf(save, "[");
  for (int i=0; i < TotalPoint; i++) {
    fprintf(save, "%.2f", Point[i].x);
    if (i != TotalPoint-1)
      fprintf(save, ", ");
  }
  fprintf(save, "]");
  
  //Print P_ys;
  fprintf(save, ", [");
  for (int i=0; i < TotalPoint; i++) {
    fprintf(save, "%.2f", Point[i].y);
    if (i != TotalPoint-1)
      fprintf(save, ", ");
  }
  fprintf(save, "]");
  
  //Print P_zs;
  fprintf(save, ", [");
  for (int i=0; i < TotalPoint; i++) {
    fprintf(save, "%.2f", Point[i].z);
    if (i != TotalPoint-1)
      fprintf(save, ", ");
  }
  fprintf(save, "]");
  fprintf(save, "]\n"); // The end.
  
  fclose(save);
}

__global__ void CaculateTheNextTick(float4 *Point, float4 *Point_v, float4 *T) {
         
  int i =  blockIdx.x * blockDim.x + threadIdx.x; // Get thread's index

  //if (i < TotalPoint) {
    float da_i = 0.0f;
    float rd = 0.0f; 

    float x_i = Point[i].x;
    float y_i = Point[i].y;
    float z_i = Point[i].z;
    
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    
    float da_ix = 0.0f;
    float da_iy = 0.0f;
    float da_iz = 0.0f;
    
    __shared__ float4 Temp[BLOCK_X];
#pragma unroll 
    for (int tile=0; tile < gridDim.x; tile++) { 
      Temp[threadIdx.x] = Point[tile * blockDim.x + threadIdx.x]; 
      __syncthreads(); 
      
      for (int j=0; j<BLOCK_X; j++) {
        dx = Temp[j].x - x_i;//1 flo
        dy = Temp[j].y - y_i;//1 flo
        dz = Temp[j].z - z_i;//1 flo
      
        rd = rsqrtf((dx * dx) + (dy * dy) + (dz * dz) + fix);//7 flo
        da_i = Temp[j].w * rd * rd * rd;//3 flo
        da_ix += dx * da_i;//2 flo
        da_iy += dy * da_i;//2 flo
        da_iz += dz * da_i;//2 flo
      }//total 19 * TotalPoint flo 
      __syncthreads(); 
    }
    
    T[i].x = x_i + Point_v[i].x * dt + half_dt * da_ix;//4 flo
    T[i].y = y_i + Point_v[i].y * dt + half_dt * da_iy;//4 flo
    T[i].z = z_i + Point_v[i].z * dt + half_dt * da_iz;//4 flo
    
    Point_v[i].x = Point_v[i].x + da_ix;//1 flo
    Point_v[i].y = Point_v[i].y + da_iy;//1 flo
    Point_v[i].z = Point_v[i].z + da_iz;//1 flo
  //}//total 15 + 19*TotalPoint flo
}

int main() {
  //Define what we need on CPU.
  float4 *Point;
  
  float4 *Point_v;
  
  Point = (float4 *)malloc(TotalPoint * sizeof(float4));
  Point_v = (float4 *)malloc(TotalPoint * sizeof(float4));
  
  //Define what we need on GPU.
  float4 *GPU_Point;
  
  float4 *GPU_Point_v;

  float4 *GPU_T;
  
  cudaMalloc((void**)&GPU_Point, TotalPoint * sizeof(float4));
  cudaMalloc((void**)&GPU_Point_v, TotalPoint * sizeof(float4));
  cudaMalloc((void**)&GPU_T, TotalPoint * sizeof(float4));
  
  int count = 0;
  float starttime, endtime;
  
  dim3 grid(GRID_X, GRID_Y, GRID_Z); 
  dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

  FILE *save;
  if ((save=fopen("data.data", "w")) == NULL) {
    printf("Can't save data.\n");
  }
  fclose(save);
  //Generate random point.
  GenerateRandomPoints(Point, Point_v);
  cudaMemcpy(GPU_Point, Point, TotalPoint * sizeof(float4), cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_Point_v, Point_v, TotalPoint * sizeof(float4), cudaMemcpyHostToDevice);
  free(Point_v);

  printf("Start calc. N=%d, dt=%f, frame per save=%d\n", TotalPoint, dt, rool);
  while (1==1) {
    count++;
    
    printf("[Save %d]: Computing... ", count);
    //Caculate the location of next tick.
    starttime = clock();
    
    for (int k=0; k < rool; k++) {
      if (k%2) {
        CaculateTheNextTick<<<grid, block>>>(GPU_T, GPU_Point_v, GPU_Point);
      } else {
        CaculateTheNextTick<<<grid, block>>>(GPU_Point, GPU_Point_v, GPU_T);
      }
      //cudaDeviceSynchronize();
      cudaThreadSynchronize();
      //Update the locations of particles.
      //cudaMemcpy(GPU_Point_x, GPU_T_x, size, cudaMemcpyDeviceToDevice);
      //cudaMemcpy(GPU_Point_y, GPU_T_y, size, cudaMemcpyDeviceToDevice);
      //cudaMemcpy(GPU_Point_z, GPU_T_z, size, cudaMemcpyDeviceToDevice);
    } 
    
    endtime = clock();
    cudaMemcpy(Point, GPU_T, TotalPoint * sizeof(float4), cudaMemcpyDeviceToHost);
    printf("Done. %.2lf fps, %.3lf Sps, %.2lf GBps, %.2lf GFLOPS",
            rool / (endtime-starttime)*CLOCKS_PER_SEC,
            1 / (endtime-starttime)*CLOCKS_PER_SEC,
            BodiesPerSave / (endtime-starttime)*CLOCKS_PER_SEC,
            flopS / (endtime-starttime)*CLOCKS_PER_SEC);
    printf(" Saving... ");
    Save(Point);
    printf("Done. \n");
  }
  //fclose(save);
  //General end of C programs.
  return 0;
}
