#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define LY 9460730472580800 // Light-year
#define G 6.67408e-11

#define TPB 1024 // TPB = Threads Per Block.
#define TotalPoint TPB*256
#define BlackHoles 0
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
#define Rm_l 1e5

#define fix 1e3

#define flopf (28*TotalPoint + 15)*1e-9
#define flopS flopf*rool*TotalPoint

void GenerateRandomPoints(float *Point_x, float *Point_y, float *Point_z, float *Point_Gmdt,
                          float *Point_vx, float *Point_vy, float *Point_vz) {
  srand(time(NULL));
  //Generate random location for points.
  for (int Tv=0; Tv < TotalPoint; Tv++) {
    Point_x[Tv] = rand()/(float)RAND_MAX * (Rx_u - Rx_l) + Rx_l;
    Point_y[Tv] = rand()/(float)RAND_MAX * (Ry_u - Ry_l) + Ry_l;
    Point_z[Tv] = rand()/(float)RAND_MAX * (Rz_u - Rz_l) + Rz_l;
    
    Point_Gmdt[Tv] = rand()/(float)RAND_MAX * (Rm_u + Rm_l) * G * dt;
    
    Point_vx[Tv] = rand()/(float)RAND_MAX * (Rvx_u - Rvx_l) + Rvx_l;
    Point_vy[Tv] = rand()/(float)RAND_MAX * (Rvy_u - Rvy_l) + Rvy_l;
    Point_vz[Tv] = rand()/(float)RAND_MAX * (Rvz_u - Rvz_l) + Rvz_l;
  }
  
  for (int Tv=0; Tv < BlackHoles; Tv++) {
    Point_Gmdt[Tv] = 1e5 * Point_Gmdt[Tv];
    /*
    Point_Gmdt[Tv] = G * Rm_u * 5e9 * dt;
    Point_vx[Tv] = 0;
    Point_vy[Tv] = 0;
    Point_vz[Tv] = 0;
    Point_x[Tv] = 0;
    Point_y[Tv] = 0;
    Point_z[Tv] = 0;
    */
  }
}

void Draw(float *Point_x, float *Point_y, float *Point_z) {
  FILE *save;
  if ((save=fopen("data.data", "a+")) == NULL) {
    printf("Can't save data.\n");
  }
  
  //Data = [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
  fprintf(save, "[");
  //Print P_xs;
  fprintf(save, "[");
  for (int i=0; i < TotalPoint; i++) {
    fprintf(save, "%.2f", Point_x[i]);
    if (i != TotalPoint-1)
      fprintf(save, ", ");
  }
  fprintf(save, "]");
  
  //Print P_ys;
  fprintf(save, ", [");
  for (int i=0; i < TotalPoint; i++) {
    fprintf(save, "%.2f", Point_y[i]);
    if (i != TotalPoint-1)
      fprintf(save, ", ");
  }
  fprintf(save, "]");
  
  //Print P_zs;
  fprintf(save, ", [");
  for (int i=0; i < TotalPoint; i++) {
    fprintf(save, "%.2f", Point_z[i]);
    if (i != TotalPoint-1)
      fprintf(save, ", ");
  }
  fprintf(save, "]");
  fprintf(save, "]\n"); // The end.
  
  fclose(save);
}

__global__ void CaculateTheNextTick(float *Point_x, float *Point_y, float *Point_z, float *Point_Gmdt,
                                    float *Point_vx, float *Point_vy, float *Point_vz,
                                    float *T_x, float *T_y, float *T_z) {
         
  int i =  blockIdx.x * blockDim.x + threadIdx.x; // Get thread's index

  if (i < TotalPoint) {
    float da_i = 0.0f;
    float rR = 0.0f;
    
    float x_i = Point_x[i];
    float y_i = Point_y[i];
    float z_i = Point_z[i];
    
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    
    float da_ix = 0.0f;
    float da_iy = 0.0f;
    float da_iz = 0.0f;
    
    for (int j=0; j<TotalPoint; j++) {
      dx = Point_x[j] - x_i;//1 flops
      dy = Point_y[j] - y_i;//1 flops
      dz = Point_z[j] - z_i;//1 flops
      rR = rsqrtf((dx * dx) + (dy * dy) + (dz * dz) + fix); //6+10 flops
      da_i = Point_Gmdt[j] * rR * rR * rR;//3 flops
      da_ix += dx * da_i;//2 flops
      da_iy += dy * da_i;//2 flops
      da_iz += dz * da_i;//2 flops
    }//total 28 * TotalPoint flops
    T_x[i] = x_i + Point_vx[i] * dt + half_dt * da_ix;//4 flops
    T_y[i] = y_i + Point_vy[i] * dt + half_dt * da_iy;//4 flops
    T_z[i] = z_i + Point_vz[i] * dt + half_dt * da_iz;//4 flops
    Point_vx[i] = Point_vx[i] + da_ix;//1 flops
    Point_vy[i] = Point_vy[i] + da_iy;//1 flops
    Point_vz[i] = Point_vz[i] + da_iz;//1 flops
  }//total 15 flops
}

int main() {
  int size = TotalPoint * sizeof(float *);

  //Define what we need on CPU.
  float *Point_x;
  float *Point_y;
  float *Point_z;
  
  float *Point_vx;
  float *Point_vy;
  float *Point_vz;
  
  float *Point_Gmdt;
  
  Point_x = (float *)malloc(size);
  Point_y = (float *)malloc(size);
  Point_z = (float *)malloc(size);
  Point_vx = (float *)malloc(size);
  Point_vy = (float *)malloc(size);
  Point_vz = (float *)malloc(size);
  Point_Gmdt = (float *)malloc(size);
  
  //Define what we need on GPU.
  float *GPU_Point_x;
  float *GPU_Point_y;
  float *GPU_Point_z;
  
  float *GPU_Point_vx;
  float *GPU_Point_vy;
  float *GPU_Point_vz;

  float *GPU_T_x;
  float *GPU_T_y;
  float *GPU_T_z;
  
  float *GPU_Point_Gmdt;
  
  cudaMalloc((void**)&GPU_Point_x, size);
  cudaMalloc((void**)&GPU_Point_y, size);
  cudaMalloc((void**)&GPU_Point_z, size);
  cudaMalloc((void**)&GPU_Point_vx, size);
  cudaMalloc((void**)&GPU_Point_vy, size);
  cudaMalloc((void**)&GPU_Point_vz, size);
  cudaMalloc((void**)&GPU_T_x, size);
  cudaMalloc((void**)&GPU_T_y, size);
  cudaMalloc((void**)&GPU_T_z, size);
  cudaMalloc((void**)&GPU_Point_Gmdt, size);
  
  int count = 0;
  float starttime, endtime;
  
  FILE *save;
  if ((save=fopen("data.data", "w")) == NULL) {
    printf("Can't save data.\n");
  }
  fclose(save);
  //Generate random point.
  GenerateRandomPoints(Point_x, Point_y, Point_z, Point_Gmdt,
                       Point_vx, Point_vy, Point_vz);
  cudaMemcpy(GPU_Point_Gmdt, Point_Gmdt, size, cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_Point_vx, Point_vx, size, cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_Point_vy, Point_vy, size, cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_Point_vz, Point_vz, size, cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_Point_x, Point_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_Point_y, Point_y, size, cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_Point_z, Point_z, size, cudaMemcpyHostToDevice);
  free(Point_Gmdt); // FREEDOM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  free(Point_vx);
  free(Point_vy);
  free(Point_vz);

  printf("Start calc. N=%d, dt=%f, frame per save=%d\n", TotalPoint, dt, rool);
  while (1==1) {
    count++;
    
    printf("[Save %d]: Calculating. ", count);
    //Caculate the location of next tick.
    starttime = clock();
    for (int k=0; k < rool; k++) {
      if (k%2) {
        CaculateTheNextTick<<<(TotalPoint+TPB-1)/TPB, TPB>>>(GPU_T_x, GPU_T_y, GPU_T_z, GPU_Point_Gmdt,
                                                             GPU_Point_vx, GPU_Point_vy, GPU_Point_vz,
                                                             GPU_Point_x, GPU_Point_y, GPU_Point_z);
      } else {
        CaculateTheNextTick<<<(TotalPoint+TPB-1)/TPB, TPB>>>(GPU_Point_x, GPU_Point_y, GPU_Point_z, GPU_Point_Gmdt,
                                                             GPU_Point_vx, GPU_Point_vy, GPU_Point_vz,
                                                             GPU_T_x, GPU_T_y, GPU_T_z);
      }
      //cudaDeviceSynchronize();
      cudaThreadSynchronize();
      //Update the locations of particles.
      //cudaMemcpy(GPU_Point_x, GPU_T_x, size, cudaMemcpyDeviceToDevice);
      //cudaMemcpy(GPU_Point_y, GPU_T_y, size, cudaMemcpyDeviceToDevice);
      //cudaMemcpy(GPU_Point_z, GPU_T_z, size, cudaMemcpyDeviceToDevice);
    }
    endtime = clock();
    cudaMemcpy(Point_x, GPU_T_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Point_y, GPU_T_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Point_z, GPU_T_z, size, cudaMemcpyDeviceToHost);
    printf("Done. %.2lf fps, %.3lf Sps, %.2lf GBps, %.2lf GFLOPS",
            rool / (endtime-starttime)*CLOCKS_PER_SEC,
            1 / (endtime-starttime)*CLOCKS_PER_SEC,
            BodiesPerSave / (endtime-starttime)*CLOCKS_PER_SEC,
            flopS / (endtime-starttime)*CLOCKS_PER_SEC);
    printf(" Drawing. \n");
    Draw(Point_x, Point_y, Point_z);
    
  }
  //fclose(save);
  //General end of C programs.
  return 0;
}
