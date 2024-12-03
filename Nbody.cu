#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
//#include <unistd.h>

#ifndef __linux__
#include <windows.h>
#endif

#ifdef __linux__ 
//#include <sys/time.h>
  double __getMillisecond() {
    struct timeval time; 
    gettimeofday(&time, NULL); 
    return (double)(time.tv_sec * 1000.0 + time.tv_usec / 1000.0); 
  }

#define getMillisecond __getMillisecond
#else
#define getMillisecond GetTickCount
#endif

#define pi 3.141592653589793238462643383279502884197169399
#define tau 2 * pi
#define LY 9460730472580800 // Light-year
#define G 6.67408e-11

#define TILE_SIZE 128 

#define BLOCK_Z 1
#define BLOCK_Y 1
#define BLOCK_X 64
#define GRID_Z 1
#define GRID_Y 1
#define GRID_X 524288 / BLOCK_X
#define TotalPoint BLOCK_X * BLOCK_Y * BLOCK_Z * GRID_X * GRID_Y * GRID_Z
#define rool 50

#define dt 10 * 365 * 24 * 3600.0f
#define half_dt dt/2
#define BodiesPerSave (long long)TotalPoint*(TotalPoint/ 1000000000.0)*rool

#define fix 1e3

#define PRECISION double
#define PRECISION_4VECTOR double4

#define flopf (long double)(20*BLOCK_X + 15)*GRID_X*1e-9
#define flopS (long double)flopf*rool*TotalPoint


// Disk, uneven.
void GenerateRandomPoints(PRECISION_4VECTOR *Point, PRECISION_4VECTOR *Point_v) {
  PRECISION SUP_v = 1e7;
  PRECISION INF_v = 1e4;
  PRECISION SUP_v_z = 1e2;
  PRECISION INF_v_z = -1e2;
  PRECISION SUP_z = 10 * LY;
  PRECISION INF_z = -10 * LY;
  PRECISION SUP_radiu = 1000 * LY;
  PRECISION INF_radiu = 1 * LY;
  PRECISION SUP_mass = 1e30;
  PRECISION INF_mass = 1e10;
  PRECISION temp_theta, temp_radiu, temp_v;
  srand(time(NULL));
  // Generate.
  // BlackHole, at the center of the disk.
  Point[0].w = (PRECISION)(1e10 * SUP_mass * G * dt);
  Point[0].x = 0;
  Point[0].y = 0;
  Point[0].z = 0;
  Point_v[0].w = 0;
  Point_v[0].x = 0;
  Point_v[0].y = 0;
  Point_v[0].z = 0;
  // Generate stars.
  for (unsigned long i = 1; i<TotalPoint; i++) {
    temp_theta = tau * rand()/(PRECISION)RAND_MAX;
    temp_radiu = (SUP_radiu - INF_radiu) * rand()/(PRECISION)RAND_MAX + INF_radiu;
    // Orbiting. 
    temp_v = (SUP_v - INF_v) * rand()/(PRECISION)RAND_MAX + INF_v;//sqrt(Point[0].w / temp_radiu / dt); 
    // Generate mass*G*dt constant.
    Point[i].w = (PRECISION)(((SUP_mass - INF_mass) * rand()/(PRECISION)RAND_MAX + INF_mass) * G * dt);
    // Generate location.
    Point[i].x = temp_radiu * cos(temp_theta);
    Point[i].y = temp_radiu * sin(temp_theta);
    Point[i].z = (SUP_z - INF_z) * rand()/(PRECISION)RAND_MAX + INF_z;
    // Generate velocity.
    Point_v[i].x = temp_v * (0 - sin(temp_theta));
    Point_v[i].y = temp_v * cos(temp_theta);
    Point_v[i].z = 0;//(SUP_v_z - INF_v_z) * rand()/(PRECISION)RAND_MAX + INF_v_z;
  }

  // Some massive stars. 
  for (unsigned long i = 1; i < 100; i++) {
    Point[i].w = (PRECISION)(1e7 * SUP_mass * G * dt);
    /*
    temp_radiu = sqrt(Point[i].x * Point[i].x + 
                      Point[i].y * Point[i].y + 
                      Point[i].z * Point[i].z); 
    temp_v = sqrt(Point[0].w / temp_radiu / dt); 
    Point_v[i].x = temp_v * (0 - sin(temp_radiu));
    Point_v[i].y = temp_v * cos(temp_radiu);
    */
  }
}


/*
// Cube, even.
void GenerateRandomPoints(PRECISION_4VECTOR *Point, PRECISION_4VECTOR *Point_v) {
  PRECISION Rx_u = 300 * LY;
  PRECISION Ry_u = 300 * LY;
  PRECISION Rz_u = 300 * LY;
  PRECISION Rx_l = -300 * LY;
  PRECISION Ry_l = -300 * LY;
  PRECISION Rz_l = -300 * LY;
  PRECISION Rvx_u = 1e6;
  PRECISION Rvy_u = 1e6;
  PRECISION Rvz_u = 1e6;
  PRECISION Rvx_l = -1e6;
  PRECISION Rvy_l = -1e6;
  PRECISION Rvz_l = -1e6;
  PRECISION Rm_u = 1e30;
  PRECISION Rm_l = 1e10;
  unsigned long BlackHoles = 10;
  srand(time(NULL));
  //Generate random location for points.
  for (unsigned long Tv=0; Tv < TotalPoint; Tv++) {
    Point[Tv].x = rand()/(PRECISION)RAND_MAX * (Rx_u - Rx_l) + Rx_l;
    Point[Tv].y = rand()/(PRECISION)RAND_MAX * (Ry_u - Ry_l) + Ry_l;
    Point[Tv].z = rand()/(PRECISION)RAND_MAX * (Rz_u - Rz_l) + Rz_l;

    //Point_Gmdt = Point.w
    Point[Tv].w = rand()/(PRECISION)RAND_MAX * (Rm_u + Rm_l) * G * dt;

    Point_v[Tv].x = rand()/(PRECISION)RAND_MAX * (Rvx_u - Rvx_l) + Rvx_l;
    Point_v[Tv].y = rand()/(PRECISION)RAND_MAX * (Rvy_u - Rvy_l) + Rvy_l;
    Point_v[Tv].z = rand()/(PRECISION)RAND_MAX * (Rvz_u - Rvz_l) + Rvz_l;
  }

  for (unsigned long Tv=0; Tv < BlackHoles; Tv++) {
    Point[Tv].w = 1e10 * Rm_u * G * dt;
  }
}
*/


void Save(PRECISION_4VECTOR *Point, int count) {
  char fileName[255];
  unsigned long long t[1] = {TotalPoint};
  sprintf(fileName, "./datas/%ld,%d.nbody", TotalPoint, count);
  FILE *save;
  if ((save = fopen(fileName, "wb")) == NULL) {
    printf("Can't save datas. %d", count);
  }
  // Save the number of bodies.
  fwrite(t, 8, 1, save);

  fwrite(Point, sizeof(PRECISION_4VECTOR), TotalPoint, save); 
  fclose(save);
}


__global__ void CaculateTheNextTick(PRECISION_4VECTOR *Point, PRECISION_4VECTOR *Point_v, PRECISION_4VECTOR *T) {

  unsigned long i =  blockIdx.x * blockDim.x + threadIdx.x; // Get thread's index

  //if (i < TotalPoint) {
    PRECISION da_i = 0.0f;
    PRECISION rd = 0.0f;

    PRECISION x_i = Point[i].x;
    PRECISION y_i = Point[i].y;
    PRECISION z_i = Point[i].z;

    PRECISION dx = 0.0f;
    PRECISION dy = 0.0f;
    PRECISION dz = 0.0f;

    PRECISION da_ix = 0.0f;
    PRECISION da_iy = 0.0f;
    PRECISION da_iz = 0.0f;

    __shared__ PRECISION_4VECTOR Temp[TILE_SIZE];
    for (int tile = 0; tile < TotalPoint / TILE_SIZE; tile++) {
      __syncthreads(); 
      for (int tile_sync_index = 0; tile_sync_index < TILE_SIZE / BLOCK_X; tile_sync_index++) {
        Temp[BLOCK_X * tile_sync_index + threadIdx.x] = Point[BLOCK_X * tile_sync_index + tile * TILE_SIZE + threadIdx.x];
      }
      __syncthreads();

      //#pragma unroll
      for (int j = 0; j < TILE_SIZE; j++) {
        //if (i != j) {
          dx = Temp[j].x - x_i;//1 flo
          dy = Temp[j].y - y_i;//1 flo
          dz = Temp[j].z - z_i;//1 flo

          rd = rsqrtf((dx * dx) + (dy * dy) + (dz * dz) + fix); // 8 flo

          da_i = Temp[j].w * rd * rd * rd; //3 flo

          da_ix += dx * da_i;//2 flo
          da_iy += dy * da_i;//2 flo
          da_iz += dz * da_i;//2 flo
        //}
      }//total 20 * TotalPoint flo
    }

    T[i].x = x_i + Point_v[i].x * dt + half_dt * da_ix;//4 flo
    T[i].y = y_i + Point_v[i].y * dt + half_dt * da_iy;//4 flo
    T[i].z = z_i + Point_v[i].z * dt + half_dt * da_iz;//4 flo

    Point_v[i].x = Point_v[i].x + da_ix;//1 flo
    Point_v[i].y = Point_v[i].y + da_iy;//1 flo
    Point_v[i].z = Point_v[i].z + da_iz;//1 flo
  //}//total 15 + 12*TotalPoint flo
}

int main() {
  //Define what we need on CPU.
  cudaSetDevice(2);

  PRECISION_4VECTOR *Point;

  PRECISION_4VECTOR *Point_v;

  Point = (PRECISION_4VECTOR *)malloc(TotalPoint * sizeof(PRECISION_4VECTOR));
  Point_v = (PRECISION_4VECTOR *)malloc(TotalPoint * sizeof(PRECISION_4VECTOR));

  //Define what we need on GPU.
  PRECISION_4VECTOR *GPU_Point;

  PRECISION_4VECTOR *GPU_Point_v;

  PRECISION_4VECTOR *GPU_T;

  cudaMalloc((void**)&GPU_Point, TotalPoint * sizeof(PRECISION_4VECTOR));
  cudaMalloc((void**)&GPU_Point_v, TotalPoint * sizeof(PRECISION_4VECTOR));
  cudaMalloc((void**)&GPU_T, TotalPoint * sizeof(PRECISION_4VECTOR));

  int count = 0;
  double starttime, endtime, deltatime;

  dim3 grid(GRID_X, GRID_Y, GRID_Z);
  dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

  FILE *save;
  if ((save=fopen("data.data", "w")) == NULL) {
    printf("Can't save data.\n");
  }
  fclose(save);
  //Generate random point.
  GenerateRandomPoints(Point, Point_v);
  Save(Point, count); 
  
  cudaMemcpy(GPU_Point, Point, TotalPoint * sizeof(PRECISION_4VECTOR), cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_T, Point, TotalPoint * sizeof(PRECISION_4VECTOR), cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_Point_v, Point_v, TotalPoint * sizeof(PRECISION_4VECTOR), cudaMemcpyHostToDevice);
  free(Point_v);

  printf("Start to compute. N=%ld, dt=%f, frame per save=%d\n", TotalPoint, dt, rool);
  while (1==1) {
    count++;

    printf("[Save %d]: Computing... \n", count);
    //fflush(stdout);
    //Caculate the location of next tick.
    starttime = getMillisecond(); 

    for (int k=0; k < rool; k++) {
      printf("\tCalculating %d... \r", (count - 1) * rool + k);
      fflush(stdout);
      if (k%2) {
        CaculateTheNextTick<<<grid, block>>>(GPU_T, GPU_Point_v, GPU_Point);
      } else {
        CaculateTheNextTick<<<grid, block>>>(GPU_Point, GPU_Point_v, GPU_T);
      }
      cudaDeviceSynchronize();
      cudaThreadSynchronize();
    }

    endtime = getMillisecond();

    deltatime = ( endtime - starttime ) / 1000;
    cudaMemcpy(Point, GPU_T, TotalPoint * sizeof(PRECISION_4VECTOR), cudaMemcpyDeviceToHost);

    printf("\n\tGPPS = %Lf, \tGFlops=%Lf\n",
        BodiesPerSave / (long double)deltatime,
        flopS / (long double)deltatime);
    
    fflush(stdout); 
    printf(" Saving... ");
    Save(Point, count);
    printf("Done. \n");
  }
  //fclose(save);

  //free(Point);
  //free(Point_v);
  //cudaFree(GPU_Point);
  //cudaFree(GPU_Point_v);
  //cudaFree(GPU_T);

  //General end of C programs.

  return 0;
}
