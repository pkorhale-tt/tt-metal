#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define PI 3.14159265358979323846264338327950288

void calc(float*, int);
void fft(float*, float*, int);
void bitreverse(float*, int);
float* computeTwiddleFactors(int);
void fillData(float*, int);
void moveorigin(float*, int);
void descale(float*, int);
void invert(float*, int);
void compare(float*, float*, int);
int checkIfPowerOfTwo(int);
int getLog(int);

int main(int argc, char * argv[]) {
  if (argc != 2) {
    fprintf(stderr, "You must provide the size of the domain as an argument\n");
    return -1;
  }

  int domain_size=atoi(argv[1]);
  if (!checkIfPowerOfTwo(domain_size)) {
    fprintf(stderr, "%d provided as domain size, but this must be a power of two\n", domain_size);
    return -1;
  }

  srand((unsigned int)time(NULL));

  float * orig_data=(float*) malloc(sizeof(float) * domain_size * 2);
  float * data=(float*) malloc(sizeof(float) * domain_size * 2);
  fillData(orig_data, domain_size);
  memcpy(data, orig_data, sizeof(float) * domain_size * 2);

  calc(data, domain_size);
  invert(data, domain_size);
  calc(data, domain_size);
  moveorigin(data, domain_size);
  descale(data, domain_size);
  compare(data, orig_data, domain_size);
  free(data);
  return 0;
}

void calc(float * data, int domain_size) {
  float * twiddle_factors=computeTwiddleFactors(domain_size);
  bitreverse(data, domain_size);
  fft(data, twiddle_factors, domain_size);
  free(twiddle_factors);
}

void fft(float * data, float * twiddle_factors, int domain_size) {
  int num_steps=getLog(domain_size);  
  for (int step=0; step <= num_steps; step++) {    
    // The number of spectra in this step is 2^step, for performance we implement via a bitshift
    int num_spectra_in_step=step == 0 ? 1 : 2 << (step-1);    
    // This is the increment of next point in the spectra it will process (i.e. jumping over points in other spectra)
    int increment_next_point_in_step=2 << step;
    // The increment of the second, corresponding point, to handle from the current point (e.g. 1, then 2, then 4 etc)
    int matching_second_point=increment_next_point_in_step/2;    
    for (int spectra=0; spectra < num_spectra_in_step; spectra++) {
      // For each spectra we have a specific twiddle factor, the index of this is the spectra number
      // multipled by two n times (where n starts out as the number of steps and decreases on each step)
      int twiddle_index=spectra << (num_steps-step);
      for (int point=0; point < domain_size; point+=increment_next_point_in_step) {        
        //printf("Compute step=%d spectra=%d, and i1=%d with twiddle %d\n", step, spectra, i1, twiddle_index);        
        int d0_data_index=(spectra + point)*2;
        int d1_data_index=(spectra + point + matching_second_point)*2;
        float f0=(data[d1_data_index] * twiddle_factors[twiddle_index*2]) - (data[d1_data_index+1] * twiddle_factors[(twiddle_index*2)+1]);
        float f1=(data[d1_data_index] * twiddle_factors[(twiddle_index*2)+1]) + (data[d1_data_index+1] * twiddle_factors[(twiddle_index*2)]);
        
        printf("[step %d, spectra %d, point %d] Twiddle index %d, D0 index %d, D1 index %d\n", step, spectra, point, twiddle_index, d0_data_index/2, d1_data_index/2);

        data[d1_data_index]=data[d0_data_index] - f0;
        data[d1_data_index+1]=data[d0_data_index+1] - f1;
        data[d0_data_index]=data[d0_data_index] + f0;
        data[d0_data_index+1]=data[d0_data_index+1] + f1;
      }
    }
  }
}

void bitreverse(float * data, int n) {
  int j=0;
  for (int i=0;i<n-1;i++) {
    if (i < j) {
      float temp_r=data[i*2];
      float temp_i=data[(i*2)+1];

      data[i*2]=data[j*2];
      data[(i*2)+1]=data[(j*2)+1];

      data[j*2]=temp_r;
      data[(j*2)+1]=temp_i;
    }
    int k=n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j+=k;
  }
}

float* computeTwiddleFactors(int n) {
   int num_twiddle_factors=n/2;
   float * twiddle_factors=(float*) malloc(sizeof(float) * num_twiddle_factors * 2);

   for (int i=0;i<num_twiddle_factors;i++) {
     float base_factor=(2.0 * PI * i)/(float) n;
     twiddle_factors[i*2]=(float) cos((double) base_factor);
     twiddle_factors[(i*2)+1]=(float) -sin((double) base_factor);
   }

   return twiddle_factors;
}

void fillData(float * data, int domain_size) {
  for (int i=0;i<domain_size;i++) {
    data[i*2]=0.0; //(float)rand()/(float)(RAND_MAX);
    data[(i*2)+1]=0.0; //(float)rand()/(float)(RAND_MAX);
  }
  data[(domain_size/2)*2]=data[((domain_size/2)*2)+1]=(float) domain_size;
}

void invert(float * data, int domain_size) {
  for (int i=0; i<domain_size;i++) {
    data[(i*2)+1]=-data[(i*2)+1];
  }
}

void moveorigin(float* data, int domain_size) {
  for (int i=0;i<domain_size;i++) {
    data[i*2]=data[i*2] * pow(-1, i);
    data[(i*2)+1]=data[(i*2)+1] * pow(-1, i);
  }
}

void descale(float* data, int domain_size) {
  for (int i=0;i<domain_size;i++) {
    data[i*2]=data[i*2] / domain_size;
    data[(i*2)+1]=-(data[(i*2)+1] / domain_size);
  }
}

void compare(float * a_data, float * b_data, int domain_size) {
  int matching, missmatching;
  matching=missmatching=0;
  for (int i=0; i<domain_size;i++) {
    float a_r=a_data[i*2];
    float a_i=a_data[(i*2)+1];
    float b_r=b_data[i*2];
    float b_i=b_data[(i*2)+1];

    if (a_r != b_r || a_i != b_i) {
      printf("Miss match index %d: (%.2f, %.2f) vs (%.2f, %.2f)\n", i, a_r, a_i, b_r, b_i);
      missmatching++;
    } else {
      matching++;
    }
  }
  printf("Checked %d elements: %d match and %d missmatched\n", domain_size, matching, missmatching);
}

int checkIfPowerOfTwo(int v) {
  return (v != 0) && ((v & (v - 1)) == 0);
}

int getLog(int n) {
   int logn=0;
   n >>= 1;
   while ((n >>=1) > 0) {
      logn++;
   }
   return logn;
}

