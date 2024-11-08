#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv){

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        printf("I am thread %d \n",id);

    }
    
    
    return 0;

}