#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int* generate_primes(int n){

    int* primes = malloc((n+1)* sizeof(int));  // 1 --> prime, 0 --> composite
    for(int i = 2;i<=n;++i){
        primes[i]=1;
    }
    
    for(int p = 2;p<=(int)sqrt(n);++p){

        if(primes[p]==1){ // if it is prime

            for(int j = p*p;j<=n;j+=p){
                primes[j] = 0;
            }
        }
    }

    return primes;


}

int main(int argc, char** argv){

    int rank, size;
    double sequential_start, sequential_end,sequential_time;
    double parallel_start, parallel_end, parallel_time;
    int n = 8010000; // we need to check all the prime numbers that come before n

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);


    if(rank == 0){
       

        sequential_start = MPI_Wtime();
        generate_primes(n);
        sequential_end = MPI_Wtime();
        sequential_time = sequential_end - sequential_start;
    }
        
        MPI_Barrier(MPI_COMM_WORLD);
        parallel_start = MPI_Wtime();

       
        // parallel implementation
        // consider having n processors, each proccessor will eliminate some multiple of a number and pass it to next one
        // buffer to hold primes at each process

        

        int total_primes;
        int* primes_to_gen;
        if(rank == 0 ){
            
           primes_to_gen = generate_primes((int) sqrt(n));
           
    
            total_primes = 0;
            for (int i = 2; i <= (int)sqrt(n); ++i) {
                if (primes_to_gen[i]) {
                    total_primes++;
                }
            }
            
        }
        if(total_primes%size !=0){
            printf("Cant distribute prime numbers on processors, we have %d prime numbers and %d processors\n",total_primes,size);
            free(primes_to_gen);
            MPI_Finalize();
            return 1;
        }
        MPI_Bcast(&total_primes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        
        int primes_per_process = total_primes/size;
        int* recvbuf = malloc(primes_per_process*sizeof(int));
        int* primes_to_send = malloc(total_primes * sizeof(int));

        if(rank == 0){

            int idx = 0;
            for (int i = 2; i <= (int)sqrt(n); ++i) {
                if (primes_to_gen[i]) {
                    primes_to_send[idx] = i;
                    idx++;
                }
            }
             free(primes_to_gen);
        }
        
        
    
        MPI_Scatter(primes_to_send, primes_per_process, MPI_INT, recvbuf, primes_per_process, MPI_INT, 0, MPI_COMM_WORLD);

        // printf("I am proc %d\n",rank);
        // for(int i = 0;i<primes_per_process;++i){
        //     printf("%d ",recvbuf[i]);
        // }
        // printf("\n");
        

        //start sending other number to check primes
        if(rank == 0){

            for(int num = (int) sqrt(n);num<=n;++num){

                int is_prime = 1;
                for (int i = 0; i < primes_per_process; ++i) {
                    if (num % recvbuf[i] == 0) {
                        is_prime = 0;
                        break;
                    }
                }
                if (is_prime) {
                    if(rank+1<size){
                        MPI_Send(&num, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
                    }
                    
                }

            }
            int terminate = -1;
            MPI_Send(&terminate,1,MPI_INT,rank+1,0,MPI_COMM_WORLD);

        }
        else{
            int num;

            
            while(1){
                MPI_Recv(&num,1,MPI_INT,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

                if(num==-1){
                    if(rank+1<size){
                        MPI_Send(&num,1,MPI_INT,rank+1,0,MPI_COMM_WORLD);
                    }
                    break;
                }

                int is_prime = 1;
                for (int i = 0; i < primes_per_process; ++i) {
                    if (num % recvbuf[i] == 0) {
                        is_prime = 0;
                        break;
                    }
                }
                if (is_prime) {
                    if (rank + 1 < size) {
                        MPI_Send(&num, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
                    } else {
                        //printf("%d ", num);
                        // you can generate the rest of the primes here
                    }
                }
            }
                
        }
        MPI_Barrier(MPI_COMM_WORLD);
        parallel_end = MPI_Wtime();
        parallel_time = parallel_end-parallel_start;
        if (rank==0){
            printf("\n");
            printf("###Parallel Results###\n");
            printf("The time taken on pipelined approach is %f \n",parallel_time);
            printf("###Seq Results###\n");
            printf("The time taken on Seq approach is %f \n",sequential_time);
             float speed_up =  sequential_time/parallel_time;
            printf("The speedup factor is %f \n",speed_up);
            printf("The efficiency is %f",(float)(speed_up/size)*100);

        }

        free(primes_to_send);
        MPI_Finalize();
        
        


        

        
        return 0;
        
}