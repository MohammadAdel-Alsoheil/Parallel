#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to compare two integers for qsort
int compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

// Serial bucket sort for comparison
void serial_bucket_sort(int *data, int N) {
    // Simple serial sort using qsort
    qsort(data, N, sizeof(int), compare);
}

void main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1000000; // Total number of elements (increased for better timing)
    int *data = NULL;
    int *sorted_serial = NULL;
    double serial_start, serial_end, parallel_start, parallel_end;
    double serial_time, parallel_time, speedup;

    // Allocate memory and initialize data on root
    if (rank == 0) {
        data = (int *)malloc(N * sizeof(int));
        sorted_serial = (int *)malloc(N * sizeof(int));
        srand(time(0));
        for (int i = 0; i < N; i++) {
            data[i] = rand() % 1000000; // random numbers between 0 and 999999
        }

        // Copy data for serial sort
        for (int i = 0; i < N; i++) {
            sorted_serial[i] = data[i];
        }

        // Serial Sort Timing
        serial_start = MPI_Wtime();
        // Simple serial sort using qsort
        qsort(sorted_serial, N, sizeof(int), compare);
        serial_end = MPI_Wtime();
        serial_time = serial_end - serial_start;
        printf("Serial sort completed in %f seconds.\n", serial_time);
    }

    // Parallel Sort Timing Start
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes start together
    parallel_start = MPI_Wtime();
    
    // Broadcast N to all processes in case it's needed
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = N / size;
    int *local_data = (int *)malloc(chunk_size * sizeof(int));

    // Scatter data to all processes
    MPI_Scatter(data, chunk_size, MPI_INT, local_data, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process sorts its local data
    qsort(local_data, chunk_size, sizeof(int), compare);

    // Prepare buckets
    int *sendcounts = (int *)calloc(size, sizeof(int));  // counts of numbers to send to each process
    int *sdispls = (int *)calloc(size, sizeof(int));     // displacements of numbers to send to each process
    int *recvcounts = (int *)calloc(size, sizeof(int));  // counts of numbers to receive from each process
    int *rdispls = (int *)calloc(size, sizeof(int));     // displacements of numbers to receive from each process

    int **buckets = (int **)malloc(size * sizeof(int *));
    for (int i = 0; i < size; i++) {
        buckets[i] = (int *)malloc(chunk_size * sizeof(int));
        sendcounts[i] = 0;
    }

    // Distribute local data into appropriate buckets based on value range
    for (int i = 0; i < chunk_size; i++) {
        int target_proc = (int)(((double)local_data[i] / 1000000) * size);
        if (target_proc >= size) target_proc = size - 1; // Handle max value
        buckets[target_proc][sendcounts[target_proc]++] = local_data[i];
    }

    // Calculate send displacements
    sdispls[0] = 0;
    for (int i = 1; i < size; i++) {
        sdispls[i] = sdispls[i - 1] + sendcounts[i - 1];
    }

    int total_send = sdispls[size - 1] + sendcounts[size - 1];
    int *sendbuf = (int *)malloc(total_send * sizeof(int));

    // Pack sendbuf with values from buckets
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < sendcounts[i]; j++) {
            sendbuf[sdispls[i] + j] = buckets[i][j];
        }
    }

    // Exchange the counts of data each process will receive
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate receive displacements
    rdispls[0] = 0;
    for (int i = 1; i < size; i++) {
        rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
    }

    int total_recv = rdispls[size - 1] + recvcounts[size - 1];
    int *recvbuf = (int *)malloc(total_recv * sizeof(int));

    // All-to-all exchange of the data
    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_INT, recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);

    // Sort the received data
    qsort(recvbuf, total_recv, sizeof(int), compare);

    // Gather the sizes of sorted data from all processes to root
    MPI_Gather(&total_recv, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements for Gatherv
    if (rank == 0) {
        rdispls[0] = 0;
        for (int i = 1; i < size; i++) {
            rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
        }
    }

    // Allocate memory for the sorted parallel data on root
    int *sorted_parallel = NULL;
    if (rank == 0) {
        sorted_parallel = (int *)malloc(N * sizeof(int));
    }

    // Gather all sorted data to root
    MPI_Gatherv(recvbuf, total_recv, MPI_INT, sorted_parallel, recvcounts, rdispls, MPI_INT, 0, MPI_COMM_WORLD);

    // Parallel Sort Timing End
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes have finished
    parallel_end = MPI_Wtime();
    parallel_time = parallel_end - parallel_start;

    if (rank == 0) {
        printf("Parallel sort completed in %f seconds.\n", parallel_time);

        // Verify if the serial and parallel sorted arrays are the same
        int correct = 1;
        for (int i = 0; i < N; i++) {
            if (sorted_serial[i] != sorted_parallel[i]) {
                correct = 0;
                printf("Mismatch at index %d: serial=%d, parallel=%d\n", i, sorted_serial[i], sorted_parallel[i]);
                break;
            }
        }
        if (correct) {
            printf("Verification: SUCCESS. Serial and Parallel sorted arrays match.\n");
        } else {
            printf("Verification: FAILURE. Sorted arrays do not match.\n");
        }

        // Calculate speedup
        speedup = serial_time / parallel_time;
        printf("Speedup: %f\n", speedup);
    }

    // Free allocated memory
    free(local_data);
    free(sendcounts);
    free(recvcounts);
    free(sdispls);
    free(rdispls);
    free(recvbuf);
    free(sendbuf);
    for (int i = 0; i < size; i++) {
        free(buckets[i]);
    }
    free(buckets);
    if (rank == 0) {
        free(data);
        free(sorted_serial);
        free(sorted_parallel);
    }

    MPI_Finalize();
}
