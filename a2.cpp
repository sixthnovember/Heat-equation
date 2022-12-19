#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <mpi.h>

#include "a2-helpers.hpp"

using namespace std;

int main(int argc, char **argv) {
    int max_iterations = 15000;
    double epsilon = 1.0e-3;

    // default values for M rows and N columns
    int N = 12;
    int M = 12;

    process_input(argc, argv, N, M, max_iterations, epsilon);

    /* 
    START: the main part of the code that needs to use MPI 
    */
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // auto time_mpi_1 = chrono::high_resolution_clock::now(); // change to MPI_Wait
    auto time_mpi_1 = MPI_Wtime();
    
    int M_local = ceil(M/size)+2, N_local = N;

    if(M%size != 0 && rank == size-1) {
        M_local += (M%size);
    }
    
    int i, j;
    double diffnorm, diffnorm_global;
    int iteration_count = 0;

    // allocate another 2D array
    // Mat U(M, N); // change: use local sizes with MPI, e.g., recalculate M and N
    // Mat W(M, N); // change: use local sizes with MPI
    Mat U_local(M_local, N_local);
    Mat W_local(M_local, N_local);

    // Init & Boundary
    for (i = 0; i < M_local; ++i) {
        for (j = 0; j < N_local; ++j) {
            U_local(i, j) = 0.0;
        }
    }

    if(rank == size-1) {
        for (j = 0; j < N_local; ++j) {
            U_local(M_local - 2, j) = 100.0;
        }
    }
    // End init

    int next, prev;

    if(rank == size-1) {
        next = -1;
    } else {
        next = rank+1;
    }

    if(rank == 0) {
        prev = -1;
    } else {
        prev = rank-1;
    }

    iteration_count = 0;
    diffnorm_global = 0.0;
    do {
        /* Compute new values (but not on boundary) */
        iteration_count++;
        diffnorm = 0.0;
        diffnorm_global = 0.0;

        // send/receive ghost regions
        if(rank == 0) {
            MPI_Request request1;
            MPI_Status status1;
            // receives value for ghost region - lower
            MPI_Irecv(&U_local(M_local-1, 0), N_local, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &request1);
            // send lower value of local matrix
            MPI_Send(&U_local(M_local-2, 0), N_local, MPI_DOUBLE, next, 0, MPI_COMM_WORLD);
            MPI_Wait(&request1, &status1);

            for(i = 2; i < M_local-1; ++i) {
                for(j = 1; j < N_local-1; ++j) {
                    W_local(i,j) = (U_local(i,j+1) + U_local(i,j-1) + U_local(i+1,j) + U_local(i-1,j)) * 0.25;
                    diffnorm += (W_local(i,j) - U_local(i,j))*(W_local(i,j) - U_local(i,j));
                }
            }
            // Only transfer the interior points - i = 0 does not exist and i = 1 is not interior
            for(i = 2; i < M_local-1; ++i) {
                for(j = 1; j < N_local-1; ++j) {
                    U_local(i,j) = W_local(i,j);
                }
            }
        } else if(rank == size-1) {
            MPI_Request request2;
            MPI_Status status2;
            // receives value for ghost region - upper
            MPI_Irecv(&U_local(0, 0), N_local, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &request2);
            // sends upper value of local matrix
            MPI_Send(&U_local(1, 0), N_local, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD);
            MPI_Wait(&request2, &status2);

            for(i = 1; i < M_local-2; ++i) {
                for(j = 1; j < N_local-1; ++j) {
                    W_local(i,j) = (U_local(i,j+1) + U_local(i,j-1) + U_local(i+1,j) + U_local(i-1,j)) * 0.25;
                    diffnorm += (W_local(i,j) - U_local(i,j))*(W_local(i,j) - U_local(i,j)); // TODO fix
                }
            }
            
            // Only transfer the interior points - i = 0 is ghost region
            for(i = 1; i < M_local-2; ++i) { 
                for(j = 1; j < N_local-1; ++j) {
                    U_local(i,j) = W_local(i,j);
                }
            }
            
        } else {
            MPI_Request request3;
            MPI_Status status3;
            // receives value for ghost region - lower
            MPI_Irecv(&U_local(M_local-1, 0), N_local, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &request3);
            // send lower value of local matrix
            MPI_Send(&U_local(M_local-2, 0), N_local, MPI_DOUBLE, next, 0, MPI_COMM_WORLD);
            MPI_Wait(&request3, &status3);

            MPI_Request request4;
            MPI_Status status4;
            // receives value for ghost region - upper
            MPI_Irecv(&U_local(0, 0), N_local, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &request4);
            // send upper value of local matrix
            MPI_Send(&U_local(1, 0), N_local, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD);
            MPI_Wait(&request4, &status4);

            for(i = 1; i < M_local-1; ++i) {
                for(j = 1; j < N_local-1; ++j) {
                    W_local(i,j) = (U_local(i,j+1) + U_local(i,j-1) + U_local(i+1,j) + U_local(i-1,j)) * 0.25;
                    diffnorm += (W_local(i,j) - U_local(i,j))*(W_local(i,j) - U_local(i,j));
                }
            }
            // Only transfer the interior points - i = 0 is ghost region, i = local_mat_size_M-1 is ghost region
            for(i = 1; i < M_local-1; ++i) { 
                for(j = 1; j < N_local-1; ++j) {
                    U_local(i,j) = W_local(i,j);
                }
            }
            
        }
        MPI_Allreduce(&diffnorm, &diffnorm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        diffnorm_global = sqrt(diffnorm_global);
        //diffnorm = sqrt(diffnorm);
        // if(rank==0) {printf("At iteration %d, diff is %f \n", iteration_count, diffnorm_global);} // checks
    } while (epsilon <= diffnorm_global && iteration_count < max_iterations);

    // auto time_mpi_2 = chrono::high_resolution_clock::now(); // change to MPI_Wait
    auto rank_elapsed_time_mpi = MPI_Wtime() - time_mpi_1;
    
    /* time measuring */
    double rank_max_time = -1;
    // slowest time, gets sent to rank 0
    MPI_Reduce(&rank_elapsed_time_mpi, &rank_max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // TODO for MPI: collect all local parts of the U matrix, and save it to another "big" matrix
    // that has the same size as the U_sequential matrix (see below), then verify the results below

    Mat U_local_send(M_local-2, N_local);

    // remove first and last row of the local matrix
    for (i = 0; i < M_local-1; ++i) {
        if(i == 0) {
            continue;
        } else {
            for (j = 0; j < N_local; ++j) {
                U_local_send(i-1, j) = U_local(i, j);
            }
        }
    }

    Mat U_MPI(M, N);

    int sendcount = (M_local-2)*N_local;
    int receive_counts[size];
    int receive_displs[size];

    for (int i = 0; i < size; ++i) {
        receive_counts[i] = ((M_local-2)*N_local);
        receive_displs[i] = (M_local-2)*N_local*i;
    }
    receive_counts[size-1] += (((M_local-2)*N_local)%size);

    MPI_Gatherv(&U_local_send(0,0), sendcount, MPI_DOUBLE, &U_MPI(0,0), receive_counts, receive_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // verification     
    Mat U_sequential(M, N); // init another matrix for the verification

    // start time measurement for the sequential version
    auto ts_1 = chrono::high_resolution_clock::now();
    heat2d_sequential(max_iterations, epsilon, U_sequential, iteration_count); 
    auto ts_2 = chrono::high_resolution_clock::now();

    auto sequential_time = chrono::duration<double>(ts_2 - ts_1).count();
    auto mpi_time = rank_max_time;
    auto speed_up = sequential_time/mpi_time;

    if(rank == 0) {
        // print time measurements
        cout << "Computed (MPI) in " << iteration_count << " iterations and " << rank_max_time << " seconds." << endl;
        cout << "Computed (sequential) in " << iteration_count << " iterations and " << sequential_time << " seconds." << endl;
        cout << "Verification: " << ( verify(U_MPI, U_sequential) ? "OK" : "NOT OK") << std::endl;
        cout << "Speed up: " << speed_up <<endl;
    } 

    MPI_Finalize();

    //save_to_disk(U_sequential, "heat2d.txt");
    //save_to_disk(U_MPI, "heat2d_MPI.txt");

    return 0;
}