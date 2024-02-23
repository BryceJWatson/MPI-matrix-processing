/* a3.c
MPI program that utilises multiple mpi processes to process a matrix.

Elements in the input matrix are replaced by the weighted sum of their
neighbours (1/n_depth)

Author: Bryce Watson
Student ID: 220199390

Parameters:
  1. The input matrix file
  2. The output matrix file
  3. The number of processes/matrix dimensions

The program won't correctly calculate the output matrix when there are less
processes than matrix dimensions

Returns 0 on success

To build it use: make

To run: make run (this will run the program with 4 processes on a input matrix
'test_matrix' and write the results to 'test_results' on a matrix with 4
dimensions.)

OR

  mpirun -np <dimensions> ./a3 <input_matrix> <output_matrix> <dimensions>

  e.g mpirun -np 4 ./a3 test_matrix test_results 4

to clean:
  make clean
 */

/****** Included libraries ******/

#include "matrix.h"
#include "mpi.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define MAIN_PROCESS 0

/**
 * @brief Simple function to parse command line arguments
 *
 * This function ensures that the correct command line arguments are given.
 * This function expects the following command line arguments:
 * 	argv[1]: Input matrix file name
 * 	argv[2]: Output matrix file name
 * 	argv[3]: Number of mpi processes to use
 *
 * @param argc An integer representing the number of command line arguments
 * @param *argv[] A pointer to an array of strings representing the command line
 * arguments
 * @param *fd A pointer to a integer filedescriptor representing the matrix
 * input file
 * @param *np A pointer to an integer representing the number of processes
 *
 * @return 0 on success, -1 on failure
 */

int parse_args(int argc, char *argv[], int *fd, int *np) {
  if ((argc != 4) || ((fd[0] = open(argv[1], O_RDONLY)) == -1) ||
      ((fd[1] = open(argv[2], O_WRONLY | O_CREAT, 0666)) == -1) ||
      (*np = atoi(argv[3])) <= 0) {
    fprintf(stderr,
            "Usage: mpirun -np dimension %s matrixA matrixB dimension\n",
            argv[0]);
    return (-1);
  }
  return 0;
}

/* Simple exit handler function that will call MPI_Finalize on exit, freeing
memory, is void so doesn't return anything and also doesn't accept any function
arguments*/
void cleanup() { MPI_Finalize(); }

/* Main function: Parameters: integer argc which is the number of command line
 * arguments, and an array of pointers to the command line arguments *argv[]
 Returns 0 on success, 1 on failure */
int main(int argc, char *argv[]) {
  /* Declare variables */
  int me, row, col, fd[2], i, nprocs, dim;

  /* Register Exit handler function */
  atexit(cleanup);

  /* Initialize MPI environment */
  MPI_Init(&argc, &argv);
  /* Each process gets their rank */
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  /* Get number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (me == MAIN_PROCESS) { /* parent code */

    if (parse_args(argc, argv, fd, &dim) < 0) {
      close(fd[0]);
      close(fd[1]);
      exit(EXIT_FAILURE);
    }
  }
  /* broadcaste dim to all */
  MPI_Bcast(&dim, 1, MPI_INT, MAIN_PROCESS, MPI_COMM_WORLD);

  /* Matrix dimensions must be the same as the number of processes */
  if (dim != nprocs) {
    if (me == MAIN_PROCESS) {
      fprintf(stderr,
              "Usage: mpirun -np dimension %s matrixA matrixB dimension\n",
              argv[0]);
      close(fd[0]);
      close(fd[1]);
      exit(EXIT_FAILURE);
    }
  }

  /* The maximum possible neighbourhood depth is N - 1*/
  int n_depth = dim - 1;
  /* The size of the supersized matrix needs to be N + (2 * n_depth) to add
   * enough padding around the matrix*/
  int supersize = dim + (2 * n_depth);
  /* The number of rows each process will need to compute its row is 1(Itself) +
   * 2 * n_depth(the zeros on either side) */
  int use_rows = 1 + (2 * n_depth);

  int A[dim + 1][dim + 1], P[dim + 1][dim + 1], superA[supersize][supersize];
  int Arow[use_rows][supersize], Prow[supersize];

  /* 2d flag array to ensure no neighbours get double counted during processing
   */
  int considered[use_rows + 1][supersize + 1];

  if (me == MAIN_PROCESS) { /* parent code */
    /* initialize matrix A (The input matrix) */
    for (i = 1; i < dim + 1; i++)
      if (get_row(fd[0], dim, i, &A[i][1]) == -1) {
        fprintf(stderr, "Initialization of A failed\n");
        goto fail;
      }
    /* Print original matrix*/
    fprintf(stderr, "Original matrix:\n");
    for (row = 1; row < dim + 1; row++) {
      for (col = 1; col < dim + 1; col++) {
        fprintf(stderr, "%d ", A[row][col]);
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");

    /* initialize supersized matrix */
    for (row = 1; row < supersize + 1; row++) {
      for (col = 1; col < supersize + 1; col++) {
        superA[row][col] = 0;
      }
    }

    for (row = n_depth + 1; row < dim + n_depth + 1; row++)
      for (col = n_depth + 1; col < dim + n_depth + 1; col++)
        superA[row][col] = A[row - n_depth][col - n_depth];
  }

  /* Scatter super sized A */
  /* Each process will get use_rows rows of the supersized matrix */
  /* After each loop, the starting location of the data to be sent &superA[i]
   * will be shifted by 1, so every process will get a different set of rows
   * of the supersized matrix.*/
  for (i = 1; i < use_rows + 2; i++)
    /* Scatter the data starting at row i of the supersized matrix */
    /* For example, in the first loop, the first process will get the first
     * row (of size supersize elements/columns), and the second process will
     * get the second row, and so on */
    if (MPI_Scatter(&superA[i], supersize, MPI_INT, &Arow[i], supersize,
                    MPI_INT, MAIN_PROCESS, MPI_COMM_WORLD) != MPI_SUCCESS) {
      fprintf(stderr, "Scattering of A failed\n");
      goto fail;
    }

  /* compute my row of P */
  for (i = 1; i < dim + 1; i++) {
    /* Initialize the considered matrix to all zeros */
    for (int row = 1; row < use_rows + 1; row++) {
      for (int col = 1; col < supersize + 1; col++) {
        considered[row][col] = 0;
      }
    }
    /* Don't count the target element, subtract it here and it will be added
     * during the calculation so the total will be 0*/
    Prow[i] = -Arow[n_depth + 1][n_depth + i];

    // Iterate over neighborhood depths from 1 to max neighborhood depth
    // (n_depth)
    for (int depth = 1; depth < n_depth + 1; depth++) {
      // Calculate the offsets for the current neighborhood depth
      int row_offset = n_depth + 1;
      int col_offset = n_depth + i;

      // Add contributions from the neighbors at this depth
      for (int row = -depth; row <= depth; row++) {
        for (int col = -depth; col <= depth; col++) {
          if (!considered[row_offset + row][col_offset + col]) {
            int contribution = Arow[row_offset + row][col_offset + col];
            /* If within neighbourhood depth of 1, don't weight it*/
            if (depth == 1) {
              Prow[i] += contribution;
            } else {
              /* Adjust the contribution based on the neighborhood depth */
              Prow[i] += round((double)contribution / depth);
            }
            /* Flag the element so it won't be double counted */
            considered[row_offset + row][col_offset + col] = 1;
          }
        }
      }
    }
  }

  /* Gather rows of P */
  if (MPI_Gather(&Prow[0], dim + 1, MPI_INT, &P[1][0], dim + 1, MPI_INT,
                 MAIN_PROCESS, MPI_COMM_WORLD) != MPI_SUCCESS) {
    fprintf(stderr, "Gathering of Product  failed\n");
    goto fail;
  }

  /* write the matrix to a file */
  if (me == MAIN_PROCESS) {

    for (i = 1; i < dim + 1; i++)
      if (set_row(fd[1], dim, i, &P[i][1]) == -1) {
        fprintf(stderr, "Writing of matrix C failed\n");
        goto fail;
      }

    /* Print the results */
    fprintf(stderr, "Final processed matrix:\n");
    for (row = 1; row < dim + 1; row++) {
      for (col = 1; col < dim + 1; col++) {
        fprintf(stderr, "%d ", P[row][col]);
      }
      fprintf(stderr, "\n");
    }
  }
  close(fd[0]);
  close(fd[1]);
  exit(EXIT_SUCCESS);

fail:
  fprintf(stderr, "%s aborted\n", argv[0]);
  close(fd[0]);
  close(fd[1]);
  exit(EXIT_FAILURE);
}