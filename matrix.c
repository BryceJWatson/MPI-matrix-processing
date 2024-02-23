/* basic matrix library for  comp381                            */
/* written by ian a. mason @ une  march 15  '99                 */
/* rows & columns are numbers 1 through dimension               */
#include "matrix.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int get_slot(int fd, int matrix_size, int row, int col, int *slot) {
  off_t offset = (((row - 1) * matrix_size) + (col - 1)) * sizeof(int);
  if (offset < 0) {
    fprintf(stderr, "offset overflow");
    return -1;
  } else if ((row > matrix_size) || (col > matrix_size)) {
    fprintf(stderr, "indexes out of range");
    return -1;
  } else if (lseek(fd, offset, 0) < 0) {
    perror("lseek failed");
    return -1;
  } else if (read(fd, slot, sizeof(int)) < 0) {
    perror("read failed");
    return -1;
  };
  return 0;
}

int set_slot(int fd, int matrix_size, int row, int col, int value) {
  off_t offset = (((row - 1) * matrix_size) + (col - 1)) * sizeof(int);
  if (offset < 0) {
    fprintf(stderr, "offset overflow");
    return -1;
  } else if ((row > matrix_size) || (col > matrix_size)) {
    fprintf(stderr, "indexes out of range");
  } else if (lseek(fd, offset, 0) < 0) {
    perror("lseek failed");
    return -1;
  } else if (write(fd, &value, sizeof(int)) < 0) {
    perror("write failed");
    return -1;
  };
  return 0;
}

int get_row(int fd, int matrix_size, int row, int matrix_row[]) {
  int column;
  off_t offset = ((row - 1) * matrix_size) * sizeof(int);
  if (offset < 0) {
    fprintf(stderr, "offset overflow");
    return -1;
  } else if (row > matrix_size) {
    fprintf(stderr, "index out of range");
    return -1;
  } else if (lseek(fd, offset, 0) < 0) {
    perror("lseek failed");
    return -1;
  } else
    for (column = 0; column < matrix_size; column++)
      if (read(fd, &matrix_row[column], sizeof(int)) < 0) {
        fprintf(stderr, "read failed column = %d\n", column);
        perror("read failed");
        return -1;
      };
  return 0;
}

int set_row(int fd, int matrix_size, int row, int matrix_row[]) {
  int column;
  off_t offset = ((row - 1) * matrix_size) * sizeof(int);
  if (offset < 0) {
    fprintf(stderr, "offset overflow");
    return -1;
  } else if (row > matrix_size) {
    fprintf(stderr, "indexes out of range");
  } else if (lseek(fd, offset, 0) < 0) {
    perror("lseek failed");
    return -1;
  } else
    for (column = 0; column < matrix_size; column++)
      if (write(fd, &matrix_row[column], sizeof(int)) < 0) {
        perror("write failed");
        return -1;
      };
  return 0;
}
