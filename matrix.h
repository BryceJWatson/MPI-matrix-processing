/* basic matrix library for  comp381                            */
/* written by ian a. mason @ une  march 15  '99                 */            
/* added to easter '99                                          */  

int get_slot(int fd, int matrix_size, int row, int col, int *slot);
int set_slot(int fd, int matrix_size, int row, int col, int value);


int get_row(int fd, int matrix_size, int row, int matrix_row[]);
int set_row(int fd, int matrix_size, int row, int matrix_row[]);


