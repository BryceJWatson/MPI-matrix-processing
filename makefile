COMPILER = mpicc
CFLAGS = -Wall -pedantic -g
EXES = a3 getMatrix mkRandomMatrix

all: ${EXES}

a3: a3.c matrix.o
	${COMPILER} ${CFLAGS} a3.c matrix.o -o a3 -lm
	
getMatrix: getMatrix.c matrix.o
	${COMPILER} ${CFLAGS} getMatrix.c matrix.o -o getMatrix

mkRandomMatrix: mkRandomMatrix.c  matrix.o
	${COMPILER} ${CFLAGS} mkRandomMatrix.c matrix.o -o mkRandomMatrix

matrix.o: matrix.c matrix.h
	${COMPILER} ${CFLAGS} -c matrix.c

%.o: %.c %.h  makefile
	${COMPILER} ${CFLAGS} $< -c 

clean:
	rm -f *~ *.o ${EXES}

run:
	mpirun -np 4 ./a3 test_matrix test_results 4