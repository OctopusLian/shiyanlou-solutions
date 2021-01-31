#include <stdio.h>
#include <stdlib.h>

long long sequence(int n, long long t0, long long t1)
{
    if (n == 0) return t0;
    if (n == 1) return t1;
    return (sequence(n-1, t1, t0+ t1));
}

long long fabonacci(int n)
{
    return sequence(n, 0, 1);
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N.\n", argv[0]);
        return 0;
    }

    int n = atoi(argv[1]);
    fprintf(stdout, "The %dth item in the Fibonacci sequence is %lld.\n",n,fabonacci(n));
    return 0;
}

/**
 * $ gcc -o fabonacciv2 fabonacciv2.c
 * /