#include <stdio.h>
#include <stdlib.h>

//直接根据数学公式的定义，实现递归函数
long long fabonacci(int n)
{
    if (n == 0) return 0; // 基准情况 0
    else if (n == 1) return 1;// 基准情况 1
    else return fabonacci(n-1) + fabonacci(n-2); //分解成小问题递归求解
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
 * $ gcc -o fibonacciv1 fibonacciv1.c
 * /