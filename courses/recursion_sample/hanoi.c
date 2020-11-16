#include <stdio.h>

// 移动一个盘
// disk为需要移动的盘符，src为源杆，dest为目标杆
void move_single_disk(int disk, char src, char dest)
{
    static step = 1;
    fprintf(stdout, "step%d: disk%d %c --> %c\n", step++,disk,src,dest);    
}
// 汉诺塔函数，递归方式
// n个盘，n个盘，盘符由小到大为1——>N, 从A杆移动到C杆
// disk 为最大盘的盘符
void hanoi(int n, int disk, char A, char B, char C)
{
     // 基准情况
     if (1 == n) {
         move_single_disk(disk ,A, C);
     } else {
         // 解决子问题一
         hanoi(n-1, disk-1, A, C, B);
         // 解决子问题二
         hanoi(1, disk, A, B, C);
         // 解决子问题三
         hanoi(n-1, disk-1, B, A, C);
     }
}

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    fprintf(stdout, "=======hanoi(%d):\n", n);
    hanoi(n, n, 'A', 'B', 'C');
    fprintf(stdout, "=======hanoi finished\n");
    return 0;
}

/**
 * $ gcc -o hanoi hanoi.c
 * /