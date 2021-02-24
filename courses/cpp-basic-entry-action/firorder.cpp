/*
请编写一个 C++ 程序：

使用 for 语句计算 4 的阶乘。

目标
在屏幕上输出阶乘的结果
提示
语句 sub _= i 等于 sub = sub _ i。
*/

#include<iostream>
using namespace std;

int main()
{
    int sub = 1;
    int i;
    for(i=2;i<5;i++)
    {
        sub *= i; //执行 sub = sub * i。
    }
    cout<<sub<<endl; //输出 sub 的值。
    return 0;
}