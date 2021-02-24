/*
请编写一个 C++ 程序：

自定义减法函数 subtra(int a,int b)，并在主函数中调用此函数，返回 subtra(9,5) 的结果。

目标
自定义减法函数，并按要求调用函数。
*/

#include<iostream>
using namespace std;

int subtra(int a,int b)
{
    return a - b;
}

int main()
{
    cout<<subtra(9,5)<<endl;//输出 c 值。
    return 0;
}