/*
定义且初始化一个二维数组：int acc[2][3] = {2,4,6,8,10,12}。
访问数组元素 8。
目标
定义和初始化一个二维数组，并对指定的元素进行访问。

提示
访问二维数组的表达式为：数组名[index][index]。
初始化二维表的顺序：acc[0][0]、acc[0][1]、acc[0][2]...
*/

#include<iostream>
using namespace std;

int main ()
{
    int acc[2][3] = {2,4,6,8,10,12}; //初始化数组。
    cout<<acc[1][0]<<endl; //访问数组。
    return 0;
}