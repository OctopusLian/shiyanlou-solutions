/*
请编写一个 C++ 程序：

定义整型指针变量 p，然后使用 new 关键字为其分配 int 型的内存空间，并让指针 p 指向分配的内存空间。

随后将 9 存入内存块中，且输出其存入的值。

最后释放指针变量 p。

目标
使用 new 关键字和 delete 关键字管理堆内存。

提示
new 关键字 用于堆内存的分配；delete 关键字 用于堆内存的释放。
*/

#include <iostream>
using namespace std;

int main()
{
    int *p;
    p = new int;
    *p = 9;
    cout<<*p<<endl;
    delete p;
    return 0;
}