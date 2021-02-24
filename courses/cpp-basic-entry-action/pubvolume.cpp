/*
定义一个 volume 类，其中包含三个 double 型的公有成员 width、length、high。

在主函数中定义一个用于访问 volume 类成员的对象 volume1 和一个用于保存体积结果的双精度型变量 VOLUME。

返回 width 为 3、length 为 4，high 为 5.1 的体积结果。

目标
定义一个包含三个 double 型公有成员的类，主函数访问类的公有成员实现体积计算。

提示
一般使用 class 关键字来定义类。
对象的声明形式为：类名 对象名。
访问类的数据成员：对象名.成员名。
*/

#include<iostream>
using namespace std;

class volume
{
    public: //定义公有成员
        double width;
        double length;
        double high;
};

int main()
{
    volume volume1; //定义对象 volume1。
    double VOLUME; //定义双精度型变量 VOLUME。
    volume1.width=3; //外部访问公有成员，设置宽为 3。
    volume1.length=4; //外部访问公有成员，设置长为 4。
    volume1.high=5.1; //外部访问公有成员，设置高为 5.1。
    VOLUME=volume1.width*volume1.length*volume1.high; //计算体积。
    cout << VOLUME <<endl; //输出体积。
    return 0;
}