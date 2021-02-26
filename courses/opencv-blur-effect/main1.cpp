#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define TWO_PI 6.2831853071795864769252866

double generateGaussianNoise()    //对高斯分布的代码实现，
{
    static bool hasSpare = false;
    static double rand1, rand2;
    if(hasSpare)
    {
        hasSpare = false;
        return sqrt(rand1) * sin(rand2);
    }
    hasSpare = true;
    rand1 = rand() / ((double) RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI;
    return sqrt(rand1) * cos(rand2);
}
void AddGaussianNoise(Mat& I)
{
    CV_Assert(I.depth() != sizeof(uchar)); //只接受char类型的矩阵
    int channels = I.channels();   //返回 该图像的通道数目。
    int nRows = I.rows;       //  图像的行数
    int nCols = I.cols * channels;  //图像的列数*通道数目
    if(I.isContinuous()){//如果在每一行的结尾无间隙连续存储矩阵的元素，该方法返回 true。否则，它将返回 false。很明显，1 x 1 或 1xN 矩阵始终是连续的。
        nCols *= nRows; //  求出一共有多少像素点
        nRows = 1;  //行数置为 1
    }
    int i,j;
    uchar* p;
    for(i = 0; i < nRows; ++i){    //将这些像素点添加
        p = I.ptr<uchar>(i);
        for(j = 0; j < nCols; ++j){
            double val = p[j] + generateGaussianNoise() * 128;
            if(val < 0)     // 将像素的取值限制到0-255
                val = 0;
            if(val > 255)
                val = 255;
            p[j] = (uchar)val;

        }
    }
}



    int main()
    {
        Mat image;
        image = imread("lena.jpg"); // Read the file

        if(! image.data ) // Check for invalid input
        {
            cout << "Could not open or find the image" << std::endl ;
            return -1;
        }

        namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
        imshow( "Display window", image ); // Show our image inside it.

        // Add Gaussian noise here
        AddGaussianNoise(image);

        namedWindow( "Noisy image", WINDOW_AUTOSIZE ); // Create a window for display.
        imshow( "Noisy image", image ); // Show our image inside it.
        waitKey(0); // Wait for a keystroke in the window


        return 0;
    }
