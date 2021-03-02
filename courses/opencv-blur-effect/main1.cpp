#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define TWO_PI 6.2831853071795864769252866

double generateGaussianNoise()    //�Ը�˹�ֲ��Ĵ���ʵ�֣�
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
    CV_Assert(I.depth() != sizeof(uchar)); //ֻ����char���͵ľ���
    int channels = I.channels();   //���� ��ͼ���ͨ����Ŀ��
    int nRows = I.rows;       //  ͼ�������
    int nCols = I.cols * channels;  //ͼ�������*ͨ����Ŀ
    if(I.isContinuous()){//�����ÿһ�еĽ�β�޼�϶�����洢�����Ԫ�أ��÷������� true�������������� false�������ԣ�1 x 1 �� 1xN ����ʼ���������ġ�
        nCols *= nRows; //  ���һ���ж������ص�
        nRows = 1;  //������Ϊ 1
    }
    int i,j;
    uchar* p;
    for(i = 0; i < nRows; ++i){    //����Щ���ص�����
        p = I.ptr<uchar>(i);
        for(j = 0; j < nCols; ++j){
            double val = p[j] + generateGaussianNoise() * 128;
            if(val < 0)     // �����ص�ȡֵ���Ƶ�0-255
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