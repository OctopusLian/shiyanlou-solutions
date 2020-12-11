#include <opencv2/opencv.hpp>

bool selectObject = false;  // 用于标记是否有选取目标
int trackObject = 0;  // 1 表示有追踪对象 0 表示无追踪对象 -1 表示追踪对象尚未计算 Camshift 所需的属性
cv::Rect selection;  // 保存鼠标选择的区域
cv::Mat image;  // 用于缓存读取到的视频帧

// OpenCV 对所注册的鼠标回调函数定义为：
// void onMouse(int event, int x, int y, int flag, void *param)
// 其中第四个参数 flag 为 event 下的附加状态，param 是用户传入的参数，我们都不需要使用
// 故不填写其参数名
void onMouse( int event, int x, int y, int, void* ) {
    static cv::Point origin;
    if(selectObject) {
        // 确定鼠标选定区域的左上角坐标以及区域的长和宽
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
        // & 运算符被 cv::Rect 重载
        // 表示两个区域取交集, 主要目的是为了处理当鼠标在选择区域时移除画面外
        selection &= cv::Rect(0, 0, image.cols, image.rows);
    }
    switch(event) {
        // 处理鼠标左键被按下
        case CV_EVENT_LBUTTONDOWN:
            origin = cv::Point(x, y);
            selection = cv::Rect(x, y, 0, 0);
            selectObject = true;
            break;
        // 处理鼠标左键被抬起
        case CV_EVENT_LBUTTONUP:
            selectObject = false;
            if( selection.width > 0 && selection.height > 0 )
                trackObject = -1;  // 追踪的目标还未计算 Camshift 所需要的属性
            break;
    }
}

int main( int argc, const char** argv )
{
    // 创建一个视频捕获对象
    // OpenCV 提供了一个 VideoCapture 对象，它屏蔽了
    // 从文件读取视频流和从摄像头读取摄像头的差异，当构造
    // 函数参数为文件路径时，会从文件读取视频流；当构造函
    // 数参数为设备编号时(第几个摄像头, 通常只有一个摄像
    // 头时为0)，会从摄像头处读取视频流。
    cv::VideoCapture video("video.ogv");
    cv::namedWindow( "CamShift at Shiyanlou" );
    // 1. 注册鼠标事件的回调函数, 第三个参数是用户提供给回调函数的，也就是回调函数中最后的 param 参数
    cv::setMouseCallback( "CamShift at Shiyanlou", onMouse, 0 );

    // 读取文件
    // cv::VideoCapture video(0);        // 使用摄像头

    // 捕获画面的容器，OpenCV 中的 Mat 对象
    // OpenCV 中最关键的 Mat 类，Mat 是 Matrix(矩阵) 
    // 的缩写，OpenCV 中延续了像素图的概念，用矩阵来描述
    // 由像素构成的图像。
    cv::Mat frame, hsv, hue, mask, hist, backproj;  // 接收来自 video 视频流中的图像帧
    cv::Rect trackWindow;  // 追踪到的窗口  
    int hsize = 16;  // 计算直方图所必备的内容
    float hranges[] = {0,180};  // 计算直方图所必备的内容
    const float* phranges = hranges;  // 计算直方图所必备的内容
    // 2. 从视频流中读取图像
    while(true) {
        // 将 video 中的内容写入到 frame 中，
        // 这里 >> 运算符是经过 OpenCV 重载的
        video >> frame;
        // 当没有帧可继续读取时，退出循环
        if( frame.empty() )
            break;
        // 将frame 中的图像写入全局变量 image 作为进行 Camshift 的缓存
        frame.copyTo(image);
        // 转换到 HSV 空间
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        // 当有目标时开始处理
        if( trackObject ) {
            // 只处理像素值为H：0~180，S：30~256，V：10~256之间的部分，过滤掉其他的部分并复制给 mask
            cv::inRange(hsv, cv::Scalar(0, 30, 10), cv::Scalar(180, 256, 256), mask);
            // 下面三句将 hsv 图像中的 H 通道分离出来
            int ch[] = {0, 0};
            hue.create(hsv.size(), hsv.depth());
            cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);
            // 如果需要追踪的物体还没有进行属性提取，则对选择的目标中的图像属性提取
            if( trackObject < 0 ) {
                // 设置 H 通道和 mask 图像的 ROI
                cv::Mat roi(hue, selection), maskroi(mask, selection);
                // 计算 ROI所在区域的直方图
                calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                // 将直方图归一
                normalize(hist, hist, 0, 255, CV_MINMAX);
                // 设置追踪的窗口
                trackWindow = selection;
                // 标记追踪的目标已经计算过直方图属性
                trackObject = 1;
            }
            // 将直方图进行反向投影
            calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
            // 取公共部分
            backproj &= mask;
            // 调用 Camshift 算法的接口
            cv::RotatedRect trackBox = CamShift(backproj, trackWindow, cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
            // 处理追踪面积过小的情况
            if( trackWindow.area() <= 1 ) {
                int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                trackWindow = cv::Rect(trackWindow.x - r, trackWindow.y - r,
                                   trackWindow.x + r, trackWindow.y + r) &
                cv::Rect(0, 0, cols, rows);
            }
            // 绘制追踪区域
            ellipse( image, trackBox, cv::Scalar(0,0,255), 3, CV_AA );

        }
        // 如果正在选择追踪目标，则画出选择框
        if( selectObject && selection.width > 0 && selection.height > 0 ) {
            cv::Mat roi(image, selection);
            bitwise_not(roi, roi);  // 对选择的区域图像反色
        }
        // 显示当前帧
        imshow( "CamShift at Shiyanlou", image );
        // 录制视频帧率为 15, 等待 1000/15 保证视频播放流畅。
        // waitKey(int delay) 是 OpenCV 提供的一个等待函数，
        // 当运行到这个函数时会阻塞 delay 毫秒的时间来等待键盘输入
        char c = (char)cv::waitKey(1000/15.0);
        // 当按键为 ESC 时，退出循环
        if( c == 27 )
            break;
    }
    // 释放申请的相关内存
    cv::destroyAllWindows();
    video.release();
    return 0;
}