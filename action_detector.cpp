#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/gapi/ocl/core.hpp>
#include <opencv2/gapi/ocl/imgproc.hpp>
#include <opencv2/gapi/ocl/goclkernel.hpp>
#include <string>
#include <vector>
#include <tuple>

// カメラ起動用の引数(機器によって異なる)
#define DEVICE_VAR "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)15/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! videoflip method=rotate-180 !  appsink"
// キャプチャするカメラの解像度
#define IMAGE_HEIGHT 1080
#define IMAGE_WIDTH 1920

// 動体検出に用いる閾値 低いほど感度が上がる(ノイズは増える)
#define THRESH 50
// 動体検出に用いる移動平均の減衰率 低いほど感度が鈍る(ノイズは減る)
#define ATTENUATION 0.2

int main(int argc, char *argv[])
{
    //*************************************************************************
    //*カメラ、GPUの確認
    int dev_num = cv::cuda::getCudaEnabledDeviceCount();
    int device = cv::cuda::getDevice();
    cv::cuda::setDevice(device);
    std::cout << "hello! gpu num: " + std::to_string(dev_num) + " now use: " + std::to_string(device) << std::endl;
    // カメラを開く
    // カメラの引数は扱う機器によって異なるので注意
    cv::VideoCapture cap(DEVICE_VAR);

    CV_Assert(cap.isOpened());
    //*************************************************************************

    //*************************************************************************
    //* グラフにデータフローを記述。機械学習のレイヤー積み重ねと似たお気持ち
    cv::GComputation ac([]()
                        {
                        std::vector<cv::GMat> ins(2);                                 //[取得フレーム,ひとつまえのavg]
                        cv::GMat in = ins[0];
                        cv::GMat old_avg = ins[1];
                        cv::GMat vga = cv::gapi::resize(in, cv::Size(), 0.5, 0.5);
                        cv::GMat gray = cv::gapi::BGR2Gray(vga);
                        cv::GMat avg = cv::gapi::addWeighted(old_avg, ATTENUATION, gray, 1 - ATTENUATION, 0);
                        cv::GMat delta = cv::gapi::absDiff(gray, avg);
                        std::tuple<cv::GMat,cv::GScalar> thresh = cv::gapi::threshold(delta, THRESH+50, cv::THRESH_TRIANGLE);
                        cv::GMat unnoizy = cv::gapi::medianBlur(std::get<0>(thresh),3); // ブラーでノイズを除去する方法
                        // cv::GArray<cv::GArray<cv::Point>> contours_array = cv::gapi::findContours(std::get<0>(thresh),cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
                        // contours の GArray を GMat に描画する方法が謎なので回避．Cannyによるエッジ検出で実現．
                        cv::GMat edges = cv::gapi::Canny(unnoizy, THRESH, THRESH + 50, 3);
                        cv::GMat dilated_edges = cv::gapi::dilate3x3(edges);
                        cv::GMat b, g, r;
                        std::tie(b, g, r) = cv::gapi::split3(vga);        //カラー画像を3チャンネルに分割して
                        cv::GMat out = cv::gapi::merge3(b, g | dilated_edges, r); //各チャンネルをマージ. |はgとエッジでビット単位でのor演算。
                        std::vector<cv::GMat> outs{out, avg};
                        return cv::GComputation(ins, outs); });
    //*************************************************************************

    //*************************************************************************
    //*初期化 移動平均を使うので、最初の1フレームは普通に計算しておく
    std::vector<cv::Mat> input_frame(2);
    std::vector<cv::Mat> output_frame(2);
    CV_Assert(cap.read(input_frame[0]));
    cv::resize(input_frame[0], input_frame[1], cv::Size(), 0.5, 0.5);
    cv::cvtColor(input_frame[1], input_frame[1], cv::COLOR_BGR2GRAY);
    //*************************************************************************

    // 今のところGPUは使えなさそうだが、準備としてGPUカーネルを準備
    // GPUが無い場合はコメントアウトする
    cv::gapi::GBackend ocl_backend = cv::gapi::ocl::backend();
    cv::gapi::GKernelPackage ocl_kernels = cv::gapi::combine // Define a custom kernel package:
        (cv::gapi::core::ocl::kernels(),                     // ...with ocl Core kernels
         cv::gapi::imgproc::ocl::kernels());                 // ...and ocl ImgProc kernels

    //*************************************************************************
    //*画像処理の実行部分
    do
    {
        //グラフにデータを流しこむ
        ac.apply(input_frame, output_frame, cv::compile_args(ocl_kernels));
        cv::imshow("action_detector", output_frame[0]);
        // avgのフィードバック
        input_frame[1] = output_frame[1];
    } while (cap.read(input_frame[0]) && cv::waitKey(30) < 0);
    //*************************************************************************

    cap.release();
    cv::destroyAllWindows();
    return 0;
}