#include <opencv2/opencv.hpp>
#include <iostream>
#include <string.h>
#include <math.h>
#include <opencv/cv.h>

using namespace cv;
using namespace std;
//opencv�Դ����ֺ�����cv::integral()
int main()
{	
	string floder_path = "/your-path/";
	vector<cv::String> Ori_filenames;
	Ori_filenames.clear();
	glob(floder_path + "//*.jpg", Ori_filenames, false);
	size_t count = Ori_filenames.size();
	for (int i = 0; i < count; i++)
	{
		Mat srcImage = imread(Ori_filenames[i]);
		Mat sum = Mat::zeros(srcImage.rows + 1, srcImage.cols + 1, CV_32FC1); //����ȫ�����ͼfloat32
		Mat sqsum = Mat::zeros(srcImage.rows + 1, srcImage.cols + 1, CV_64FC1);  //ƽ������ֵ�Ļ���ͼ��float64
		integral(srcImage, sum, sqsum, CV_32FC1, CV_64FC1);
		// ��һ����ʾ
		Mat result;
		normalize(sum, result, 0, 255, NORM_MINMAX, CV_8UC1);  
		imwrite(Ori_filenames[i], result);
		cout << Ori_filenames[i] << endl;
	}
	return 0;
}
