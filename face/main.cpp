#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <sstream>
#include <string>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>

#ifdef WIN32
#include <io.h>
#else
#endif

using namespace cv;
using namespace std;
/** Function Headers */
void detectAndDisplay(Mat frame, size_t imgNum);
string int2str(size_t n);
/** Global variables */
//-- Note, either copy this file from opencv/data/haarscascades to your current folder, or change these locations
String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Face detection";

void ReadDirectory(const string& directoryName, const string fileExt, vector<string>& filenames, bool addDirectoryName = true)
{
	filenames.clear();

#ifdef WIN32
	struct _finddata_t s_file;
	string str = directoryName + "\\*" + fileExt;

	intptr_t h_file = _findfirst(str.c_str(), &s_file);
	if (h_file != static_cast<intptr_t>(-1.0))
	{
		do
		{
			if (addDirectoryName)
				filenames.push_back(directoryName + "\\" + s_file.name);
			else
				filenames.push_back((string)s_file.name);
		} while (_findnext(h_file, &s_file) == 0);
	}
	_findclose(h_file);
#else

#endif

	sort(filenames.begin(), filenames.end());
}

/**
* @function main
*/
int main(void)
{

	vector<string> fileNames;
	ReadDirectory("input", ".jpg", fileNames);
	for (int i = 0; i<fileNames.size(); i++)
	{
		printf("%s \n", fileNames[i].c_str());


		//-- 1. Load the cascade
		if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
		//-- 2. Read the image
		IplImage* img = cvLoadImage(fileNames[i].c_str(), CV_LOAD_IMAGE_COLOR);
		Mat frame(img);
		//-- 3. Apply the classifier to the frame
		if (!frame.empty())
		{
			detectAndDisplay(frame,i);
		}
		else
		{
			printf("--(!)Error!\n");
		}
		waitKey(500);
		cvDestroyWindow(window_name.c_str());
		cvReleaseImage(&img);



	}
	printf("ok \n");

	return 0;

}
/**
* @function detectAndDisplay
*/
void detectAndDisplay(Mat frame, size_t imgNum)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	double t = (double)cvGetTickCount();
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	t = (double)cvGetTickCount() - t;
	printf("%gms\n", t / ((double)cvGetTickFrequency()*1000.0));

	/*for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		printf("Found a face at (%d, %d)\n", center.x, center.y);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 255, 255), 2, 8, 0);
	}*/

	for (size_t i = 0; i < faces.size(); i++)
	{
		Rect r = faces[i];
		Mat roi = frame(r);
		string name = "./output/image" + int2str(imgNum) + "u" + int2str(i) + ".jpg";
		imwrite(name, roi);
	}
	for (size_t i = 0; i < faces.size(); i++)
	{
		Rect r = faces[i];
		rectangle(frame, r.tl(), r.br(), Scalar(0, 0, 255), 3);
	}
	//-- Show what you got
	imshow(window_name, frame);
}

string int2str(size_t n) {
	stringstream ss;
	ss << n;
	string result;
	ss >> result;
	return result;
}
