/*
   Copyright 2012 Will Sackfield

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#include "ImagePipeline.h"

using namespace IP;
using namespace cv;
using namespace std;

#define ANGLE(p1,p2,p3) (((p1.x-p3.x)*(p2.x-p3.x))+((p1.y-p3.y)*(p2.y-p3.y)))/sqrt(((pow(p1.x-p3.x,2.0)+pow(p1.y-p3.y,2.0))*(pow(p2.x-p3.x,2.0)+pow(p2.y-p3.y,2.0)))+1e-10)

void findSquares(const cv::Mat inputImage,const void* context)
{
	if(context == NULL || inputImage.type() != CV_8U)
		return;
	
	vector<vector<Point> >* contextVector = (vector<vector<Point> >*)context;
	vector<vector<Point> > contours;
	
	findContours(inputImage,contours,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
	
	for(int i=0;i<contours.size();i++)
	{
		vector<Point> approx;
		approxPolyDP(Mat(contours[i]),approx,arcLength(Mat(contours[i]),true)*0.02,true);
		if(approx.size() == 4 && fabs(contourArea(Mat(approx))) > 1000 && isContourConvex(Mat(approx)))
		{
			double maxCosine = 0.0;
			for(int j=2;j<5;j++)
				maxCosine = MAX(maxCosine,fabs(ANGLE(approx[j%4],approx[j-2],approx[j-1])));
			if(maxCosine < 0.3)
				contextVector->push_back(approx);
		}
	}
}

int main(int argc,char* argv[])
{
	ImageGraph graph;
	graph.addNode(downscaleImageBy2);
	graph.addNode(upscaleImageBy2);
	graph.addNode(splitChannels);
	graph.addNode(split11Thresholds);
	graph.addNode(findSquares);
	
	vector<vector<Point> > contextVector;
	Mat inputImage = imread("parking_sign.jpg");
	
	ImagePipeline pipeline(graph);
	pipeline.feed(inputImage,&contextVector);
	
	for(int i=0;i<contextVector.size();i++)
	{
		const Point* p = &contextVector[i][0];
		int n = (int)contextVector[i].size();
		polylines(inputImage,&p,&n,1,true,Scalar(0,255,0),3,CV_AA);
	}
	
	imwrite("parking_sign_squares.jpg",inputImage);
	
	return EXIT_SUCCESS;
}
