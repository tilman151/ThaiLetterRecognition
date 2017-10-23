#include <opencv2/opencv.hpp>
#include "SWT.h"
#include "util.h"
#include "LetterCandidate.h"
#include <math.h>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
    {

	// Read image from file
	Mat image;
	image = imread( argv[1], 1 );

	// Check if image valid
	if( argc != 2 || !image.data )
	{
	  printf( "No image data \n" );
	  return -1;
	}
	resize(image, image, Size(1440, 1080), 0, 0, 1);

	// Start time measurement
	double t = (double)getTickCount();

	// Convert to gray scale
	Mat srcGray;
	cvtColor( image, srcGray, CV_BGR2GRAY);

	// Perform MSER detection on gray scale image
	Ptr<MSER> ms = MSER::create(4, 40, 8000, 0.05, 1.25, 200, 1.01, 0.003, 5);
	vector<vector<Point> > regions;
	vector<cv::Rect> mserBbox;
	ms->detectRegions(srcGray, regions, mserBbox);
	cout << "Found " << regions.size() << " MSER regions" << "\n";

	// Draw and re-extract MSER regions
	Mat mserStroke = Mat(image.size(), CV_8UC1);
	for(uint i = 0; i < regions.size(); i++)
	{
		for(uint j = 0; j < regions[i].size(); j++)
		{
			mserStroke.at<uchar>(regions[i][j]) = 255;
		}
	}

	vector<Vec4i> hierarchy = vector<Vec4i>();
	findContours(mserStroke, regions, hierarchy,
				 CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	cout << "Merging left " << regions.size() << " regions\n";

	// Filter regions based on geometric region properties
	vector<vector<Point>> validRegions;
	getValidRegionIdx(regions, validRegions);
	cout << "Geometric tests left " << validRegions.size() << " regions" << "\n";

	// Free memory
	//regions.clear();

	mserStroke = Mat::zeros(mserStroke.size(), CV_8UC1);
	drawContours(mserStroke, validRegions, -1, Scalar(255), -1);

	// Calculate stroke width based on MSER regions
	Mat outStroke;
	SWT SWDetector = SWT();
	SWDetector.detectStrokeInBinary(mserStroke, outStroke);

	vector<LetterCandidate> letters;
	regionsToLetterCandidates(validRegions, letters, outStroke);
	SWDetector.testStrokeWidthVariance(letters, 0.5);
	cout << "Stroke width filtering left " << letters.size() << " regions\n";

	vector<TextLine> textLines;
	formWordLines(letters, textLines);
	for(uint i = 0; i < textLines.size(); i++)
	{
		cout << "Line " << i << ": ";
		for(uint j = 0; j < textLines[i].size(); j++)
		{
			cout << textLines[i][j].getIndex() << ", ";
		}
		cout << "\n";
	}

	// Print time measurement
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Times passed in seconds: " << t << endl;

	for(uint i = 0; i < textLines.size(); i++)
	{
		if(textLines[i].size() < 3)
		{
			continue;
		}
		Rect box = Rect();
		for(uint j = 0; j < textLines[i].size(); j++)
		{
			box = box | textLines[i][j].getBBox();
		}
		rectangle(image, box, Scalar(0, 0, 255), 2);
	}

	for(uint i = 0; i < letters.size(); i++)
	{
		rectangle(image, letters[i].getMainBoundary(), Scalar(0, 255, 0), 1);

	}

	// Resize image to fit screen
	resize(image, image, Size(1200, 900), 0, 0, 1);

	// Display image
	namedWindow( "Display Image", WINDOW_AUTOSIZE);
	imshow( "Display Image", image);
	waitKey(0);
	return 0;
}
