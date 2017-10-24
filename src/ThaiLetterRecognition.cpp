#include <opencv2/opencv.hpp>
#include "LetterLocalizer.h"
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

	//LetterLocalizer call
	vector<thai::TextLine> textLines;
	thai::LetterLocalizer localizer = thai::LetterLocalizer();
	localizer.localizeText(image, textLines);

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

	// Resize image to fit screen
	resize(image, image, Size(1200, 900), 0, 0, 1);

	// Display image
	namedWindow( "Display Image", WINDOW_AUTOSIZE);
	imshow( "Display Image", image);
	waitKey(0);
	return 0;
}
