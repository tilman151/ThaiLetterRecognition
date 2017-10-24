/*
 * SWT.cpp
 *
 *  Created on: 11.09.2017
 *      Author: tilman
 */

#include "SWT.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "util.h"

namespace thai {

SWT::SWT() {
	// TODO Auto-generated constructor stub

}

SWT::~SWT() {
	// TODO Auto-generated destructor stub
}

void SWT::detectStroke(InputArray 	_src,
		  	  	  	   OutputArray 	_dst,
					   int 			threshold1,
					   int 			threshold2,
					   int 			maxStrokeWidth,
					   int			direction) {
	// TODO Write documentation

	const Size size = _src.size();
	const Rect imageBoundary = Rect(Point(), size);
	Mat src = _src.getMat();

	direction /= abs(direction);

	// Set dst to same size as src with maxInt as initial value
	_dst.create(size, CV_32FC1);
	Mat dst = _dst.getMat();
	dst = Scalar::all(numeric_limits<float>::max());

	// Use Canny edge detector on src
	Mat detectedEdges;
	Mat gradX;
	Mat gradY;
	Mat gradAngular;
	blur(src, detectedEdges, Size(3, 3));
	if(src.type() != CV_8UC1)
	{
		cvtColor( detectedEdges, detectedEdges, CV_BGR2GRAY);
	}
	// Compute gradient direction
	Sobel(detectedEdges, gradX, CV_16S, 1, 0, 3);
	Sobel(detectedEdges, gradY, CV_16S, 0, 1, 3);
	gradX.convertTo(gradX, CV_32F);
	gradY.convertTo(gradY, CV_32F);
	phase(gradX, gradY, gradAngular, false);
	Canny( detectedEdges, detectedEdges, threshold1, threshold2);

	// For each edge pixel p
	Mat edgePixel;
	findNonZero(detectedEdges, edgePixel);
	vector<vector<Point>> savedRays = vector<vector<Point>>();
	for(int p = 0; p < edgePixel.rows; p++)
	{
		Point pointP = edgePixel.at<Point>(p);
		Point pointQ = Point(-1, -1);
		Point currP;
		vector<Point> currRay = vector<Point>();
		currRay.reserve(maxStrokeWidth);
		double currGrad = gradAngular.at<float>(pointP);

		// For each pixel in the gradient direction ray
		for(int n = 1; n < maxStrokeWidth; n++)
		{
			currP = pointP + direction*Point(round(n*cos(currGrad)),
								   	   	     round(n*sin(currGrad)));

			if(!imageBoundary.contains(currP))
			{
				break;
			}

			currRay.push_back(currP);

			// Check if edge pixel q
			if(detectedEdges.at<uchar>(currP) == 255)
			{
				// If gradient of q approximately the same as gradient of p
				double antiGrad = fmod(
						gradAngular.at<float>(currP)+M_PI, 2*M_PI);
				if(currGrad < antiGrad+M_PI/6 && currGrad > antiGrad-M_PI/6)
				{
					// Found other side of stroke
					pointQ = currP;
					break;
				}
				else
				{
					break;
				}
			}
		}
		if(pointQ != Point(-1, -1))
		{
			// Stroke width is |p-q|
			double strokeWidth = norm(pointP - pointQ);

			// Set stroke width of pixel on ray to |p-q| if it is smaller
			// than current stroke width
			for(uint n = 0; n < currRay.size(); n++)
			{
				if(dst.at<float>(currRay[n]) > strokeWidth)
				{
					dst.at<float>(currRay[n]) = strokeWidth;
				}
			}

			// Save ray
			savedRays.push_back(currRay);
		}

	}

	// For each ray
	for(uint r = 0; r < savedRays.size(); r++)
	{
		vector<float> pointValues = vector<float>();
		pointValues.reserve(savedRays[r].size());
		// Compute median stroke width m
		for(uint p = 0; p < pointValues.capacity(); p++)
		{
			pointValues.push_back(dst.at<float>(savedRays[r][p]));
		}
		nth_element(pointValues.begin(),
					pointValues.begin() + pointValues.size()/2,
					pointValues.end());
		double medianSW = pointValues[pointValues.size()/2];

		// Set stroke width on ray greater than m to m
		for(uint p = 0; p < pointValues.size(); p++)
		{
			if(dst.at<float>(savedRays[r][p]) > medianSW)
			{
				dst.at<float>(savedRays[r][p]) = medianSW;
			}
		}
	}

	dst.copyTo(_dst);

}

void SWT::detectStrokeInBinary(InputArray	_src,
							   OutputArray	_dst)
{

	// Get image
	const Size size = _src.size();
	Mat src = _src.getMat();

	// Check if binary
	CV_Assert(src.type() == CV_8UC1);

	// Calculate distance transform
	Mat dst;
	distanceTransform(src, dst, CV_DIST_L2, 5);
	dst.convertTo(dst, CV_8UC1);

	vector<Point> fgPixels;
	findNonZero(src, fgPixels);
	double min, max;
	minMaxLoc(dst, &min, &max);

	// For each stroke width
	for(uchar sw = max; sw > 0; sw--)
	{
		// Look up pixel with current stroke width
		for(uint p = 0; p < fgPixels.size(); p++)
		{
			// Set to current stroke width
			queue<Point> q = queue<Point>();
			if(dst.at<uchar>(fgPixels[p]) != sw)
			{
				continue;
			}
			q.push(fgPixels[p]);
			// For each neighbor
			while(!q.empty())
			{
				Point front = q.front();
				q.pop();
				Point west = front;
				// Check west
				while(west.x > 0 &&
					  dst.at<uchar>(west+Point(-1,0)) > 0 &&
					  dst.at<uchar>(west+Point(-1,0)) < dst.at<uchar>(west))
				{
					// Check north
					Point north = west + Point(0,-1);
					if(north.y > 0 &&
					   dst.at<uchar>(north) > 0 &&
					   dst.at<uchar>(north) < dst.at<uchar>(front))
					{
						q.push(north);
					}
					// Check south
					Point south = west + Point(0,1);
					if((south.y < size.height) &&
					   (dst.at<uchar>(south) > 0) &&
					   (dst.at<uchar>(south) < dst.at<uchar>(front)))
					{
						q.push(south);
					}
					// Move to west
					west.x--;
				}
				// Check east
				Point east = front;
				while((east.x < size.width) &&
					  (dst.at<uchar>(east+Point(1,0)) > 0) &&
					  (dst.at<uchar>(east+Point(1,0)) < dst.at<uchar>(east)))
				{
					// Check north
					Point north = east + Point(0,-1);
					if(north.y > 0 &&
					   dst.at<uchar>(north) > 0 &&
					   dst.at<uchar>(north) < dst.at<uchar>(front))
					{
						q.push(north);
					}
					// Check south
					Point south = east + Point(0,1);
					if((south.y < size.height) &&
					   (dst.at<uchar>(south) > 0) &&
					   (dst.at<uchar>(south) < dst.at<uchar>(front)))
					{
						q.push(south);
					}
					// Move to east
					east.x++;
				}
				// Fill west to east
				line(dst, west, east, Scalar(sw));
			}
		}
		//normalize(dst, show, 0, 255, NORM_MINMAX, CV_8UC1);
		//imshow( "Image", show);
		//waitKey(0);
	}

	dst.copyTo(_dst);

}

void SWT::detectStrokeInBinaryPointers(InputArray		_src,
							   	   	   InputOutputArray	_dst)
{

	// Get image
	const Size size = _src.size();
	Mat src = _src.getMat();

	// Check if binary
	CV_Assert(src.type() == CV_8UC1);

	// Calculate distance transform
	Mat dst;
	distanceTransform(src, dst, CV_DIST_L2, 5);
	dst.convertTo(dst, CV_8SC1);

	// Initialize 8-neighborhood mask
	const vector<int> neighborMask = {-size.width,
									  -size.width-1,
									  1,
									  size.width+1,
									  size.width,
									  size.width-1,
									  -1};


	vector<Point> fgPixels;
	findNonZero(src, fgPixels);
	vector<MatIterator_<char>> dstPointers;
	dstPointers.reserve((fgPixels.size()));
	Mat_<char>& dstRef = (Mat_<char>&)dst;
	for(uint i = 0; i < fgPixels.size(); i++)
	{
		dstPointers.push_back(MatIterator_<char>(&dstRef, fgPixels[i]));
	}

	double min, max;
	minMaxLoc(dst, &min, &max);
	queue<MatIterator_<char>> pointerQueue = queue<MatIterator_<char>>();

	namedWindow( "Image", WINDOW_AUTOSIZE);
	imshow( "Image", dst);
	waitKey(0);

	// For each stroke width
	for(char sw = max; sw > 0; sw--)
	{
		cout << (int)sw << "\n";
		// Look up pixel with current stroke width
		for(uint p = 0; p < dstPointers.size(); p++)
		{
			// If pixel has current stroke width
			if(*dstPointers[p] < sw)
			{
				continue;
			}
			// Add neighbors with smaller stroke width to queue
			for(uint i = 0; i < neighborMask.size(); i++)
			{
				MatIterator_<char> neighborPrt = dstPointers[p] +
												 neighborMask[i];
				if(neighborPrt.pos().y < size.height &&
				   *neighborPrt 	   > 0			 &&
				   *neighborPrt 	   < sw)
				{
					pointerQueue.push(neighborPrt);
				}
			}
			cout << "\t" << dstPointers[p].pos().x << ", " << dstPointers[p].pos().y << "\n";
			// For each neighbor
			while(!pointerQueue.empty())
			{
				// Set to current stroke width
				MatIterator_<char> dstPtr = pointerQueue.front();
				pointerQueue.pop();
				*dstPtr = sw;
				// Add neighbors to queue
				for(uint i = 0; i < neighborMask.size(); i++)
				{
					MatIterator_<char> neighborPrt = dstPtr + neighborMask[i];
					if(neighborPrt.pos().y < size.height &&
					   *neighborPrt 	   > 0 			 &&
					   *neighborPrt 	   < sw)
					{
						pointerQueue.push(neighborPrt);
					}
				}
			}
		}
		imshow( "Image", dst);
			waitKey(0);

	}


	dst.copyTo(_dst);

}

void SWT::connectStroke(InputArray				_src,
						OutputArray				_dst,
					    vector<vector<Point>>&	letterCandidates,
					    vector<Rect>&			bBoxes,
					    double					maxRatio) {

	// Get stroke image
	const Size size = _src.size();
	const Rect imageBoundary = Rect(Point(), size);
	Mat src = _src.getMat();

	// Erode image to fill gaps
	int erosionSize = 1;
	Mat element = getStructuringElement(MORPH_ELLIPSE,
	                                    Size( 2*erosionSize+1, 2*erosionSize+1),
	                                    Point( erosionSize, erosionSize ) );
	erode( src, src, element);

	// Set current label to one
	const vector<Point> neighborMask = {Point(0,-1),
										Point(0,1),
										Point(-1,0),
										Point(1,0)};
	int currLabel = 1;
	Mat labels = Mat::zeros(size, CV_16SC1);
	letterCandidates = vector<vector<Point>>();
	letterCandidates.push_back(vector<Point>());
	queue<Point> pointQueue = queue<Point>();

	// For each pixel
	MatIterator_<float> iterator, end;
	MatIterator_<short> labelIterator = labels.begin<short>();
	for(iterator = src.begin<float>(), end = src.end<float>();
		iterator != end;
		++iterator, ++labelIterator)
	{
		// If pixel float max or labeled
		if(*iterator == numeric_limits<float>::max() || *labelIterator != 0)
		{
			continue;
		}

		// Add pixel to queue
		pointQueue.push(iterator.pos());
		letterCandidates[currLabel-1].push_back(iterator.pos());

		// For each element in queue
		Point currP;
		while(!pointQueue.empty())
		{
			currP = pointQueue.front();
			pointQueue.pop();
			float currVal = src.at<float>(currP);
			// For each neighbor
			for(uint n = 0; n < neighborMask.size(); n++)
			{
				Point neighborP = currP+neighborMask[n];
				if(!imageBoundary.contains(neighborP))
				{
					continue;
				}
				float neighborVal = src.at<float>(neighborP);
				// If stroke width ratio is smaller maxRatio and not labeled
				if((neighborVal != numeric_limits<float>::max()) &&
				   (neighborVal/currVal < maxRatio) &&
				   (neighborVal/currVal > 1/maxRatio) &&
				   (labels.at<short>(neighborP) == 0))
				{
					// Add to queue and label
					pointQueue.push(neighborP);
					labels.at<short>(neighborP) = currLabel;
					letterCandidates[currLabel-1].push_back(neighborP);
				}

			}
		}

		// Calculate bounding box
		bBoxes.push_back(boundingRect(letterCandidates[currLabel-1]));

		// Increment label
		currLabel++;
		letterCandidates.push_back(vector<Point>());
	}

	// Write label matrix
	labels.copyTo(_dst);

}

void SWT::testStrokeWidthVariance(vector<vector<Point>>&	regions,
								  Mat&	 					strokeImage,
								  double					maxRatio)
{

	vector<Rect> bBoxes;
	for(uint i = 0; i < regions.size(); i++)
			{
				bBoxes.push_back(boundingRect(regions[i]));
			}

	vector<uint> validIdx = vector<uint>();
	Mat currentROI;
	Scalar currMean, currSdev;
	for(uint r = 0; r < regions.size(); r++)
	{
		currentROI = strokeImage(bBoxes[r]);
		meanStdDev(currentROI, currMean, currSdev, currentROI > 0);
		if(currSdev[0]/currMean[0] < maxRatio)
		{
			validIdx.push_back(r);
		}
	}

	vector<vector<Point>> validRegions = vector<vector<Point>>();
	for(uint i = 0; i < validIdx.size(); i++)
	{
		validRegions.push_back(regions[validIdx[i]]);
	}

	regions = validRegions;

}

void SWT::testStrokeWidthVariance(vector<LetterCandidate>&	letters,
								  double					maxRatio)
{

	vector<uint> validIdx = vector<uint>();
	for(uint r = 0; r < letters.size(); r++)
	{
		if(letters[r].sDevStrokeWidth/letters[r].meanStrokeWidth < maxRatio)
		{
			validIdx.push_back(r);
		}
	}

	vector<LetterCandidate> validRegions = vector<LetterCandidate>();
	for(uint i = 0; i < validIdx.size(); i++)
	{
		validRegions.push_back(letters[validIdx[i]]);
	}

	letters = validRegions;

}

} /* namespace thai */
