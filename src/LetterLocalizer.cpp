/*
 * LetterLocalizer.cpp
 *
 *  Created on: 24.10.2017
 *      Author: tilman
 */

#include "LetterLocalizer.h"

namespace thai {

LetterLocalizer::LetterLocalizer() {
	// TODO Auto-generated constructor stub

}

LetterLocalizer::~LetterLocalizer() {
	// TODO Auto-generated destructor stub
}

void LetterLocalizer::localizeText(Mat image, vector<TextLine>& textLines)
{
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
	//TODO Improve merging of regions
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

	textLines = vector<TextLine>();
	formWordLines(letters, textLines);
}

// Geometric validation tests for MSER regions
void LetterLocalizer::getValidRegionIdx(vector<vector<Point>>& regions,
					   	   	   	   	    vector<vector<Point>>& validRegions)
{

	validRegions.reserve(regions.size());

	double aspectRatio;
	double extend;
	vector<Point> hull;
	double solidity;
	RotatedRect ellipseRect;
	double eccentricity;
	for (uint i = 0; i < regions.size(); i++)
	{
		double area = contourArea(regions[i]);

		if(area < 5 || area > 500)
		{
			continue;
		}

		Rect box = boundingRect(regions[i]);
		aspectRatio = (double) box.width / (double) box.height;
		if(aspectRatio > 3.0)
		{
			continue;
		}

		extend = area / (double) (box.width * box.height);
		if(extend < 0.2 || extend > 0.9)
		{
			continue;
		}

		convexHull(regions[i], hull);
		solidity = (double) area /
				   (double) contourArea(hull);
		if(solidity < 0.3)
		{
			continue;
		}

		ellipseRect = fitEllipse(regions[i]);
		eccentricity = (double) ellipseRect.size.height /
					   (double) ellipseRect.size.width;
		if (eccentricity > 1.0)
		{
			eccentricity = 1.0 / eccentricity;
		}
		if(eccentricity > 0.9)
		{
			continue;
		}

		validRegions.push_back(regions[i]);
	}
}

void LetterLocalizer::regionsToLetterCandidates(vector<vector<Point>>&	regions,
							   	   	   	   	   vector<LetterCandidate>&	letters,
											   Mat					strokeImage)
{

	letters = vector<LetterCandidate>();
	letters.reserve(regions.size());

	Rect bBox;
	Mat currentROI;
	Scalar currMean, currSdev;
	for(uint i = 0; i < regions.size(); i++)
	{
		// Calculate bounding box and region of interest
		bBox = boundingRect(regions[i]);
		currentROI = strokeImage(bBox);

		// Expand bounding box for later text line forming
		bBox.x -= bBox.width / 3;
		bBox.y -= bBox.height / 2.5;
		bBox.width += bBox.width / 1.5;
		bBox.height += bBox.height / 1.1;

		// Create letter candidate
		letters.push_back(LetterCandidate(i, bBox));

		// Add stroke width information to letter candidate
		meanStdDev(currentROI, currMean, currSdev, currentROI > 0);
		letters.back().meanStrokeWidth = currMean[0];
		letters.back().sDevStrokeWidth = currSdev[0];
	}

}

void LetterLocalizer::formWordLines(vector<LetterCandidate>&	letters,
				   	   	   	   	    vector<TextLine>& textLines)
{

	vector<LetterPair> letterPairs = vector<LetterPair>();
	unordered_set<uint> skipSet = unordered_set<uint>();
	for(uint r1 = 0; r1 < letters.size(); r1++)
	{

		for(uint r2 = 0; r2 < letters.size(); r2++){
			if((!skipSet.empty() && skipSet.find(r2) != skipSet.end()) ||
			   (r1 == r2))
			{
				continue;
			}
			if(r1 == 70 && r2 == 72)
						cout << "Now\n";
			int pos = letters[r1].overlaps(letters[r2]);
			if(pos == 1 || pos == 3)
			{
				letters[r1].add(letters[r2]);
				skipSet.insert(r2);
				continue;
			}
			if(pos == 2 && letters[r1].sameLineAs(letters[r2]))
			{
				letterPairs.push_back(LetterPair(letters[r1], letters[r2]));
			}
		}

	}

	vector<int> labels;
	partition(letterPairs, labels, comparePairs);
	int numPartitions = *max_element(begin(labels), end(labels));
	textLines = vector<TextLine>(numPartitions+1, TextLine());
	for(uint i = 0; i < labels.size(); i++)
	{
		textLines[labels[i]].push_back(letterPairs[i].first);
		textLines[labels[i]].push_back(letterPairs[i].second);
	}

	for(uint i = 0; i < textLines.size(); i++)
	{
		TextLine::iterator it = unique(textLines[i].begin(),
									   textLines[i].end(),
									   compareLetterCandidate);
		textLines[i].resize(distance(textLines[i].begin(), it));
	}

}

} /* namespace thai */
