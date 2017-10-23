/*
 * util.cpp
 *
 *  Created on: 19.09.2017
 *      Author: tilman
 */

#include "util.h"
#include <opencv2/opencv.hpp>
#include <boost/functional/hash.hpp>
#include <unordered_set>

// Geometric validation tests for MSER regions
void getValidRegionIdx(vector<vector<Point>>& regions,
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

// Custom hash function for CV::Point
unsigned long hashCvPoint(const Point& p)
{
		unsigned long seed = 0;
		boost::hash_combine(seed, p.x);
		boost::hash_combine(seed, p.y);
		return seed;
}

bool compareCvPoints(const Point& a, const Point& b)
{
        return ( a.x==b.x && a.y==b.y );
}

bool comparePairs(const LetterPair& a, const LetterPair& b)
{
	return a.second == b.first;
}

bool compareLetterCandidate(const LetterCandidate& a, const LetterCandidate& b)
{
	return a.getIndex() == b.getIndex();
}

// Check list for intersecting regions and merge them
void removeApproxDuplicatesRegions(vector<vector<Point>>& regions,
					  	  	  	   double ratio,
								   vector<Rect>& bBoxes)
{

	if(bBoxes.size() == 0)
	{
		for(uint i = 0; i < regions.size(); i++)
		{
			bBoxes.push_back(boundingRect(regions[i]));
		}
	}

	vector<uint> validIdx = vector<uint>();
	unordered_set<uint> skipSet = unordered_set<uint>();
	for(uint r1 = 0; r1 < regions.size()-1; r1++)
	{

		if(skipSet.find(r1) != skipSet.end())
		{
			continue;
		}

		uint maxRegionIdx = r1;
		for(uint r2 = r1+1; r2 < regions.size(); r2++)
		{
			if((bBoxes[r1] & bBoxes[r2]).area() == 0)
			{
				continue;
			}

			Rect unionBound = bBoxes[r1] | bBoxes[r2];
			Point upperLeftBound = Point(unionBound.x, unionBound.y);
			Mat r1Map = Mat::zeros(unionBound.size(), CV_8UC1);
			Mat r2Map = Mat::zeros(unionBound.size(), CV_8UC1);
			for(uint p = 0; p < regions[r1].size(); p++)
			{
				r1Map.at<uchar>(regions[r1][p]-upperLeftBound) = 255;
			}

			for(uint p = 0; p < regions[r2].size(); p++)
			{
				r2Map.at<uchar>(regions[r2][p]-upperLeftBound) = 255;
			}

			Mat interMap;
			bitwise_and(r1Map, r2Map, interMap);

			int intersectionArea = countNonZero(interMap);

			if((double)intersectionArea / (double)regions[r1].size() >= ratio ||
			   (double)intersectionArea / (double)regions[r2].size() >= ratio)
			{
				if(regions[r2].size() > regions[r1].size())
				{
					maxRegionIdx = r2;
				}
				skipSet.insert(r2);
			}
		}

		validIdx.push_back(maxRegionIdx);
	}

	vector<vector<Point>> validRegions = vector<vector<Point>>();
	for(uint i = 0; i < validIdx.size(); i++)
	{
		validRegions.push_back(regions[validIdx[i]]);
	}

    regions = validRegions;

}

void regionsToLetterCandidates(vector<vector<Point>>&	regions,
							   vector<LetterCandidate>&	letters,
							   Mat						strokeImage)
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

void formWordLines(vector<LetterCandidate>&	letters,
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
