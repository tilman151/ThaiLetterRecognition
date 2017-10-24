/*
 * util.h
 *
 *  Created on: 19.09.2017
 *      Author: tilman
 */

#ifndef SRC_UTIL_H_
#define SRC_UTIL_H_

#include <opencv2/opencv.hpp>
#include <shared_ptr.hpp>
#include "LetterCandidate.h"

using namespace cv;
using namespace std;

namespace thai {

typedef pair<LetterCandidate&,LetterCandidate&> LetterPair;
typedef vector<LetterCandidate> TextLine;

void getValidRegionIdx(vector<vector<Point>>& regions,
					   vector<vector<Point>>& validRegions);

unsigned long hashCvPoint(const Point& p);

bool compareCvPoints(const Point& a, const Point& b);

bool comparePairs(const LetterPair& a, const LetterPair& b);

bool compareLetterCandidate(const LetterCandidate& a, const LetterCandidate& b);

void removeApproxDuplicatesRegions(vector<vector<Point>>& regions,
					  	  	  	   double ratio,
								   vector<Rect>& bBoxes);

void regionsToLetterCandidates(vector<vector<Point>>&	regions,
							   vector<LetterCandidate>&	letters,
							   Mat						strokeImage);

void formWordLines(vector<LetterCandidate>&	letters,
				   vector<TextLine>&	    textLines);

} /* namespace thai */

#endif /* SRC_UTIL_H_ */
