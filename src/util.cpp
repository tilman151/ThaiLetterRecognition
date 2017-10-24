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

namespace thai {


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

} /* namespace thai */
