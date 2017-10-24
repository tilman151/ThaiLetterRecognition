/*
 * LetterCandidate.h
 *
 *  Created on: 30.09.2017
 *      Author: tilman
 */

#ifndef SRC_LETTERCANDIDATE_H_
#define SRC_LETTERCANDIDATE_H_

#include <opencv2/opencv.hpp>
#include <unordered_set>

using namespace cv;
using namespace std;

namespace thai {

class LetterCandidate {

public:
	double meanStrokeWidth;
	double sDevStrokeWidth;

	LetterCandidate();
	LetterCandidate(int index, Rect bBox);
	virtual ~LetterCandidate();

	int getIndex() const;
	Rect getBBox();
	Rect getMainBoundary();

	bool operator==(LetterCandidate& candidate);
	bool operator<(LetterCandidate& candidate);

	void add(LetterCandidate& candidate);
	int overlaps(LetterCandidate& candidate);
	bool sameLineAs(LetterCandidate& candidate);

private:
	int index;
	Rect mainBoundary;
	Rect bBox;
	unordered_set<int> members;

};

} /* namespace thai */

#endif /* SRC_LETTERCANDIDATE_H_ */
