/*
 * LetterLocalizer.h
 *
 *  Created on: 24.10.2017
 *      Author: tilman
 */

#ifndef SRC_LETTERLOCALIZER_H_
#define SRC_LETTERLOCALIZER_H_

#include <opencv2/opencv.hpp>
#include "SWT.h"
#include "util.h"
#include "LetterCandidate.h"

using namespace cv;
using namespace std;

namespace thai {

typedef vector<LetterCandidate> TextLine;

class LetterLocalizer {
public:
	LetterLocalizer();
	virtual ~LetterLocalizer();

	void localizeText(Mat image, vector<TextLine>& textLines);

private:
	typedef pair<LetterCandidate&,LetterCandidate&> LetterPair;

	void getValidRegionIdx(vector<vector<Point>>& regions,
						   vector<vector<Point>>& validRegions);

	void removeApproxDuplicatesRegions(vector<vector<Point>>& regions,
						  	  	  	   double ratio,
									   vector<Rect>& bBoxes);

	void regionsToLetterCandidates(vector<vector<Point>>&	regions,
								   vector<LetterCandidate>&	letters,
								   Mat						strokeImage);

	void formWordLines(vector<LetterCandidate>&	letters,
					   vector<TextLine>&	    textLines);

};

} /* namespace thai */

#endif /* SRC_LETTERLOCALIZER_H_ */
