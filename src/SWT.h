/*
 * SWT.hpp
 *
 *  Created on: 11.09.2017
 *      Author: tilman
 */

#ifndef SRC_SWT_H_
#define SRC_SWT_H_

#include <opencv2/opencv.hpp>
#include "LetterCandidate.h"

using namespace cv;
using namespace std;

namespace thai {

class SWT {
public:
	SWT();
	virtual ~SWT();
	void detectStroke(InputArray	_src,
					  OutputArray 	_dst,
					  int 			threshold1,
					  int 			threshold2,
					  int 			maxStrokeWidth,
					  int			direction);
	void detectStrokeInBinary(InputArray	_src,
							  OutputArray	_dst);
	void testStrokeWidthVariance(vector<LetterCandidate>&	letters,
								 double						maxRatio);
};

} /* namespace thai */

#endif /* SRC_SWT_H_ */
