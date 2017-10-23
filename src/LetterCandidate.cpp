/*
 * LetterCandidate.cpp
 *
 *  Created on: 30.09.2017
 *      Author: tilman
 */

#include "LetterCandidate.h"

namespace std {
  template <> struct hash<LetterCandidate>
  {
    size_t operator()(const LetterCandidate & x) const
    {
      return hash<int>()(x.getIndex());
    }
  };
}

LetterCandidate::LetterCandidate() {
	this->index = -1;
	this->mainBoundary = Rect();
	this->bBox = Rect();
	this->meanStrokeWidth = -1;
	this->sDevStrokeWidth = -1;
}

LetterCandidate::LetterCandidate(int index, Rect bBox)
{
	this->index = index;
	this->mainBoundary = bBox;
	this->bBox = bBox;
	this->meanStrokeWidth = -1;
	this->sDevStrokeWidth = -1;
}

LetterCandidate::~LetterCandidate() {
}

int LetterCandidate::getIndex() const
{
	return this->index;
}

Rect LetterCandidate::getBBox()
{
	return this->bBox;
}

Rect LetterCandidate::getMainBoundary()
{
	return this->mainBoundary;
}

bool LetterCandidate::operator==(LetterCandidate& candidate)
{
	return this->index == candidate.getIndex();
}

bool LetterCandidate::operator<(LetterCandidate& candidate)
{
	return this->bBox.x < candidate.getBBox().x;
}

void LetterCandidate::add(LetterCandidate& candidate)
{
	this->bBox = this->bBox | candidate.getBBox();
	this->members.insert(candidate.index);
}

int LetterCandidate::overlaps(LetterCandidate& candidate)
{
	double interArea = (this->bBox &
						candidate.getMainBoundary()).area();

	// No overlap
	if(interArea == 0)
	{
		return 0;
	}

	const Point thisCenter = (this->mainBoundary.br() +
					   	   	  this->mainBoundary.tl())*0.5;
	const Point otherCenter = (candidate.getMainBoundary().br() +
							   candidate.getMainBoundary().tl())*0.5;
	const Point direction = otherCenter - thisCenter;
	double angle = (double)direction.y / (double)direction.x;

	// On top or bottom
	if(angle < -1.5 || angle > 1.5)
	{
		// Top
		if(direction.y < 0)
		{
			return 1;
		}
		// Bottom
		else
		{
			return 3;
		}
	}
	// Right or left
	if(angle > -1 && angle < 1)
	{
		// Right
		if(direction.x > 0)
		{
			return 2;
		}
		// Left
		else
		{
			return 4;
		}
	}


	// Diagonal
	return 5;
}

bool LetterCandidate::sameLineAs(LetterCandidate& candidate)
{
	double heightRatio = (double)this->mainBoundary.height /
						 (double)candidate.getMainBoundary().height;
	// Not the same height
	if(heightRatio > 2 || heightRatio < 0.5)
	{
		return false;
	}

	double swRatio = this->meanStrokeWidth / candidate.meanStrokeWidth;
	// Not the same stroke width
	if(swRatio > 1.5 || swRatio < 0.66)
	{
		return false;
	}

	// Probably same line
	return true;
}
