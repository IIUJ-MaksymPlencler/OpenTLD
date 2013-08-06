/*
 * ShapeClassifier.h
 *
 *  Created on: May 20, 2013
 *      Author: Maksym Plencler
 */

#ifndef SHAPECLASSIFIER_H_
#define SHAPECLASSIFIER_H_

#include <opencv/cv.h>

#include "DetectionResult.h"
#include "NormalizedPatch.h"

namespace tld
{

class ShapeClassifier
{
	std::vector<NormalizedPatch>* positivePatches;
    cv::Mat img;
	

public:
    bool enabled;
	int *windows;
    int *windowOffsets;

    DetectionResult *detectionResult;

    float similarityThreshold;

    ShapeClassifier();
    virtual ~ShapeClassifier();

    void release();
    void nextIteration(const cv::Mat &img);
    bool filter(int idx);
    float calcSimilarity(int windowIdx);
	void setPositivePatches(std::vector<NormalizedPatch>* positivePatches);
	float similarityWithModel(NormalizedPatch *patch);
};

} /* namespace tld */
#endif /* SHAPECLASSIFIER_H_ */
