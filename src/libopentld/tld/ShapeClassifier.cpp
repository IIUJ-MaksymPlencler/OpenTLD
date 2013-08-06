/*
 * ShapeClassifier.h
 *
 *  Created on: May 20, 2013
 *      Author: Maksym Plencler
 */
#include "ShapeClassifier.h"
#include "DetectorCascade.h"
#include "TLDUtil.h"
#include <iostream>

using namespace cv;

namespace tld
{

ShapeClassifier::ShapeClassifier(void)
{
	this->img = NULL;
    enabled = false;
    similarityThreshold = 0.6f;
}

ShapeClassifier::~ShapeClassifier(void)
{
	release();
}

void ShapeClassifier::release()
{
	positivePatches->clear();
}

void ShapeClassifier::nextIteration(const Mat &img)
{
    if(!enabled) return;

    this->img = img;
}

bool ShapeClassifier::filter(int i)
{
    if(!enabled) return true;

    if(calcSimilarity(i) < similarityThreshold) {
		//cout << "rejected" << endl;	
		return false;
	}
	return true;
}

float ShapeClassifier::calcSimilarity(int windowIdx)
{
	NormalizedPatch patch;

    int *bbox = &windows[TLD_WINDOW_SIZE * windowIdx];
    tldExtractNormalizedPatchBB(img, bbox, patch.values);

	return similarityWithModel(&patch);
}

float similarityOf(Mat patch1, Mat patch2)
{
	Mat result;
	absdiff(patch1, patch2, result);

	float sum = 0;
    unsigned char *imgData = (unsigned char *)result.data;

    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 15; j++)
        {
			sum += imgData[j * result.step + i];
        }
    }
	float res = 1 - (sum/(15*15))/255;
	return res;
}

float ShapeClassifier::similarityWithModel(NormalizedPatch *patch)
{
	if(positivePatches->empty())
    {
        return 0;
    }

	float similarity = 0;
	
	Mat patchMat = Mat(15, 15, CV_8U, patch);
	Canny(patchMat, patchMat, 10, 30, 3);
	//int element_shape = CV_SHAPE_RECT;
	//IplConvKernel* element = cvCreateStructuringElementEx(2+1, 2+1, 1, 1, element_shape);
	dilate(patchMat, patchMat, Mat(), Point(-1,-1));
	//imshow( "propozycja", patchMat );

	Mat modelElemMat;
	//cout << "PP: " << positivePatches->size() << endl;
    
	for(size_t i = 0; i < positivePatches->size(); i++)
    {
		modelElemMat = Mat(15, 15, CV_8U, positivePatches->at(i).values);
		Canny(patchMat, patchMat, 10, 30, 3);
		dilate(patchMat, patchMat, Mat(), Point(-1,-1));
		
		similarity = similarityOf(modelElemMat, patchMat);

        if(similarity > similarityThreshold)
        {
			patchMat.release();
			modelElemMat.release();
			return similarity;
        }
    }

	patchMat.release();
	modelElemMat.release();	
    return similarity;
}


void ShapeClassifier::setPositivePatches(std::vector<NormalizedPatch>* pp)
{
	positivePatches = pp;
}

} /* namespace tld */