/*
 * Median Flow Tracker
 *   1. Track points between frame.
 *      Points are tracked using Lucas-Kanade Tracker with pyramids.
 *   2. Estimate reliability of the points.
 *      To get reliable points, Forward-Backward error method is used.
 *      In FB method, points are tracked twice. (previous image -> current image, current image -> previous image)
 *   3. Filter out 50% of the outliers.
 *      50% of the points are filtered out using median filter.
 *      First the median is calculated for the vector of points, and almost reliable points are chosen.
 */
#pragma once

#include <opencv2/opencv.hpp>
#include "Utils.h"

typedef std::vector<cv::Point2f> Points;
class MFTracker
{
public:
    MFTracker();
    bool track(cv::Mat const& img1, cv::Mat const& img2, Points& points1, Points& points2);
    float getMedianFbError() const { return median_fb_error; }
    
private:
    void normCrossCorrelation(cv::Mat const& img1, cv::Mat const& img2,
                              Points& points1, Points& points2, std::vector<uchar>& status);
    bool filterPoints(Points& points1, Points& points2, std::vector<uchar>& status);
    bool filterPointsByFb(Points& points1, Points& points2, std::vector<uchar>& status);
    bool filterPointsByNcc(Points& points1, Points& points2, std::vector<uchar>& status);
    
private:
    cv::TermCriteria term_criteria;
    cv::Size window_size;
    int level;
    float deriv_lambda;

    std::vector<cv::Point2f> pointsFB;
    std::vector<float> ncc;
    std::vector<float> fb_error;
    
    float median_fb_error;
};
