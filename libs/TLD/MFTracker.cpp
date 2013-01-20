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
#include "MFTracker.h"

MFTracker::MFTracker()
:term_criteria( cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 20, 0.03)
,window_size(cv::Size(4,4))
,level(5)
,deriv_lambda(0.5)
{
}

bool MFTracker::track(const cv::Mat &img1, const cv::Mat &img2, Points &points1, Points &points2)
{
    std::vector<uchar> f_status, b_status;
    std::vector<float> f_error,  b_error;
    //Forward-Backward tracking
    cv::calcOpticalFlowPyrLK( img1,img2, points1, points2,  f_status, f_error, window_size, level, term_criteria, deriv_lambda, 0);
    cv::calcOpticalFlowPyrLK( img2,img1, points2, pointsFB, b_status, b_error, window_size, level, term_criteria, deriv_lambda, 0);
    //Compute the forward-backward-error
    fb_error.resize(points1.size());
    for( int i= 0; i<points1.size(); ++i ){
        fb_error[i] = norm(pointsFB[i]-points1[i]);
    }
    //Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
    normCrossCorrelation(img1,img2,points1,points2,f_status);
    return filterPoints(points1,points2, f_status);
}

void MFTracker::normCrossCorrelation(const cv::Mat &img1, const cv::Mat &img2,
                                     Points &points1, Points &points2, std::vector<uchar>& status)
{
    cv::Mat rec0(10,10,CV_8U);
    cv::Mat rec1(10,10,CV_8U);
    cv::Mat res(1,1,CV_32F);
    
    ncc.resize(points1.size());
    for (int i = 0; i < points1.size(); i++) {
        if (status[i] == 1) {
            getRectSubPix( img1, cv::Size(10,10), points1[i],rec0 );
            getRectSubPix( img2, cv::Size(10,10), points2[i],rec1);
            matchTemplate( rec0,rec1, res, CV_TM_CCOEFF_NORMED);
            ncc[i] = ((float *)(res.data))[0];
        } else {
            ncc[i] = 0.0;
        }
    }
    rec0.release();
    rec1.release();
    res.release();

}

bool MFTracker::filterPoints(Points &points1, Points &points2, std::vector<uchar>& status)
{
    if (filterPointsByNcc(points1, points2, status)) {
        return filterPointsByFb(points1, points2, status);
    } else {
        return false;
    }
}

bool MFTracker::filterPointsByFb(Points &points1, Points &points2, std::vector<uchar>& status)
{
    //Get Error Medians
    float median_ncc = median(ncc);
    size_t i, k;
    for( i=k = 0; i<points2.size(); ++i ){
        if( !status[i])
            continue;
        if(ncc[i]> median_ncc){
            points1[k] = points1[i];
            points2[k] = points2[i];
            fb_error[k] = fb_error[i];
            status[k] = status[i];
            k++;
        }
    }
    if (k==0)
        return false;
    points1.resize(k);
    points2.resize(k);
    fb_error.resize(k);
    status.resize(k);

    return true;
}

bool MFTracker::filterPointsByNcc(Points &points1, Points &points2, std::vector<uchar>& status)
{
    median_fb_error = median(fb_error);
    size_t i, k;
    for( i=k = 0; i<points2.size(); ++i ){
        if( !status[i])
            continue;
        if(fb_error[i] <= median_fb_error){
            points1[k] = points1[i];
            points2[k] = points2[i];
            k++;
        }
    }
    points1.resize(k);
    points2.resize(k);
    if (k>0)
        return true;
    else
        return false;
}
