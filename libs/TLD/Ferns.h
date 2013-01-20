#pragma once

#include <opencv2/opencv.hpp>

class Ferns
{
public:
    void readParams(cv::FileNode const& file);
    void init(std::vector<cv::Size> const& scales);
    
    int getNumTrees() const { return nstructs; }
    void calcFeatures(cv::Mat const& patch, int const& scale_idx, std::vector<int>& fern);
    float measure_forest(std::vector<int> fern);
    
    void trainF(std::vector<std::pair<std::vector<int>,int> > const& ferns, int resample);
    void trainNN(std::vector<cv::Mat> const& nn_examples);
    void NNConf(cv::Mat const& example, std::vector<int>& isin, float& rsconf, float& csconf);
    void evaluateTh(std::vector<std::pair<std::vector<int>,int> > const& nXT, std::vector<cv::Mat> const& nExT);

    float getThreshFern() const { return thr_fern; }
    float getThreshNN() const { return thr_nn; }
    float getThreshNNValid() const { return thr_nn_valid; }
 
private:
    void clear();
    void generateFeatures(std::vector<cv::Size> const& scales);
    void update(std::vector<int> const& fern, int C, int N);
    
private:
    //Parameters --------------------------
    static int const nBIT = 1;
    float valid;
    float ncc_thesame;
    int nstructs;
    int structSize;
    float thr_fern;
    float thr_nn;
    float thr_nn_valid;
    
    //Variables ---------------------------
    int acum;
    //Ferns Members
    struct Feature {
        Feature() : x1(0), y1(0), x2(0), y2(0) {}
        Feature(uchar _x1, uchar _y1, uchar _x2, uchar _y2) : x1(_x1), y1(_y1), x2(_x2), y2(_y2) {}
        float x1, y1, x2, y2;
        bool operator()(cv::Mat const& patch) const { return patch.at<uchar>(y1,x1) > patch.at<uchar>(y2,x2); }
    };
    std::vector< std::vector<Feature> > features;
    std::vector< std::vector<float> > posteriors; //Ferns posteriors
    std::vector< std::vector<int> > pCounter; //positive counter
    std::vector< std::vector<int> > nCounter; //negative counter
    float thrP; //Positive thershold
    float thrN; //Negative threshold

    //NN Members
    std::vector<cv::Mat> pEx; //NN positive examples
    std::vector<cv::Mat> nEx; //NN negative examples
};
