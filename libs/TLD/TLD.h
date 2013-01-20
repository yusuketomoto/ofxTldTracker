#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "MFTracker.h"
#include "Ferns.h"
#include "Utils.h"

struct BoundingBox : public cv::Rect
{
    BoundingBox(){}
    BoundingBox(cv::Rect& r) : cv::Rect(r) {}
public:
    float overlap;
    int scale_index;
};

struct PosExampleParams
{
    int num_closest;
    int num_warps;
    int noise;
    float angle;
    float shift;
    float scale;
};

struct NegExampleParams
{
    float overlap;
    int num_patches;
};

struct TemporalStructure
{
    std::vector< std::vector<int> > patt;
    std::vector<float> conf;
};
struct DetectionStructure
{
    std::vector<int> bb;
    std::vector<std::vector<int> > patt;
    std::vector<float> conf1;
    std::vector<float> conf2;
    std::vector<std::vector<int> > isin;
    std::vector<cv::Mat> patch;
};

struct OverlapCompare : public std::binary_function<int, int, bool>
{
    OverlapCompare(std::vector<BoundingBox> const& _grid) : grid(_grid) {}
    std::vector<BoundingBox> grid;
    bool operator()(int idx1, int idx2) { return grid[idx1].overlap > grid[idx2].overlap; }
};
struct ConfidenceCompare : public std::binary_function<int, int, bool>
{
    ConfidenceCompare(std::vector<float> const& _conf) : conf(_conf) {}
    std::vector<float> conf;
    bool operator()(int idx1, int idx2) { return conf[idx1] > conf[idx2]; }
};


class TLD
{
public:
    TLD();
    TLD(cv::FileNode const& file);
    void readParams(cv::FileNode const& file);
    void init(cv::Mat const& frame, cv::Rect const& box);
    void clear();
    void processFrame(cv::Mat const& img1, cv::Mat const& img2, Points& points1, Points& points2,
                      BoundingBox& next_box, bool& last_box_found, bool is_enable_train_and_learn);
    float bbOverlap(cv::Rect const& bb1, cv::Rect const& bb2) const;
  
private:
    bool tracking(cv::Mat const& img1, cv::Mat const& img2, Points& points1, Points& points2);
    bool detection(cv::Mat const& img);
    void learning(cv::Mat const& img);
    
    void allocate(cv::Mat const& frame);
    void generatePositiveData(cv::Mat const& frame, PosExampleParams const& p_par);
    void generateNegativeData(cv::Mat const& frame);
    void buildDataSet(std::vector<std::pair<std::vector<int>,int> >& ferns_data, std::vector<cv::Mat>& nn_data);
    void initVarianceThreshold(cv::Mat const& frame);
    
    void overlappingBoxes(PosExampleParams const& p_par);
    void getPattern(cv::Mat const& frame, cv::Mat& pattern, cv::Scalar mean, cv::Scalar std_dev);
    
    void bbScan(cv::Mat const& img, cv::Rect const& bb);
    void bbHull();
    
    float bbVarOffset(BoundingBox const& box, cv::Mat const& iisum, cv::Mat const& iisqsum) const;
    void bbPoints(std::vector<cv::Point2f>& points, BoundingBox const& bb);
    void bbPredict(std::vector<cv::Point2f> const& points1, std::vector<cv::Point2f> const& points2,
                   BoundingBox const& bb1,BoundingBox& bb2);

    void clusterConf(std::vector<BoundingBox> const& dbb, std::vector<float> const& dconf,
                     std::vector<BoundingBox>      & cbb, std::vector<float>      & cconf);
    int clusterBB(const std::vector<BoundingBox>& dbb,std::vector<int>& indexes);
    
private:
    ///Parameters
    int min_win;
    int patch_size;
    int num_trees;
    int bbox_step;
    //initial parameters for positive examples
    PosExampleParams p_par_init;
    PosExampleParams p_par_update;
    //parameters for negative examples
    NegExampleParams n_par;
    
    // Method classes
    cv::PatchGenerator generator;
    Ferns classifier;
    MFTracker tracker;

    
    ///Variables
    // Integral Images
    cv::Mat iisum;
    cv::Mat iisqsum;
    float bbox_var;
    // Training Data
    std::vector<std::pair<std::vector<int>,int> > pX; //positive ferns <features,labels=1>
    std::vector<std::pair<std::vector<int>,int> > nX; //negative ferns <features,labels=0>
    cv::Mat pEx; // positive NN example
    std::vector<cv::Mat> nEx; // negative NN examples
    //Test data
    std::vector<std::pair<std::vector<int>,int> > nXT; //negative data to Test
    std::vector<cv::Mat> nExT; //negative NN examples to Test
    //Last frame data
    BoundingBox lastbox;
    bool lastvalid;
    float lastconf;
    //Current frame data
    //Tracker return values
    BoundingBox tBB;
    float tConf;
    bool tValid;
    //Detector Data
    TemporalStructure tmp;
    DetectionStructure dt;
    //Detector return values
    std::vector<BoundingBox> dBB;
    std::vector<float> dConf;

    //Bounding Boxes
    std::vector<BoundingBox> grid;
    std::vector<cv::Size> scales;
    // Close Bounding Box
    BoundingBox bbP0; // the closest box
    std::vector<int> idxP; // index of close boxes
    std::vector<int> idxN; // index of not close boxes
    BoundingBox bbhull; // hull of close boxes

};