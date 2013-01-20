#include "Ferns.h"

void Ferns::readParams(cv::FileNode const& file)
{
    ///Classifier Parameters
    valid        = static_cast<float>(file["valid"]);
    ncc_thesame  = static_cast<float>(file["ncc_thesame"]);
    nstructs     = static_cast<int>(file["num_trees"]);
    structSize   = static_cast<int>(file["num_features"]);
    thr_fern     = static_cast<float>(file["thr_fern"]);
    thr_nn       = static_cast<float>(file["thr_nn"]);
    thr_nn_valid = static_cast<float>(file["thr_nn_valid"]);
}

void Ferns::init(std::vector<cv::Size> const& scales)
{
    this->clear();
    generateFeatures(scales);
    
    acum = 0;
    thrN = 0.5 * nstructs;
    
    for (int i=0; i<nstructs; i++) {
        posteriors.push_back(std::vector<float>(std::pow(2.0, nBIT*structSize), 0));
        pCounter.push_back(std::vector<int>(std::pow(2.0, nBIT*structSize), 0));
        nCounter.push_back(std::vector<int>(std::pow(2.0, nBIT*structSize), 0));
    }
}

void Ferns::calcFeatures(cv::Mat const& patch, int const& scale_idx, std::vector<int>& fern)
{
    int leaf;
    for (int t=0; t<nstructs; t++){
        leaf = 0;
        for (int f=0; f<structSize; f++){
            leaf = (leaf << 1) + features[scale_idx][t*nstructs+f](patch);
        }
        fern[t] = leaf;
    }
}

float Ferns::measure_forest(std::vector<int> fern) {
    float votes = 0;
    for (int i = 0; i < nstructs; i++) {
        votes += posteriors[i][fern[i]];
    }
    return votes;
}

void Ferns::trainF(std::vector<std::pair<std::vector<int>,int> > const& ferns, int resample)
{
    thrP = thr_fern*nstructs;
    //for (int j = 0; j < resample; j++) {
    for (int i = 0; i < ferns.size(); i++) {
        if (ferns[i].second==1) {
            if (measure_forest(ferns[i].first) <= thrP) {
                update(ferns[i].first,1,1);
            }
        } else {
            if (measure_forest(ferns[i].first) >= thrN) {
                update(ferns[i].first,0,1);
            }
        }
    }
    //}
}

void Ferns::trainNN(std::vector<cv::Mat> const& nn_examples)
{
    float conf, dummy;
    std::vector<int> y(nn_examples.size(),0);
    y[0] = 1;
    std::vector<int> isin;
    for (int i=0; i<nn_examples.size(); i++) {
        NNConf(nn_examples[i], isin,conf,dummy);
        if (y[i]==1 && conf<=thr_nn) {
            if (isin[1] < 0) {
                pEx = std::vector<cv::Mat>(1, nn_examples[i]);
                continue;
            }
            pEx.push_back(nn_examples[i]);
        }
        if (y[i]==0 && conf>0.5) {
            nEx.push_back(nn_examples[i]);
        }
    }
    acum++;
}

void Ferns::NNConf(cv::Mat const& example, std::vector<int>& isin, float& rsconf, float& csconf){
    isin = std::vector<int>(3, -1);
    if (pEx.empty()) {
        rsconf = 0;
        csconf = 0;
        return;
    }
    if (nEx.empty()) {
        rsconf = 1;
        csconf = 1;
        return;
    }
    cv::Mat ncc(1, 1, CV_32F);
    float nccP, csmaxP, maxP = 0;
    bool anyP = false;
    int maxPidx, validatedPart = std::ceil(pEx.size()*valid);
    float nccN, maxN = 0;
    bool anyN = false;
    for (int i=0; i<pEx.size(); i++){
        cv::matchTemplate(pEx[i],example,ncc,CV_TM_CCORR_NORMED);
        nccP = (reinterpret_cast<float*>(ncc.data)[0]+1)*0.5;
        if (nccP > ncc_thesame)
            anyP = true;
        if(nccP > maxP){
            maxP = nccP;
            maxPidx = i;
            if(i < validatedPart)
                csmaxP = maxP;
        }
    }
    for (int i=0;i<nEx.size();i++){
        cv::matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);
        nccN = (reinterpret_cast<float*>(ncc.data)[0]+1)*0.5;
        if (nccN > ncc_thesame)
            anyN = true;
        if(nccN > maxN)
            maxN = nccN;
    }
    if (anyP) isin[0] = 1;
    isin[1] = maxPidx;
    if (anyN) isin[2] = 1;
    //Measure Relative Similarity
    float dN = 1-maxN;
    float dP = 1-maxP;
    rsconf = static_cast<float>(dN/(dN+dP));
    //Measure Conservative Similarity
    dP = 1-csmaxP;
    csconf = static_cast<float>(dN/(dN+dP));
}


void Ferns::evaluateTh(std::vector<std::pair<std::vector<int>,int> > const& nXT, std::vector<cv::Mat> const& nExT)
{
    float fconf;
    for (int i=0; i<nXT.size(); i++){
        fconf = static_cast<float>(measure_forest(nXT[i].first)/nstructs) ;
        if (fconf > thr_fern)
            thr_fern = fconf;
    }
    std::vector<int> isin;
    float conf, dummy;
    for (int i=0; i<nExT.size(); i++){
        NNConf(nExT[i], isin, conf, dummy);
        if (conf > thr_nn)
            thr_nn = conf;
    }
    if (thr_nn > thr_nn_valid)
        thr_nn_valid = thr_nn;
}

void Ferns::clear()
{
    features.clear();
    posteriors.clear();
    pCounter.clear();
    nCounter.clear();
    pEx.clear();
    nEx.clear();
}

void Ferns::update(std::vector<int> const& fern, int C, int N) {
    int idx;
    for (int i = 0; i < nstructs; i++) {
        idx = fern[i];
        (C==1) ? pCounter[i][idx] += N : nCounter[i][idx] += N;
        if (pCounter[i][idx]==0) {
            posteriors[i][idx] = 0;
        } else {
            posteriors[i][idx] = static_cast<float>((pCounter[i][idx])) / (pCounter[i][idx] + nCounter[i][idx]);
        }
    }
}

void Ferns::generateFeatures(std::vector<cv::Size> const& scales)
{
    int totalFeatures = nstructs*structSize;
    features = std::vector< std::vector<Feature> >(scales.size(), std::vector<Feature> (totalFeatures) );
    
    cv::RNG& rng = cv::theRNG();
    for (int i=0; i<totalFeatures; i++) {
        float x1 = static_cast<float>(rng);
        float y1 = static_cast<float>(rng);
        float x2 = static_cast<float>(rng);
        float y2 = static_cast<float>(rng);
        for (int s=0; s<scales.size(); s++) {
            int scaled_x1 = scales[s].width * x1;
            int scaled_y1 = scales[s].height* y1;
            int scaled_x2 = scales[s].width * x2;
            int scaled_y2 = scales[s].height* y2;
            features[s][i] = Feature(static_cast<uchar>(scaled_x1), static_cast<uchar>(scaled_y1),
                                     static_cast<uchar>(scaled_x2), static_cast<uchar>(scaled_y2));
        }
    }
}