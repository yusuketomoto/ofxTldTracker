#include "TLD.h"

TLD::TLD()
{
}

TLD::TLD(cv::FileNode const& file)
{
    readParams(file);
}

void TLD::readParams(cv::FileNode const& file)
{
    ///Bounding Box Parameters
    min_win            = static_cast<int>(file["min_win"]);
    
    ///Genarator Parameters
    patch_size         = static_cast<int>(file["patch_size"]);
    //initial parameters for positive examples
    p_par_init.num_closest = static_cast<int>(file["num_closest_init"]);
    p_par_init.num_warps   = static_cast<int>(file["num_warps_init"]);
    p_par_init.noise       = static_cast<int>(file["noise_init"]);
    p_par_init.angle       = static_cast<float>(file["angle_init"]);
    p_par_init.shift       = static_cast<float>(file["shift_init"]);
    p_par_init.scale       = static_cast<float>(file["scale_init"]);
    //update parameters for positive examples
    p_par_update.num_closest = static_cast<int>(file["num_closest_update"]);
    p_par_update.num_warps   = static_cast<int>(file["num_warps_update"]);
    p_par_update.noise       = static_cast<int>(file["noise_update"]);
    p_par_update.angle       = static_cast<float>(file["angle_update"]);
    p_par_update.shift       = static_cast<float>(file["shift_update"]);
    p_par_update.scale       = static_cast<float>(file["scale_update"]);
    //parameters for negative examples
    n_par.overlap     = static_cast<float>(file["overlap"]);
    n_par.num_patches = static_cast<int>(file["num_patches"]);
    
    num_trees = static_cast<int>(file["num_trees"]);

    classifier.readParams(file);
}

void TLD::init(cv::Mat const& frame, cv::Rect const& box)
{
    int count=0;
    // INITIALIZE DETECTOR ------------------------------
    //Scaning Grid
    bbScan(frame, box);
    allocate(frame);
    bbox_step =7;

    classifier.init(scales);

    // TRAIN DETECTOR -----------------------------------
    generator = cv::PatchGenerator (0, 0, p_par_init.noise, true, 1-p_par_init.scale, 1+p_par_init.scale,
                                    -p_par_init.angle*CV_PI/180, p_par_init.angle*CV_PI/180,
                                    -p_par_init.angle*CV_PI/180, p_par_init.angle*CV_PI/180);
    
    overlappingBoxes(p_par_init);
    bbHull();
    lastbox = bbP0;
    lastconf = 1;
    lastvalid = true;
    
    // Generate Positive Examples
    generatePositiveData(frame, p_par_init);
    // Variance threshold
    initVarianceThreshold(frame);
    // Generate Negative Examples
    generateNegativeData(frame);
    
    // Generate Apriori Negative Examples
    std::vector<std::pair<std::vector<int>,int> > ferns_data;
    std::vector<cv::Mat> nn_data;
    buildDataSet(ferns_data, nn_data);
    
    ///Training
    classifier.trainF(ferns_data,2); //bootstrap = 2
    classifier.trainNN(nn_data);
    ///Threshold Evaluation on testing sets
    classifier.evaluateTh(nXT,nExT);
}

void TLD::clear()
{
    dBB.clear();
    dConf.clear();
    grid.clear();
    scales.clear();
    idxP.clear();
    idxN.clear();
    pX.clear();
    nX.clear();
    nEx.clear();
}

void TLD::processFrame(cv::Mat const& img1, cv::Mat const& img2, Points &points1, Points &points2,
                       BoundingBox &next_box, bool &last_box_found, bool is_enable_train_and_learn)
{
    std::vector<BoundingBox> cBB;
    std::vector<float> cConf;
    int confident_detections = 0;
    int detection_index; //detection index

    // TRACKER ---------------------------
    bool isTracked = false;
    if (last_box_found && is_enable_train_and_learn)
        isTracked = tracking(img1, img2, points1, points2);
    
    // DETECTOR --------------------------
    bool isDetected = detection(img2);
    
    // INTEGRATOR ------------------------
    if (isTracked) { // if tracker is defined
        next_box = tBB;
        lastconf = tConf;
        lastvalid = tValid;

        if (isDetected) { // if detections are also defined
            // cluster detections
            clusterConf(dBB, dConf, cBB, cConf);
            for (int i=0; i<cBB.size(); i++){
                if (bbOverlap(tBB,cBB[i])<0.5 && cConf[i]>tConf){
                    // Get index of a clusters that is far from tracker and are more confident than the tracker
                    confident_detections++;
                    detection_index = i;
                }
            }
            if (confident_detections == 1){
                //if there is ONE such a cluster, re-initialize the tracker
                next_box = cBB[detection_index];
                lastconf = cConf[detection_index];
                lastvalid = false;
            }
            else {
                int cx=0, cy=0, cw=0, ch=0;
                int close_detections = 0;
                for (int i=0; i<dBB.size(); i++){
                    if (bbOverlap(tBB,dBB[i]) > 0.7){
                        // Get mean of close detections
                        cx += dBB[i].x;
                        cy += dBB[i].y;
                        cw += dBB[i].width;
                        ch += dBB[i].height;
                        close_detections++;
                    }
                }
                if (close_detections>0){
                    // weighted average trackers trajectory with the close detections
                    next_box.x = cvRound(static_cast<float>(10*tBB.x+cx)/static_cast<float>(10+close_detections));
                    next_box.y = cvRound(static_cast<float>(10*tBB.y+cy)/static_cast<float>(10+close_detections));
                    next_box.width  = cvRound(static_cast<float>(10*tBB.width +cw)/static_cast<float>(10+close_detections));
                    next_box.height = cvRound(static_cast<float>(10*tBB.height+ch)/static_cast<float>(10+close_detections));
                }
            }
            
        }
    } else { // if tracker is not defined
        last_box_found = false;
        lastvalid = false;
        if (isDetected) { // and detector is defined
            // cluster detections
            clusterConf(dBB, dConf, cBB, cConf);
            if (cConf.size()==1) {
                next_box = cBB[0];
                lastconf = cConf[0];
                last_box_found = true;
            }
        }
    }
    lastbox = next_box;
    
    // LEARNING --------------------------
    if (lastvalid && is_enable_train_and_learn)
        learning(img2);
}

bool TLD::tracking(cv::Mat const& img1, cv::Mat const& img2, Points &points1, Points &points2)
{
    bool tracked;
    //Generate points
    bbPoints(points1,lastbox);
    if (points1.size()<1){
        tValid = false;
        tracked = false;
        return tracked;
    }
    std::vector<cv::Point2f> points = points1;
    //Frame-to-frame tracking with forward-backward error cheking
    tracked = tracker.track(img1,img2,points,points2);
    if (tracked){
        //Bounding box prediction
        bbPredict(points,points2,lastbox,tBB);
        if (tracker.getMedianFbError()>10 || tBB.x>img2.cols ||  tBB.y>img2.rows || tBB.br().x < 1 || tBB.br().y <1){
            tValid =false; //too unstable prediction or bounding box out of image
            tracked = false;
            return tracked;
        }
        //Estimate Confidence and Validity
        cv::Mat pattern;
        cv::Scalar mean, stdev;
        BoundingBox bb;
        bb.x = std::max(tBB.x,0);
        bb.y = std::max(tBB.y,0);
        bb.width  = std::min(std::min(img2.cols-tBB.x,tBB.width), std::min(tBB.width, tBB.br().x));
        bb.height = std::min(std::min(img2.rows-tBB.y,tBB.height),std::min(tBB.height,tBB.br().y));
        getPattern(img2(bb),pattern,mean,stdev);
        std::vector<int> isin;
        float dummy;
        classifier.NNConf(pattern,isin,dummy,tConf); //Conservative Similarity
        tValid = lastvalid;
        if (tConf > classifier.getThreshNNValid()){
            tValid = true;
        }
    }
    return tracked;
}

bool TLD::detection(cv::Mat const& frame)
{
    bool detected;
    //cleaning
    dBB.clear();
    dConf.clear();
    dt.bb.clear();
    double t = static_cast<double>(cv::getTickCount());
    cv::Mat img(frame.rows, frame.cols, CV_8U);
    cv::integral(frame, iisum, iisqsum);
    cv::GaussianBlur(frame, img, cv::Size(9,9), 1.5);
    int numtrees = classifier.getNumTrees();
    float fern_th = classifier.getThreshFern();
    std::vector <int> ferns(10);
    float conf;
    int a=0;
    cv::Mat patch;
    for (int i=0; i<grid.size(); i++) {
        if (bbVarOffset(grid[i],iisum,iisqsum)>=bbox_var){
            a++;
            patch = img(grid[i]);
            classifier.calcFeatures(patch,grid[i].scale_index,ferns);
            conf = classifier.measure_forest(ferns);
            tmp.conf[i]=conf;
            tmp.patt[i]=ferns;
            if (conf>numtrees*fern_th){
                dt.bb.push_back(i);
            }
        }
        else
            tmp.conf[i]=0.0;
    }
    int detections = dt.bb.size();
    if (detections > 100) {
        nth_element(dt.bb.begin(),dt.bb.begin()+100,dt.bb.end(),ConfidenceCompare(tmp.conf));
        dt.bb.resize(100);
        detections = 100;
    }
    if (detections == 0) {
        detected=false;
        return detected;
    }
    t = static_cast<double>(cv::getTickCount()-t);
    //  Initialize detection structure
    dt.patt = std::vector<std::vector<int> >(detections,std::vector<int>(10,0));
    dt.conf1 = std::vector<float>(detections);
    dt.conf2 =std::vector<float>(detections);
    dt.isin = std::vector<std::vector<int> >(detections,std::vector<int>(3,-1));
    dt.patch = std::vector<cv::Mat>(detections,cv::Mat(patch_size,patch_size,CV_32F));
    int idx;
    cv::Scalar mean, stdev;
    float nn_th = classifier.getThreshNN();
    for (int i=0; i<detections; i++) {
        idx = dt.bb[i];
        patch = frame(grid[idx]);
        getPattern(patch,dt.patch[i],mean,stdev);
        classifier.NNConf(dt.patch[i],dt.isin[i],dt.conf1[i],dt.conf2[i]);
        dt.patt[i]=tmp.patt[idx];
        if (dt.conf1[i] > nn_th) {
            dBB.push_back(grid[idx]);
            dConf.push_back(dt.conf2[i]);
        }
    }
    if (dBB.size()>0){
        detected=true;
    }
    else{
        detected=false;
    }
    return detected;
}

void TLD::learning(cv::Mat const& img)
{
    ///Check consistency
    BoundingBox bb;
    bb.x = std::max(lastbox.x,0);
    bb.y = std::max(lastbox.y,0);
    bb.width = std::min(std::min(img.cols-lastbox.x,lastbox.width),std::min(lastbox.width,lastbox.br().x));
    bb.height = std::min(std::min(img.rows-lastbox.y,lastbox.height),std::min(lastbox.height,lastbox.br().y));
    cv::Scalar mean, stdev;
    cv::Mat pattern;
    getPattern(img(bb), pattern, mean, stdev);
    std::vector<int> isin;
    float dummy, conf;
    classifier.NNConf(pattern, isin, conf, dummy);
    if (conf < 0.5) {
        lastvalid = false;
        return;
    }
    if (pow(stdev.val[0],2) < bbox_var) {
        lastvalid = false;
        return;
    }
    if(isin[2] == 1) {
        lastvalid = false;
        return;
    }
    /// Data generation
    for (int i=0; i<grid.size(); i++) {
        grid[i].overlap = bbOverlap(lastbox, grid[i]);
    }
    std::vector<std::pair<std::vector<int>,int> > fern_examples;
    idxP.clear();
    idxN.clear();
    overlappingBoxes(p_par_update);
    if (idxP.size() > 0) {
        generatePositiveData(img,p_par_update);
    }
    else {
        lastvalid = false;
        return;
    }
    fern_examples.reserve(pX.size()+idxN.size());
    fern_examples.assign(pX.begin(),pX.end());
    int idx;
    for (int i=0; i<idxN.size(); i++){
        idx = idxN[i];
        if (tmp.conf[idx] >= 1){
            fern_examples.push_back(make_pair(tmp.patt[idx],0));
        }
    }
    std::vector<cv::Mat> nn_examples;
    nn_examples.reserve(dt.bb.size()+1);
    nn_examples.push_back(pEx);
    for (int i=0; i<dt.bb.size(); i++){
        idx = dt.bb[i];
        if (bbOverlap(lastbox,grid[idx]) < n_par.overlap)
            nn_examples.push_back(dt.patch[i]);
    }
    /// Classifiers update
    classifier.trainF(fern_examples,2);
    classifier.trainNN(nn_examples);
//    classifier.show();
}

void TLD::allocate(cv::Mat const& frame)
{
    iisum.create(frame.rows+1, frame.cols+1, CV_32F);
    iisqsum.create(frame.rows+1, frame.cols+1, CV_64F);
    dConf.reserve(100);
    dBB.reserve(100);
    tmp.conf = std::vector<float>(grid.size());
    tmp.patt = std::vector<std::vector<int> >(grid.size(),std::vector<int>(num_trees,0));
    dt.bb.reserve(grid.size());
    idxP.reserve(grid.size());
    idxN.reserve(grid.size());
    pEx.create(patch_size,patch_size,CV_64F);
}

void TLD::generatePositiveData(cv::Mat const& frame, PosExampleParams const& p_par)
{
    cv::Scalar mean;
    cv::Scalar stdev;
    getPattern(frame(bbP0), pEx, mean, stdev);
    //Get Fern features on warped patches
    cv::Mat img_blur;
    cv::Mat warped;
    cv::GaussianBlur(frame, img_blur, cv::Size(9,9), 1.5);
    warped = img_blur(bbhull);
    cv::RNG& rng = cv::theRNG();
    cv::Point2f pt(bbhull.x+(bbhull.width-1)*0.5f, bbhull.y+(bbhull.height-1)*0.5f);
    std::vector<int> fern(classifier.getNumTrees());
    pX.clear();
    cv::Mat patch;
    if (pX.capacity() < p_par.num_warps*idxP.size()) {
        pX.reserve(p_par.num_warps*idxP.size());
    }
    int idx;
    for (int i=0; i<p_par.num_warps; i++) {
        if (i > 0) {
            generator(frame,pt,warped,bbhull.size(),rng);
        }
        for (int b=0; b<idxP.size(); b++) {
            idx = idxP[b];
            patch = img_blur(grid[idx]);
            // measures on blured image
            classifier.calcFeatures(patch, grid[idx].scale_index, fern);
            pX.push_back(std::make_pair(fern,1));
        }
    }
}

void TLD::generateNegativeData(cv::Mat const& frame)
{
    std::random_shuffle(idxN.begin(), idxN.end());
    int idx;
    int a=0;
    std::vector<int> fern(classifier.getNumTrees());
    nX.reserve(idxN.size());
    cv::Mat patch;
    for (int j=0; j<idxN.size(); j++){
        idx = idxN[j];
        if (bbVarOffset(grid[idx], iisum, iisqsum) < bbox_var*0.5f)
            continue;
        patch =  frame(grid[idx]);
        classifier.calcFeatures(patch, grid[idx].scale_index, fern);
        nX.push_back(make_pair(fern, 0));
        a++;
    }
    cv::Scalar dum1, dum2;
    nEx = std::vector<cv::Mat>(n_par.num_patches);
    for (int i=0; i<n_par.num_patches; i++){
        idx = idxN[i];
        patch = frame(grid[idx]);
        getPattern(patch, nEx[i], dum1, dum2);
    }
}

void TLD::buildDataSet(std::vector<std::pair<std::vector<int>,int> >& ferns_data, std::vector<cv::Mat>& nn_data)
{
    //Split Negative Ferns into Training and Testing sets (they are already shuffled)
    int half = static_cast<int>(nX.size()*0.5f);
    nXT.assign(nX.begin()+half, nX.end());
    nX.resize(half);
    ///Split Negative NN Examples into Training and Testing sets
    half = static_cast<int>(nEx.size()*0.5f);
    nExT.assign(nEx.begin()+half, nEx.end());
    nEx.resize(half);
    //Merge Negative Data with Positive Data and shuffle it
    ferns_data.resize(nX.size()+pX.size());
    std::vector<int> idx = indexShuffle(0, ferns_data.size());
    int a = 0;
    for (int i=0; i<pX.size(); i++) {
        ferns_data[idx[a]] = pX[i];
        a++;
    }
    for (int i=0; i<nX.size(); i++) {
        ferns_data[idx[a]] = nX[i];
        a++;
    }
    //Data already have been shuffled, just putting it in the same vector
    nn_data.resize(nEx.size()+1);
    nn_data[0] = pEx;
    for (int i=0; i<nEx.size(); i++) {
        nn_data[i+1] = nEx[i];
    }

}

void TLD::initVarianceThreshold(cv::Mat const& frame)
{
    cv::Scalar stdev, mean;
    cv::meanStdDev(frame(bbP0),mean,stdev);
    cv::integral(frame,iisum,iisqsum);
    bbox_var = pow(stdev.val[0],2)*0.5;
//    double vr =  bbVarOffset(bbP0,iisum,iisqsum)*0.5;
//    assert(bbox_var == vr);
}

void TLD::overlappingBoxes(PosExampleParams const& p_par)
{
    float max_overlap = 0;
    for (int i=0;i<grid.size();i++){
        if (grid[i].overlap > max_overlap) {
            max_overlap = grid[i].overlap;
            bbP0 = grid[i];
        }
        if (grid[i].overlap > 0.6){
            idxP.push_back(i);
        }
        else if (grid[i].overlap < n_par.overlap){
            idxN.push_back(i);
        }
    }
    if (idxP.size()>p_par.num_closest){
        std::nth_element(idxP.begin(),idxP.begin()+p_par.num_closest,idxP.end(),OverlapCompare(grid));
        idxP.resize(p_par.num_closest);
    }
}

void TLD::getPattern(const cv::Mat &frame, cv::Mat &pattern, cv::Scalar mean, cv::Scalar std_dev)
{
    //Output: resized Zero-Mean patch
    resize(frame,pattern,cv::Size(patch_size,patch_size));
    cv::meanStdDev(pattern,mean,std_dev);
    pattern.convertTo(pattern,CV_32F);
    pattern = pattern-mean.val[0];
}

// make bounding boxes by multi-scale grid sampling
void TLD::bbScan(cv::Mat const& img, cv::Rect const& bb)
{
    const float SHIFT = 0.1;
    const float SCALES[] = {0.16151,0.19381,0.23257,0.27908,0.33490,0.40188,0.48225, // 1.2^[-10:10]
                            0.57870,0.69444,0.83333,      1,1.20000,1.44000,1.72800,
                            2.07360,2.48832,2.98598,3.58318,4.29982,5.15978,6.19174};
    
    BoundingBox bbox;
    int index_count = 0;
    for (int s=0; s<21; s++) {
        int bbw = round(bb.width *SCALES[s]);
        int bbh = round(bb.height*SCALES[s]);
        int min_wh = std::min(bbw, bbh);
        int grid_step = round(SHIFT * min_wh);
        if (min_wh < min_win || bbw > img.cols || bbh > img.rows) {
            continue;
        }
        cv::Size scale(bbw,bbh);
        scales.push_back(scale);
        for (int y=1; y<img.rows-bbh; y+=grid_step) {
            for (int x=1; x<img.cols-bbw; x+=grid_step) {
                bbox.x = x;
                bbox.y = y;
                bbox.width  = bbw;
                bbox.height = bbh;
                bbox.overlap = bbOverlap(bb, bbox);
                bbox.scale_index = index_count;
                grid.push_back(bbox);
            }
        }
        index_count++;
    }
}

float TLD::bbOverlap(cv::Rect const& bb1, cv::Rect const& bb2) const
{
	if (bb1.x > bb2.x+bb2.width)  { return 0.0; }
	if (bb1.y > bb2.y+bb2.height) { return 0.0; }
	if (bb1.x+bb1.width  < bb2.x) { return 0.0; }
	if (bb1.y+bb1.height < bb2.y) { return 0.0; }
	
	float colInt =  std::min(bb1.x+bb1.width,  bb2.x+bb2.width)  - std::max(bb1.x, bb2.x);
	float rowInt =  std::min(bb1.y+bb1.height, bb2.y+bb2.height) - std::max(bb1.y, bb2.y);
    
	float intersection = colInt * rowInt;
	float area1 = bb1.width*bb1.height;
	float area2 = bb2.width*bb2.height;
    
	return intersection / (area1 + area2 - intersection);
}

void TLD::bbHull()
{
    int x1 = INT_MAX, x2 = 0;
    int y1 = INT_MAX, y2 = 0;
    int idx;
    for (int i=0;i<idxP.size();i++){
        idx= idxP[i];
        x1 = std::min(grid[idx].x, x1);
        y1 = std::min(grid[idx].y, y1);
        x2 = std::max(grid[idx].x+grid[idx].width, x2);
        y2 = std::max(grid[idx].y+grid[idx].height, y2);
    }
    bbhull.x = x1;
    bbhull.y = y1;
    bbhull.width = x2-x1;
    bbhull.height = y2-y1;
}

float TLD::bbVarOffset(BoundingBox const& box, cv::Mat const& iisum, cv::Mat const& iisqsum) const
{
    double brs = iisum.at<int>(box.y+box.height, box.x+box.width);
    double bls = iisum.at<int>(box.y+box.height, box.x);
    double trs = iisum.at<int>(box.y, box.x+box.width);
    double tls = iisum.at<int>(box.y, box.x);
    double brsq = iisqsum.at<double>(box.y+box.height, box.x+box.width);
    double blsq = iisqsum.at<double>(box.y+box.height, box.x);
    double trsq = iisqsum.at<double>(box.y, box.x+box.width);
    double tlsq = iisqsum.at<double>(box.y, box.x);
    double mean   = (brs +tls -trs -bls)  / static_cast<double>(box.area());
    double sqmean = (brsq+tlsq-trsq-blsq) / static_cast<double>(box.area());
    
    return sqmean-mean*mean;
}

void TLD::bbPoints(std::vector<cv::Point2f> &points, BoundingBox const& bb)
{
    int max_pts = 10;
    int margin_h = 0;
    int margin_v = 0;
    int stepx = ceil((bb.width-2*margin_h) / max_pts);
    int stepy = ceil((bb.height-2*margin_v) / max_pts);
    for (int y=bb.y+margin_v; y<bb.y+bb.height-margin_v; y+=stepy) {
        for (int x=bb.x+margin_h; x<bb.x+bb.width-margin_h; x+=stepx) {
            points.push_back(cv::Point2f(x,y));
        }
    }
}

void TLD::bbPredict(std::vector<cv::Point2f> const& points1, std::vector<cv::Point2f> const& points2,
                    BoundingBox const& bb1, BoundingBox& bb2)
{
    int npoints = (int)points1.size();
    std::vector<float> xoff(npoints);
    std::vector<float> yoff(npoints);
    for (int i=0; i<npoints; i++) {
        xoff[i]=points2[i].x-points1[i].x;
        yoff[i]=points2[i].y-points1[i].y;
    }
    float dx = median(xoff);
    float dy = median(yoff);
    float s;
    if (npoints > 1){
        std::vector<float> d;
        d.reserve(npoints*(npoints-1)/2);
        for (int i=0; i<npoints; i++) {
            for (int j=i+1; j<npoints; j++){
                d.push_back(cv::norm(points2[i]-points2[j]) / cv::norm(points1[i]-points1[j]));
            }
        }
        s = median(d);
    }
    else {
        s = 1.0;
    }
    float s1 = 0.5*(s-1)*bb1.width;
    float s2 = 0.5*(s-1)*bb1.height;
    bb2.x = round( bb1.x + dx -s1);
    bb2.y = round( bb1.y + dy -s2);
    bb2.width = round(bb1.width*s);
    bb2.height = round(bb1.height*s);
}

bool bbComp(BoundingBox const& b1, BoundingBox const& b2)
{
    TLD t;
    if (t.bbOverlap(b1,b2)<0.5)
        return false;
    else
        return true;
}

void TLD::clusterConf(std::vector<BoundingBox> const& dbb, std::vector<float> const& dconf, std::vector<BoundingBox> &cbb, std::vector<float> &cconf)
{
    int numbb = dbb.size();
    std::vector<int> T;
    float space_thr = 0.5;
    int c = 1;
    switch (numbb){
        case 1:
            cbb = std::vector<BoundingBox>(1, dbb[0]);
            cconf = std::vector<float>(1, dconf[0]);
            return;
            break;
        case 2:
            T =std::vector<int>(2,0);
            if (1-bbOverlap(dbb[0],dbb[1]) > space_thr) {
                T[1] = 1;
                c = 2;
            }
            break;
        default:
            T = std::vector<int>(numbb,0);
            c = cv::partition(dbb, T, (*bbComp));
            //c = clusterBB(dbb,T);
            break;
    }
    cconf = std::vector<float>(c);
    cbb = std::vector<BoundingBox>(c);
    BoundingBox bx;
    for (int i=0; i<c; i++){
        float cnf = 0;
        int N=0, mx=0, my=0, mw=0, mh=0;
        for (int j=0; j<T.size(); j++){
            if (T[j] == i) {
                cnf = cnf+dconf[j];
                mx = mx+dbb[j].x;
                my = my+dbb[j].y;
                mw = mw+dbb[j].width;
                mh = mh+dbb[j].height;
                N++;
            }
        }
        if (N > 0){
            cconf[i] = cnf/N;
            bx.x = cvRound(mx/N);
            bx.y = cvRound(my/N);
            bx.width = cvRound(mw/N);
            bx.height = cvRound(mh/N);
            cbb[i] = bx;
        }
    }
}
