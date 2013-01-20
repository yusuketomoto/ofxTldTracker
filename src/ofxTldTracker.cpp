#include "ofxTldTracker.h"

ofxTldTracker::ofxTldTracker()
:tld_state(TLD_STATE_GETBOUNDINGBOX)
,is_enable_track_and_learn(true)
,has_got_bounding_box(false)
,is_detect(false)
,is_drawing_box(false)
{
}

void ofxTldTracker::setup()
{
    std::string filename = ofToDataPath("parameters.yml");
    cv::FileStorage fs;
    fs.open(filename, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        std::cerr << "Couldn't read parameters file." << std::endl;
        std::cerr << "Please put it in your data folder." << std::endl;
        exitApp();
    }
    
    tld.readParams(fs.getFirstTopLevelNode());    
}

void ofxTldTracker::update(cv::Mat const& image)
{
    switch (tld_state) {
        case TLD_STATE_GETBOUNDINGBOX:
            if (has_got_bounding_box) {
                tld_state = TLD_STATE_INITIALIZE;
                is_detect = true;
                num_processed_frames = 1;
                num_detected_frames = 1;
            }
            break;
            
        case TLD_STATE_INITIALIZE:
            image.copyTo(frame);
            cv::cvtColor(frame, last_gray, CV_RGB2GRAY);
            tld.init(last_gray, bounding_box);
            tld_state = TLD_STATE_REPEAT;
            break;
            
        case TLD_STATE_REPEAT:
            image.copyTo(frame);
            cv::cvtColor(frame, current_gray, CV_RGB2GRAY);
            points1.clear();
            points2.clear();
            
            tld.processFrame(last_gray, current_gray, points1, points2, predicted_box, is_detect, is_enable_track_and_learn);
            if (isDetect()) {
                num_detected_frames++;
            }
            cv::swap(last_gray, current_gray);
            num_processed_frames++;
            break;
    }
}

void ofxTldTracker::reset()
{
    tld.clear();
    
    tld_state = TLD_STATE_GETBOUNDINGBOX;
    is_detect = false;
    has_got_bounding_box = false;
    num_detected_frames = 0;
    num_processed_frames = 0;
    
    bounding_box = cv::Rect();
    predicted_box = BoundingBox();
}

void ofxTldTracker::draw()
{
    ofNoFill();
    
    switch (tld_state) {
        case TLD_STATE_GETBOUNDINGBOX:
            if (is_drawing_box) {
                ofRect(bounding_box.x, bounding_box.y, bounding_box.width, bounding_box.height);
            }
            break;
        case TLD_STATE_INITIALIZE:
            if (is_drawing_box) {
                ofRect(bounding_box.x, bounding_box.y, bounding_box.width, bounding_box.height);
            }
            break;
        case TLD_STATE_REPEAT:
            if (isDetect()) {
                ofRect(predicted_box.x, predicted_box.y, predicted_box.width, predicted_box.height);
                drawPoints(points1);
                drawPoints(points2, cv::Scalar(0, 255, 0));
            }
            break;
    }
}

void ofxTldTracker::drawPoints(std::vector<cv::Point2f> points,cv::Scalar color) const
{
    ofSetColor(color[0], color[1], color[2]);
    for (std::vector<cv::Point2f>::iterator it = points.begin(); it!=points.end(); ++it) {
        ofCircle(it->x, it->y, 2);
    }
    ofSetColor(255);
}

void ofxTldTracker::setBoundingBox(float x, float y, float width, float height)
{
    bounding_box = cv::Rect(x, y, width, height);
    has_got_bounding_box = true;
}

void ofxTldTracker::setBoundingBox(ofRectangle const& r)
{
    bounding_box = ofxCv::toCv(r);
    has_got_bounding_box = true;
}


void ofxTldTracker::mouseDragged(int x, int y)
{
    if (is_drawing_box) {
        bounding_box.width = x-bounding_box.x;
        bounding_box.height = y-bounding_box.y;
    }
}

void ofxTldTracker::mousePressed(int x, int y)
{
    is_drawing_box = true;
    bounding_box = cv::Rect( x, y, 0, 0 );
}

void ofxTldTracker::mouseReleased(int x, int y)
{
    is_drawing_box = false;
    if( bounding_box.width < 0 ){
        bounding_box.x += bounding_box.width;
        bounding_box.width *= -1;
    }
    if( bounding_box.height < 0 ){
        bounding_box.y += bounding_box.height;
        bounding_box.height *= -1;
    }
    has_got_bounding_box = true;
}

std::vector<ofVec2f> ofxTldTracker::getPointsToTrack() const
{
    int n = points1.size();
    std::vector<ofVec2f> ret_points(n);
    for (int i=0; i<n; i++) {
        ret_points[i] = ofxCv::toOf(points1[i]);
    }
    return ret_points;
}

std::vector<ofVec2f> ofxTldTracker::getReliablePointsToTrack() const
{
    int n = points2.size();
    std::vector<ofVec2f> ret_points(n);
    for (int i=0; i<n; i++) {
        ret_points[i] = ofxCv::toOf(points2[i]);
    }
    return ret_points;
}