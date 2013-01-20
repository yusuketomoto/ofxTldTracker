#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "TLD.h"

class ofxTldTracker
{
public:
    ofxTldTracker();
    void setup();
    void update(cv::Mat const& image);
    void reset();
    void draw();
    
    // you have two ways to set bounding box.
    // one is inputting parameters directly, and another is simply using mouse callbacks.
    void setBoundingBox(float x, float y, float width, float height);
    void setBoundingBox(ofRectangle const& r);

    void mouseDragged(int x, int y);
    void mousePressed(int x, int y);
    void mouseReleased(int x, int y);

    // if you want to stop re-training(updating) classifier, then should be set disable.
    void enableTrackAndLearn()  { is_enable_track_and_learn = true; } //Track Learn Detection
    void disableTrackAndLearn() { is_enable_track_and_learn = false; } //Detection only
    
    // getters
    ofRectangle getBoundingBox() const { return ofxCv::toOf(predicted_box); }
    ofPoint getCentroid() const {
        ofPoint centroid(predicted_box.x+predicted_box.width/2.0, predicted_box.y+predicted_box.height/2.0);
        return centroid;
    }

    std::vector<ofVec2f> getPointsToTrack() const;
    std::vector<ofVec2f> getReliablePointsToTrack() const;

    int getProcessedFrames() const { return num_processed_frames; }
    int getDetectedFrames()  const { return num_detected_frames; }
    float getDetectionRate() const { return static_cast<float>(num_detected_frames) / static_cast<float>(num_processed_frames); }
    
    bool isDetect() const { return is_detect; }
    
private:
    void drawPoints(std::vector<cv::Point2f> points,cv::Scalar color=cv::Scalar::all(255)) const;

private:
    TLD tld;
    
    enum {
        TLD_STATE_GETBOUNDINGBOX,
        TLD_STATE_INITIALIZE,
        TLD_STATE_REPEAT
    } tld_state;
    
    bool is_enable_track_and_learn;
    bool has_got_bounding_box;
    bool is_detect;
    bool is_drawing_box;
    
    int num_processed_frames;
    int num_detected_frames;
    
    cv::Mat frame;
    cv::Mat last_gray;
    cv::Mat current_gray;
    Points points1;
    Points points2;
    cv::Rect bounding_box;
    
    BoundingBox predicted_box;
};