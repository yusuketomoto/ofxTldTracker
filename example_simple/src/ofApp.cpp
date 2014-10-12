#include "ofMain.h"
#include "ofxTldTracker.h"

class ofApp : public ofBaseApp
{
    ofVideoGrabber video;
    ofxTldTracker tracker;
    
public:
    void setup() {
        ofSetFrameRate(60);
        ofSetVerticalSync(true);
        ofBackground(0);
        
        video.initGrabber(640, 480);
        tracker.setup();
    }
    void update() {
        video.update();
        if (video.isFrameNew()) {
            tracker.update(video);
        }
    }
    void draw() {
        video.draw(0,0);
        tracker.draw();
    }
    void keyPressed(int key) {
        if (key == 'r') tracker.reset();
    }
    void mouseDragged(int x, int y, int button) {
        tracker.mouseDragged(x, y);
    }
    void mousePressed(int x, int y, int button) {
        tracker.mousePressed(x, y);
    }
    void mouseReleased(int x, int y, int button) {
        tracker.mouseReleased(x, y);
    }
};

//========================================================================
int main( ){
    ofSetupOpenGL(640,480,OF_WINDOW);
    ofRunApp(new ofApp());
}
