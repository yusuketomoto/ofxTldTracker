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
        video.draw(0, 0);
        
        ofRectangle predict = tracker.getBoundingBox();
        if (predict.getArea() > 0) {
            float w = predict.width;
            float h = predict.height;
            int grid_size = 10;
            const unsigned char* pix = video.getPixels();
            ofPushStyle();
            ofFill();
            ofPushMatrix();
            ofTranslate(predict.x, predict.y);
            for (int y=0; y<h; y+=grid_size) {
                for (int x=0; x<w; x+=grid_size) {
                    int index = ((y+predict.y)*video.width + x + predict.x)*3;
                    ofSetColor(pix[index], pix[index+1], pix[index+2]);
                    ofRect(x, y, grid_size, grid_size);
                }
            }
            ofPopMatrix();
            ofPopStyle();
        }
        else
        {
            tracker.draw();
        }
    }
    void keyPressed(int key) {
        if (key == 'r') tracker.reset();
    }
    void mousePressed(int x, int y, int button) {
        tracker.mousePressed(x, y);
    }
    void mouseDragged(int x, int y, int button) {
        tracker.mouseDragged(x, y);
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
