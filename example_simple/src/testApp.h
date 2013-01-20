#pragma once

#include "ofMain.h"
#include "ofxCv.h"
using namespace ofxCv;
using namespace cv;

#include "ofxTldTracker.h"

class testApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed  (int key);
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
    
    ofVideoGrabber cam;
    ofxTldTracker tld;
};
