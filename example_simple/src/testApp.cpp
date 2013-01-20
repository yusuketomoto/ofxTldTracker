#include "testApp.h"

static const float scale_factor = 2.0;

//--------------------------------------------------------------
void testApp::setup(){
    ofSetVerticalSync(true);
    ofBackground(0);
    
    cam.initGrabber(320, 240);
    tld.setup();
}

//--------------------------------------------------------------
void testApp::update(){
    cam.update();
    if(cam.isFrameNew()) {
        tld.update(toCv(cam));
    }
}

//--------------------------------------------------------------
void testApp::draw(){
    ofScale(scale_factor, scale_factor);
    cam.draw(0,0);
    
    ofNoFill();
    ofSetColor(255);
    tld.draw();
}

//--------------------------------------------------------------
void testApp::keyPressed(int key){
    if (key == 'r') {
        tld.reset();
    }
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){
    tld.mouseDragged(x/scale_factor, y/scale_factor);
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){
    tld.mousePressed(x/scale_factor, y/scale_factor);
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){
    tld.mouseReleased(x/scale_factor, y/scale_factor);
}
