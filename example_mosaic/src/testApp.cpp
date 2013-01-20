#include "testApp.h"
static const float scale_factor = 2.0;
bool is_drawing_box = false;
ofRectangle bounding_box;

bool isInside(int x, int y, int width, int height)
{
    if (x < 0)       return false;
    if (x >= width)  return false;
    if (y < 0)       return false;
    if (y >= height) return false;
    return true;
}

void drawMosaic(ofRectangle const& r, ofPixels const& frame)
{
    if (r.width==0)  return false;
    if (r.height==0) return false;
    
    ofEnableBlendMode(OF_BLENDMODE_DISABLED);
    const int mosaic_size = 10;
    
    ofImage mosaic_patch;
    mosaic_patch.allocate(r.width, r.height, OF_IMAGE_COLOR);
    
    for (int j=0; j<r.height; j+=mosaic_size) {
        for (int i=0; i<r.width; i+=mosaic_size) {
            
            int rsum=0, gsum=0, bsum=0, cnt=0;
            
            for (int y=0; y<mosaic_size; y++) {
                for (int x=0; x<mosaic_size; x++) {
                    if (isInside(r.x+i+x, r.y+j+y, frame.getWidth(), frame.getHeight()) && isInside(i+x, j+y, r.width, r.height)) {
                        ofColor_<unsigned char> color = frame.getColor(r.x+i+x, r.y+j+y);
                        rsum += color.r;
                        gsum += color.g;
                        bsum += color.b;
                        cnt++;
                    }
                }
            }
            
            ofColor local_mean;
            if (cnt>0) {
                local_mean = ofColor(rsum/cnt, gsum/cnt, bsum/cnt);
            } else {
                local_mean = ofColor(0,0,0);
            }
            
            for (int y=0; y<mosaic_size; y++) {
                for (int x=0; x<mosaic_size; x++) {
                    if (isInside(r.x+i+x, r.y+j+y, frame.getWidth(), frame.getHeight()) && isInside(i+x, j+y, r.width, r.height)) {
                        mosaic_patch.setColor(i+x, j+y, local_mean);
                    }
                }
            }
            
        }
    }
    mosaic_patch.update();
    mosaic_patch.draw(r.x, r.y);
}



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
    
    if (is_drawing_box) {
        ofRect(bounding_box);
    }
    if (tld.isDetect()) {
        ofRectangle predicted_box = tld.getBoundingBox();
        drawMosaic(predicted_box, cam.getPixelsRef());
    }
}

//--------------------------------------------------------------
void testApp::keyPressed(int key){
    if (key == 'r') {
        tld.reset();
    }
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){
    float scaled_x = x / scale_factor;
    float scaled_y = y / scale_factor;
    if (is_drawing_box) {
        bounding_box.width = scaled_x-bounding_box.x;
        bounding_box.height = scaled_y-bounding_box.y;
    }
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){
    float scaled_x = x / scale_factor;
    float scaled_y = y / scale_factor;
    is_drawing_box = true;
    bounding_box = ofRectangle(scaled_x,scaled_y,0,0);
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){
    is_drawing_box = false;
    if (bounding_box.width < 0) {
        bounding_box.x += bounding_box.width;
        bounding_box.width *= -1;
    }
    if (bounding_box.height < 0) {
        bounding_box.y += bounding_box.height;
        bounding_box.height *= -1;
    }
    tld.setBoundingBox(bounding_box);
}
