/*
[Interspecifics] Comunicaciones Especulativas
Player1: An app to see them all.
-------------------------------------------------
[2017.12] APP:
Video Library // CVision // MLearning // AI // OSC composer // MicroCtrl

The app takes an incoming image files list
The app analyses the images using different techniques
The app produces OSC data stream with result of analysis

v0.1: load differnt file directories and get optic flow representations
v0.2: add optical processing: colormap, threshold, blob tracking;
v0.4: optoflow, 3 displays: input/flow/blob
v0.8: send osc for ofblob, sblob, column movement
V1.0
*/



#pragma once

#include "ofMain.h"
#include "ofxOsc.h"
#include "ofxCv.h"
#include "ofxOpenCv.h"
#include "ofxImageSequenceRecorder.h"
#include "ofxGui.h"
#include "ofxXmlSettings.h"

//#include "ofxFBOTexture.h"

#define OSC_HOST "localhost"
#define OSC_IN_PORT 11111
#define OSC_OUT_PORT 10001

class Stuff : public ofxCv::RectFollower {
protected:
	ofColor color;
	ofVec2f cur, smooth;
	float startedDying;
	ofPolyline all;
public:
	Stuff()
		:startedDying(0) {
	}
	void setup(const cv::Rect& track);
	void update(const cv::Rect& track);
	void kill();
	void draw();
};


class ofApp : public ofBaseApp{

public:
	void setup();
	void update();
	void draw();

	void load_dir();

	void keyPressed(int key);
	void keyReleased(int key);
	void exit();

	ofVideoGrabber camera;
	cv::Mat grabberGray;
	ofxCv::FlowPyrLK flow;
	ofVec2f p1;
	ofRectangle rect;
	vector<cv::Point2f> feats;

	// ------ ------ ------ ------ ------ ------ | gui |
	bool showGui;
	ofxPanel gui;

	ofxGuiGroup inputGroup;
	ofxGuiGroup flowGroup, diffGroup;
	ofxGuiGroup outputGroup;

	ofxToggle load_new;
	ofxFloatSlider init_seq, len_seq, vel_seq;
	ofxToggle en_edges;

	ofxToggle en_Optoflow;
	ofParameter<float> lkQualityLevel;
	ofParameter<int> lkMaxLevel, lkMinDistance;
	ofParameter<float> edge_thr1, edge_thr2;
	ofxToggle draw_colormap;
	ofxFloatSlider cut_colormap;
	ofxToggle alpha_colormap;
	ofxToggle en_trackOpto;
	ofxToggle en_labelOpto;

	ofxToggle en_Simpleblob;
	ofxFloatSlider ampli, diff_thresh, contour_thresh;
	ofxToggle en_track;

	ofxIntSlider val_ofColumn;
	ofxToggle send_ofColumn;
	ofxToggle send_ofBlobs;
	ofxToggle send_sBlobs;

	// ------ ------ ------ ------ ------ ------ | osc |
	ofxOscReceiver osc_recver;
	ofxOscSender osc_sender;

	// ------ ------ ------ ------ ------ ------ | rec stuff |
	ofxImageSequenceRecorder img_recorder;
	bool isRecording;

	// ------ ------ ------ ------ ------ ------ | imgs |
	ofDirectory dir;
	bool bLoaded;
	
	int nImgs;
	int ih, iw;
	int gridSz, scale;
	bool blackFlag;
	int ind_ff;
	int t0, t1, index1;
	vector<ofImage> imgList;
	vector<cv::Mat> matList;
	cv::Mat tempMat;
	ofImage tempImg;
	ofFbo fbo_A, fbo_B, fbo_C;
	ofPixels pixs;
	ofColor cColor, nColor;
	ofImage optoMap;
	ofFbo buffer;

	ofxCv::FlowPyrLK LK;
	ofxCv::FlowFarneback FB;
	ofxCv::Flow* curFlow;
	vector<ofPoint> old_feats;
	vector<ofVec2f> new_feats;

	// --------------------- NRML
	int w, h, t;
	float tt;

	// ...... mode blob
	ofImage gray, edge, sobel;
	cv::Mat prevImg, diffImg, actImg, grayImg;
	ofImage nImg, tImg;

	ofxCv::ContourFinder cFinder;
	ofxCv::RectTrackerFollower<Stuff> tracker;
	ofxCv::ContourFinder cFinderOpto;
	ofxCv::RectTrackerFollower<Stuff> trackerOpto;
};



