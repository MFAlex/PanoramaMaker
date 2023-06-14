#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/xfeatures2d/cuda.hpp>
#include "opencv2/core/cuda.hpp"
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/warpers.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <chrono>
#include <thread>
#include <mutex>

using namespace cv;
using namespace std;

cuda::GpuMat lastGpuImg;

static void download(const cuda::GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void upload(cuda::GpuMat& d_mat, vector<Point2f>& vec)
{
    Mat mat(1, vec.size(), CV_32FC2, (void*)&vec[0]);
    d_mat.upload(mat);
}

static void download(const cuda::GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}

Mat concatenateMatrix(Mat first, Mat second) {

    Mat mul1 = Mat::eye(3, 3, CV_64F);
    Mat mul2 = Mat::eye(3, 3, CV_64F);
    Mat x_;
    Mat temp_inv_;
    Mat mul_r;
    first.convertTo(temp_inv_, CV_64F);
    second.convertTo(x_, CV_64F);

    temp_inv_.row(0).copyTo(mul1.row(0));
    temp_inv_.row(1).copyTo(mul1.row(1));

    x_.row(1).copyTo(mul2.row(1));
    x_.row(0).copyTo(mul2.row(0));

    try {
        mul_r = mul1 * mul2;
    }
    catch (Exception& e) {
        const char* err_msg = e.what();
        cout << err_msg;
    }

    mul1.release();
    mul2.release();
    temp_inv_.release();

    return mul_r;
}

void savePanorama(int fromFrame, int toFrame, string imgPath);

Ptr<cuda::SparsePyrLKOpticalFlow> pyrLK = cuda::SparsePyrLKOpticalFlow::create(
    Size(9, 9), 4, 50);
Ptr<cuda::SparsePyrLKOpticalFlow> pyrLKBack = cuda::SparsePyrLKOpticalFlow::create(
    Size(9, 9), 4, 30, true);
Ptr<cuda::CornersDetector> detector;
cuda::GpuMat ofStatusForward;
cuda::GpuMat ofDstForward;
cuda::GpuMat ofStatusBackwards;
cuda::GpuMat ofDstBackwards;

int twoWayOpticalFlow(cuda::GpuMat srcImg, cuda::GpuMat dstImg, cuda::GpuMat srcPoints, vector<Point2f>* dstPoints, vector<Point2f>* dstPointsSrc) {
    pyrLK->calc(srcImg, dstImg, srcPoints, ofDstForward, ofStatusForward);
    srcPoints.copyTo(ofDstBackwards);
    pyrLKBack->calc(dstImg, srcImg, ofDstForward, ofDstBackwards, ofStatusBackwards);

    vector<Point2f> fwOF, bkOF, srcPts;
    vector<uchar> fwStatus, bkStatus;
    download(ofDstForward, fwOF);
    download(ofDstBackwards, bkOF);
    download(srcPoints, srcPts);
    download(ofStatusForward, fwStatus);
    download(ofStatusBackwards, bkStatus);

    int size = fwOF.size();
    dstPoints->clear();
    dstPointsSrc->clear();
    if (bkOF.size() != size || fwStatus.size() != size || bkStatus.size() != size || srcPts.size() != size) {
        return 0; //only when something goes wrong somehow
    }
    else {
        dstPoints->reserve(size);
        dstPointsSrc->reserve(size);
        Point2f bkPt, srcPt;
        float dSqr;
        for (int i = 0; i < size; i++) {
            bkPt = bkOF[i];
            srcPt = srcPts[i];
            if (fwStatus[i] && bkStatus[i]) {
                dSqr = pow(srcPt.x - bkPt.x, 2) + pow(srcPt.y - bkPt.y, 2);
                if (dSqr < 1) {
                    dstPoints->push_back(fwOF[i]);
                    dstPointsSrc->push_back(srcPt);
                }
            }
        }
        return dstPoints->size();
    }
}

cuda::GpuMat canvas;
Mat element = cv::getStructuringElement(MORPH_RECT, Size(5, 5));
Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_ERODE, CV_8UC1, element);
cv::Mat cumulativeTransform;
vector<Mat> forwardPassHomography;
vector<cuda::GpuMat> forwardPassImages;
Mat lastTransform;
Size proposed_dimensions;
Point2d proposed_offset_g(0, 0);
cuda::GpuMat originalBackwardsFrame;

void resetForBuilding(cuda::GpuMat img)
{
    img.copyTo(canvas);
    cumulativeTransform = Mat::eye(3, 3, CV_64F); //reset to identify matrix
    forwardPassHomography.clear();
    forwardPassImages.clear();
    lastTransform = Mat::eye(3, 3, CV_64F);
    proposed_dimensions = Size(img.cols, img.rows);
    proposed_offset_g = Point2d(0, 0);
    img.copyTo(originalBackwardsFrame);
}

void buildPanorama(cuda::GpuMat img, Mat H2)
{
    Mat H;
    H.create(3, 3, H2.type());
    H2.copyTo(H.rowRange(0, 2));
    H.at<double>(2, 0) = 0.0;
    H.at<double>(2, 1) = 0.0;
    H.at<double>(2, 2) = 1.0;

    cuda::GpuMat copy;
    img.copyTo(copy);
    forwardPassHomography.push_back(H);
    forwardPassImages.push_back(copy);
    cumulativeTransform = concatenateMatrix(cumulativeTransform, H);

    vector<Point2d> srcPerspective, dstPerspective;
    srcPerspective.push_back(Point2d(0, 0));
    srcPerspective.push_back(Point2d(0, img.rows - 1));
    srcPerspective.push_back(Point2d(img.cols - 1, img.rows - 1));
    srcPerspective.push_back(Point2d(img.cols - 1, 0));

    perspectiveTransform(srcPerspective, dstPerspective, cumulativeTransform);
    Point2d proposed_offset(0, 0);
    int amount;
    for (vector<Point2d>::const_iterator it = dstPerspective.begin(); it != dstPerspective.end(); ++it) {
        if (it->x < 0) {
            amount = (0 - it->x) + 1;
            if (amount > proposed_offset.x) {
                proposed_dimensions.width += amount;
                proposed_offset.x += amount;
            }
        }
        if (it->x > proposed_dimensions.width) {
            amount = it->x + 1;
            if (amount > proposed_dimensions.width) {
                proposed_dimensions.width = amount + proposed_offset.x;
            }
        }
        if (it->y < 0) {
            amount = (0 - it->y) + 1;
            if (amount > proposed_offset.y) {
                proposed_dimensions.height += amount;
                proposed_offset.y += amount;
            }
        }
        if (it->y > proposed_dimensions.height) {
            amount = it->y + 1;
            if (amount > proposed_dimensions.height) {
                proposed_dimensions.height = amount + proposed_offset.y;
            }
        }
    }
    proposed_offset_g.x += proposed_offset.x;
    proposed_offset_g.y += proposed_offset.y;

    if (proposed_dimensions.height > 15000 or proposed_dimensions.width > 20000) {
        cout << "Panorama would've exceeded max image dimensions" << endl; //vram considerations
        return;
    }
    cuda::GpuMat newImg(proposed_dimensions, canvas.type());
    canvas.copyTo(newImg.colRange(proposed_offset.x, proposed_offset.x + canvas.cols).rowRange(proposed_offset.y, proposed_offset.y + canvas.rows));
    canvas.release();
    canvas = newImg;

    if (proposed_offset.x > 0 || proposed_offset.y > 0) {
        ((double*)cumulativeTransform.data)[2] += proposed_offset.x;
        ((double*)cumulativeTransform.data)[5] += proposed_offset.y;
        perspectiveTransform(srcPerspective, dstPerspective, cumulativeTransform);
    }

    cuda::GpuMat transformedImg;
    cuda::warpPerspective(img, transformedImg, cumulativeTransform, Size(canvas.cols, canvas.rows));

    vector<Point> dstPointsF32;
    for (vector<Point2d>::const_iterator it = dstPerspective.begin(); it != dstPerspective.end(); ++it) {
        dstPointsF32.push_back(Point(it->x, it->y));
    }
    Mat maskUndilated(canvas.rows, canvas.cols, CV_8UC1, Scalar(0));
    fillConvexPoly(maskUndilated, dstPointsF32, Scalar(255));
    cuda::GpuMat newAreaMask;
    newAreaMask.upload(maskUndilated);

    dilateFilter->apply(newAreaMask, newAreaMask);
    cuda::cvtColor(newAreaMask, newAreaMask, COLOR_GRAY2BGR);

    cuda::GpuMat notmask;
    cuda::bitwise_not(newAreaMask, notmask);

    cuda::GpuMat combine1;
    cuda::bitwise_and(transformedImg, newAreaMask, combine1);
    cuda::GpuMat combine2;
    cuda::bitwise_and(canvas, notmask, combine2);
    //cuda::GpuMat combine3;
    cuda::bitwise_or(combine1, combine2, canvas);

    //combine3.copyTo(canvas);
}

void backwardPass(int fromFrame, int toFrame, string imgPath)
{
    cuda::GpuMat work = cuda::GpuMat(proposed_dimensions, CV_8UC3);
    cuda::GpuMat workMask = cuda::GpuMat(proposed_dimensions, CV_8UC1);
    cout << "Creating blended panorama with dimensions: " << proposed_dimensions << endl;
    cuda::GpuMat whitemask = cuda::GpuMat(forwardPassImages[0].size(), CV_8UC1);
    whitemask.setTo(Scalar(255));

    vector<Point> dummyPoints;
    vector<Size> dummySizes;
    dummyPoints.reserve(forwardPassHomography.size());
    dummySizes.reserve(forwardPassHomography.size());
    for (int i = 0; i < forwardPassHomography.size(); i++) {
        dummyPoints.push_back(Point(0, 0));
        dummySizes.push_back(proposed_dimensions);
    }
    Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(cv::detail::Blender::FEATHER, true);
    blender->prepare(dummyPoints, dummySizes);

    Mat localWork, localMask, localWork16;

    Mat position = Mat::eye(3, 3, CV_64F);
    ((double*)position.data)[2] = proposed_offset_g.x;
    ((double*)position.data)[5] = proposed_offset_g.y;

    //load 0th frame
    work.setTo(Scalar(0, 0, 0));
    cuda::warpPerspective(originalBackwardsFrame, work, position, proposed_dimensions);
    workMask.setTo(Scalar(0, 0, 0));
    cuda::warpPerspective(whitemask, workMask, position, proposed_dimensions);
    dilateFilter->apply(workMask, workMask);
    work.download(localWork);
    workMask.download(localMask);
    localWork.convertTo(localWork16, CV_16SC3);
    blender->feed(localWork16, localMask, Point(0, 0));

    for (int i = 0; i < forwardPassHomography.size(); i++) {
        Mat homography = forwardPassHomography[i];
        position = concatenateMatrix(position, homography);

        cuda::GpuMat frame = forwardPassImages[i];
        work.setTo(Scalar(0, 0, 0));
        cuda::warpPerspective(frame, work, position, proposed_dimensions);
        workMask.setTo(Scalar(0, 0, 0));
        cuda::warpPerspective(whitemask, workMask, position, proposed_dimensions);
        dilateFilter->apply(workMask, workMask);
        work.download(localWork);
        workMask.download(localMask);

        localWork.convertTo(localWork16, CV_16SC3);
        blender->feed(localWork16, localMask, Point(0, 0));
    }
    Mat result, result_mask;
    blender->blend(result, result_mask);

    string fname = "pano_" + to_string(fromFrame) + "_" + to_string(toFrame - fromFrame) + "_blended.png";
    //cout << "Panorama from frame " << fromFrame << " to frame " << toFrame << ". Size: [" << canvas.cols << "x" << canvas.rows << "] " << fname << endl;
    imwrite(imgPath + "/" + fname, result);
}

int absFrame = 0; //change to resume from a certain frame
int decodedFrameNum = absFrame;
cuda::GpuMat gpuImg;
cuda::GpuMat decodedFrameTemp;
std::mutex frameLock;
VideoCapture cap;
bool reachedEnd = false;

string fname;
void* frameThread() {
    Mat img;
    bool done;
    int numBreaks = 0;
    while (true) {
        while (!cap.grab()) {
            numBreaks++;
            if (numBreaks >= 100) {
                cout << "100 missed frames in a row? I think we're at the end of the video." << endl;
                reachedEnd = true;
                return nullptr;
            }
            continue;
        }
        if (numBreaks > 0) {
            cout << "Skipped " << numBreaks << " frames. Corrupt video file?" << endl;
        }
        done = cap.retrieve(img) && img.data;
        if (!done) break;
        numBreaks = 0;
        decodedFrameTemp.upload(img);

        frameLock.lock();
        {
            decodedFrameTemp.copyTo(gpuImg);
            decodedFrameNum += 1;
        }
        frameLock.unlock();
    }
    reachedEnd = true;
    return nullptr;
}

int main(int argc, const char* argv[])
{
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <video file path> <image save path>" << endl;
        return 1;
    }

    
    int panoFrom = 0;
    double movementX = 0.0, movementY = 0.0, linearMovement = 0.0;
    fname = argv[1];
    const cv::String imgPath(argv[2]);

    cout << "********** LAUNCH PARAMS **********" << endl;
    cout << "Analysing file \"" << fname << "\"" << endl;
    cout << "Saving panoramas into folder \"" << imgPath << "\"" << endl;
    cout << "***********************************" << endl;

    cap.open(fname, CAP_FFMPEG);
    cap.set(CAP_PROP_POS_FRAMES, absFrame);

    Mat img;

    cuda::GpuMat grayFrame;
    cuda::GpuMat d_prevPts;
    cuda::GpuMat d_status;


    bool grabbedFirstFrame = cap.grab() && cap.retrieve(img) && img.data;
    cout << "Video opened: " << grabbedFirstFrame << endl;
    if (!grabbedFirstFrame) {
        return 1;
    }

    lastGpuImg.upload(img);
    cuda::cvtColor(lastGpuImg, grayFrame, COLOR_BGR2GRAY);


    detector = cuda::createGoodFeaturesToTrackDetector(grayFrame.type(), 4000, 0.00001, 10.0, 5, true, 0.043);

    detector->detect(grayFrame, d_prevPts);
    resetForBuilding(lastGpuImg);

    int i = 0;
    bool hadFrame = false;
    bool hadNoPoints = true;

    std::thread thread(&frameThread);
    thread.detach();

    cout << "It hasn't crashed yet, you're probably fine..." << endl;
    while (true) {
        while (absFrame >= decodedFrameNum && !reachedEnd) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            //wait for frame to be available
        }

        if (reachedEnd) break;

        frameLock.lock();
        absFrame = decodedFrameNum;

        vector<Point2f> outputPoints, sourcePoints;

        int l = twoWayOpticalFlow(lastGpuImg, gpuImg, d_prevPts, &outputPoints, &sourcePoints);
        cuda::cvtColor(gpuImg, grayFrame, COLOR_BGR2GRAY);
        detector->detect(grayFrame, d_prevPts);

        /*Mat draw;
        gpuImg.download(draw);
        for (int i = 0; i < l; i++) {
            line(draw, sourcePoints[i], outputPoints[i], Scalar(0, 255, 0));
        }
        imshow("show", draw);
        waitKey(100);*/

        if (l >= 30) {
            if (hadNoPoints) {
                hadNoPoints = false;
                panoFrom = absFrame - 1;
                movementX = 0;
                movementY = 0;
                resetForBuilding(lastGpuImg);
            }
            Mat H = estimateAffinePartial2D(outputPoints, sourcePoints, noArray(), LMEDS);
            if (!H.empty()) {
                //todo translate midpoint of frame by homography instead of using translation component
                hadFrame = true;
                movementX += ((double*)H.data)[2];
                movementY += ((double*)H.data)[5];
                double velocity = sqrt(pow(((double*)H.data)[2], 2) + pow(((double*)H.data)[5], 2));
                if (velocity < 2) {
                    double cumulativeMovement = (linearMovement = sqrt(movementX * movementX + movementY * movementY));
                    if (cumulativeMovement > 100) {
                        cout << cumulativeMovement << endl;
                        savePanorama(panoFrom, absFrame, imgPath);
                    }
                    panoFrom = absFrame;
                    movementX = 0;
                    movementY = 0;
                    resetForBuilding(gpuImg);
                }
                else {
                    buildPanorama(gpuImg, H);
                }
            }
            else {
                cout << "No homography [" << absFrame << "]" << endl;
            }
        }
        else if ((linearMovement = sqrt(movementX * movementX + movementY * movementY)) > 100) {
            savePanorama(panoFrom, absFrame, imgPath);
            resetForBuilding(gpuImg);
            panoFrom = absFrame;
            movementX = 0.0;
            movementY = 0.0;
        }
        else if (hadFrame) {
            hadFrame = false;
            hadNoPoints = true;
            cout << "Camera shot [" << absFrame << "] (" << l << ")" << endl;
            panoFrom = absFrame;
            movementX = 0.0;
            movementY = 0.0;
        }
        else {
            hadNoPoints = true;
        }

        gpuImg.copyTo(lastGpuImg);
        frameLock.unlock();
    }
    cout << "Graceful exit at frame " << absFrame << endl;

    return 0;
}

void savePanorama(int fromFrame, int toFrame, string imgPath) {
    string fname = "pano_" + to_string(fromFrame) + "_" + to_string(toFrame - fromFrame) + ".png";
    cout << "Panorama from frame " << fromFrame << " to frame " << toFrame << ". Size: [" << canvas.cols << "x" << canvas.rows << "] " << fname << endl;
    Mat img;
    canvas.download(img);
    imwrite(imgPath + "/" + fname, img);
    backwardPass(fromFrame, toFrame, imgPath);
}