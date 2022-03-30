//
// Created by lgj on 12/31/19.
//

#ifndef ORB_SLAM2_LK_H
#define ORB_SLAM2_LK_H

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <Frame.h>
using namespace std;
using namespace ORB_SLAM2;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
// using namespace cv;

vector<cv::Point2f> keypoints;
vector<cv::Point2f> prev_keypoints;
vector<cv::Point3f> mappointInCurrentFrame;
cv::Mat last_color; // last imgage

/**
 * 暂未解决 无法把所有帧的位姿加入CameraTrajectory.txt的问题
 * **/

/**
 * @brief
 * 非关键帧的情况下，使用光流法跟踪关键帧的特征点,使用 PNP_RANSAC 来计算相机位姿
 *
 * @param[in] lastKeyFrame      pointer to last keyframe
 * @param[in] color         current frame RGB imgage
 * @param[in] lastColorIsKeyFrame   if last frame is keyframe
 * @param[in] K
 * @param[in] DistCoef
 * @param[out] Tcw          current frame pose
 * @param[out] mnMatchesInliers     the number of current frame Matches Inliers
 *
 * @return cv::Mat img_show; //返回值是rgb图片，用于显示光流跟踪到的特征点
 *
 * Step 1 ：
 * Step 2 ：
 * Step 3 ：
 *
 */
cv::Mat computeMtcwUseLK(
    KeyFrame *lastKeyFrame,
    cv::Mat color,
    bool lastColorIsKeyFrame,
    cv::Mat K, cv::Mat DistCoef,
    cv::Mat &Tcw,
    int &mnMatchesInliers)
{
    // cout << "   entering LK.h" << endl;

    // observation time threshold
    int obsPlus = 0;
    if (lastKeyFrame->mnId > 3)
        obsPlus = 3;

    // first time to call this function, last RGB image is not existed,
    // so LK cannot work, here can be regarded as LK initialization
    if (last_color.empty())
    {
        cout << "       filling last color fist time" << endl;
        last_color = color;

        // Tcw, mnMatchesInliers not modified, and return directly
        //! bug: if LK is not initied,
        //! the current frame should be tracking by original ORBSLAM
        mnMatchesInliers = -1;
        return cv::Mat();
    }

    // 上一帧是关键帧的情况
    // use last keyframe's mappoints as current frame's mappoints
    // use last keyframe's keypoints as current frame's keypoints
    if (lastColorIsKeyFrame || keypoints.empty())
    {
        // cout << lastKeyFrame->mvKeysUn.size() << endl;
        keypoints.clear();
        mappointInCurrentFrame.clear();

        // copy map points from last keyframe
        // ! make KeyFrame member mvpMapPoints public
        for (int i = 0; i < lastKeyFrame->mvpMapPoints.size(); i++)
        {
            //  map point existing && its observation time not too few
            if (lastKeyFrame->mvpMapPoints[i] &&
                lastKeyFrame->mvpMapPoints[i]->Observations() > obsPlus) //? if the program died here, try to change 1 to 0
            {
                keypoints.push_back(lastKeyFrame->mvKeysUn[i].pt); // LK-RGBD
                // keypoints.push_back(lastKeyFrame->mvKeys[i].pt); // LK-Stereo

                cv::Point3f pt3f;
                cv::Mat temp;

                // map point x,y,z in last keyframe
                temp = lastKeyFrame->mvpMapPoints[i]->GetWorldPos();
                pt3f.x = temp.at<float>(0);
                pt3f.y = temp.at<float>(1);
                pt3f.z = temp.at<float>(2);
                mappointInCurrentFrame.push_back(pt3f);
            }
        }
    }

    // LK tracking result
    vector<cv::Point2f> next_keypoints;

    // copy keypoints to prev_keypoints, prev_keypoints = keypoints
    prev_keypoints.clear();
    for (auto kp : keypoints)
    {
        prev_keypoints.push_back(kp);
    }

    vector<unsigned char> status; //判断该点是否跟踪失败
    vector<float> error;
    cv::Mat last_gray, gray; // LK光流法用于跟踪特征点的两帧

    //! BUG??
    //! why not check before?
    //! CV_BGR2GRAY or RGB?
    bool mbRGB = true; // TODO
    // cv::imshow("last_color", last_color);
    // cv::imshow("color", color);
    // cvtColor(last_color, last_gray, CV_BGR2GRAY);
    if (last_color.channels() == 1)
    {
        last_gray = last_color;
    }
    else if (last_color.channels() == 3)
    {
        if (mbRGB)
            cvtColor(last_color, last_gray, CV_RGB2GRAY);
        else
            cvtColor(last_color, last_gray, CV_BGR2GRAY);
    }
    else if (last_color.channels() == 4)
    {
        if (mbRGB)
            cvtColor(last_color, last_gray, CV_RGBA2GRAY);
        else
            cvtColor(last_color, last_gray, CV_BGRA2GRAY);
    }

    // cvtColor(color, gray, CV_BGR2GRAY);
    if (last_color.channels() == 1)
    {
        gray = color;
    }
    if (color.channels() == 3)
    {
        if (mbRGB)
            cvtColor(color, gray, CV_RGB2GRAY);
        else
            cvtColor(color, gray, CV_BGR2GRAY);
    }
    else if (color.channels() == 4)
    {
        if (mbRGB)
            cvtColor(color, gray, CV_RGBA2GRAY);
        else
            cvtColor(color, gray, CV_BGRA2GRAY);
    }

    //计算光流
    // cout << "       calling calcOpticalFlowPyrLK" << endl;
    // cv::imshow("last_gray", last_gray);
    // cv::imshow("gray", gray);
    // cout << "       prev_keypoints NUM " << prev_keypoints.size() << endl;
    // cout << "       next_keypoints NUM " << next_keypoints.size() << endl;
    /* OpenCV光流法函数解析
     * calcOpticalFlowPyrLK(
     *              匹配图1，匹配图2，
     *              图1关键点，图2关键点存放容器（可以有初值），
     *              匹配情况，匹配误差，
     *              金字塔窗口大小，金字塔层数，
     *              终止条件（迭代次数+最小步长），
     *              使用图2关键点存放容器中的初值)
     */
    /*
    cv::calcOpticalFlowPyrLK(
                current_frame_->left_img_, current_frame_->right_img_,
                kps_left, kps_right,
                status, error,
                cv::Size(11, 11), 3,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                cv::OPTFLOW_USE_INITIAL_FLOW);
    */
    cv::calcOpticalFlowPyrLK(
        last_gray,      // [in] last frame image (converted to gray)
        gray,           // [in] current frame image (converted to gray)
        prev_keypoints, // [in]
        next_keypoints, // [in as init & out], no init guess here
        status, error);
    // cout << "       calcOpticalFlowPyrLK finished" << endl;

    // keypoints = LK tracking successful result (remove lost)
    int i = 0;
    for (auto iter = keypoints.begin(); iter != keypoints.end(); i++)
    {
        // fail to track by LK
        if (status[i] == 0)
        {
            // remove lost
            iter = keypoints.erase(iter);
            continue;
        }

        // edit keypoints' coordinate with only LK successful result
        *iter = next_keypoints[i];
        iter++;
    }

    // cout << "tracked keypoints: " << keypoints.size() << endl;

    // erase the matched mappoint if the keypoint (LK tracking failed) was erased
    //? keypoint and its mappoints share the same order, so can use same index // keypoint-mappoint pair
    i = 0;
    for (auto iter = mappointInCurrentFrame.begin(); iter != mappointInCurrentFrame.end(); i++)
    {
        // fail to track by LK
        if (status[i] == 0)
        {
            iter = mappointInCurrentFrame.erase(iter);
            continue;
        }
        iter++;
    }

    /**使用PnP Ransac计算位姿*/
    vector<cv::Mat> point3D;
    cv::Mat R_vector, R, T;
    vector<int> ransacInlier;

    // all this three container are required NON-empty
    if (!(mappointInCurrentFrame.empty() || keypoints.empty() || DistCoef.empty()))
    {
        // cout << "   the number of mappint in current frame " << mappointInCurrentFrame.size() << endl;

        if (keypoints.size() < 20)
        {
            cout << "Optical flow need more points" << endl;
            return cv::Mat();
        }

        //  Finds an object pose from 3D-2D point correspondences using the RANSAC scheme
        //  https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
        // cout << "       calling solvePnPRansac" << endl;
        cv::solvePnPRansac(
            mappointInCurrentFrame, // [in] objectPoints 3d mappoints in current frame
            keypoints,              // [in] imagePoints 2d keypoints in current frame
            K, DistCoef,
            R_vector, T, // [out] pose estimation result
            false,       // [in] no init pose guess
            50, 3, 0.98,
            ransacInlier, // [out] Output vector that contains indices of inliers in objectPoints and imagePoints
            cv::SOLVEPNP_ITERATIVE);
        // cout << "       solvePnPRansac finished" << endl;

        // Converts a rotation matrix to a rotation vector or vice versa.
        cv::Rodrigues(R_vector, R);
        // SE(3) [R,t; 0,1]
        cv::Mat_<double> Rt = (cv::Mat_<double>(4, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), T.at<double>(0),
                               R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), T.at<double>(1),
                               R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), T.at<double>(2),
                               0, 0, 0, 1);
        cv::Mat Rt_float;
        Rt.convertTo(Rt_float, CV_32FC1);

        // TODO consider move forward right after PNP // init version in the buttom
        // [out to Tracking.cc] PNP inlier
        mnMatchesInliers = ransacInlier.size();
        // cout << "PNP inlier: " << ransacInlier.size() << endl;

        // [out to Tracking.cc] current frame pose estimation result
        Tcw = Rt_float;
    }

    if (keypoints.size() == 0)
    {
        cout << "LK -- all keypoints are lost." << endl;
        return cv::Mat();
    }

    /** 画出 keypoints*/
    cv::Mat img_show = color.clone();
    if (img_show.channels() < 3)
        cvtColor(img_show, img_show, CV_GRAY2BGR);

    int point = 0;
    for (auto kp : keypoints)
    {
        for (int inlierIndex = 0; inlierIndex < ransacInlier.size(); inlierIndex++)
        {
            // keypoints matched with its inlier, then highlight them in GUI
            if (point == ransacInlier[inlierIndex])
            {
                cv::circle(img_show, kp, 5, cv::Scalar(0, 255, 0), 1);
                cv::circle(img_show, kp, 1, cv::Scalar(0, 255, 0), -1);
            }
        }
        point++;
    }

    // save last frame RGB image
    last_color = color;

    return img_show; //返回值是rgb图片，用于显示光流跟踪到的特征点
}

#endif // ORB_SLAM2_LK_H
