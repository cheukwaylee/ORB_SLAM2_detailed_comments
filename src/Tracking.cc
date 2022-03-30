/**
 * @file Tracking.cc
 * @author guoqing (1337841346@qq.com)
 * @brief 追踪线程
 * @version 0.1
 * @date 2019-02-21
 *
 * @copyright Copyright (c) 2019
 *
 */

/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
 * of Zaragoza) For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Tracking.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Converter.h"
#include "FrameDrawer.h"
#include "Initializer.h"
#include "Map.h"
#include "ORBmatcher.h"

#include "Optimizer.h"
#include "PnPsolver.h"

#include <cmath>
#include <iostream>
#include <mutex>

#include "LK.h" // add LK-RGBD-Stereo

using namespace std;

// 程序中变量名的第一个字母如果为"m"则表示为类中的成员变量，member
// 第一个、第二个字母:
// "p"表示指针数据类型
// "n"表示int类型
// "b"表示bool类型
// "s"表示set类型
// "v"表示vector数据类型
// 'l'表示list数据类型
// "KF"表示KeyFrame数据类型

namespace ORB_SLAM2
{

  ///构造函数
  Tracking::Tracking(
      System *pSys,                             //系统实例
      ORBVocabulary *pVoc,                      // BOW字典
      FrameDrawer *pFrameDrawer,                //帧绘制器
      MapDrawer *pMapDrawer,                    //地图点绘制器
      Map *pMap,                                //地图句柄
      KeyFrameDatabase *pKFDB,                  //关键帧产生的词袋数据库
      const string &strSettingPath,             //配置文件路径
      const int sensor)                         //传感器类型
      : mState(NO_IMAGES_YET),                  //当前系统还没有准备好
        mSensor(sensor), mbOnlyTracking(false), //处于SLAM模式
        mbVO(false),                            //当处于纯跟踪模式的时候，这个变量表示了当前跟踪状态的好坏
        mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
        mpInitializer(static_cast<Initializer *>(NULL)), //暂时给地图初始化器设置为空指针
        mpSystem(pSys),
        mpViewer(NULL), //注意可视化的查看器是可选的，因为ORB-SLAM2最后是被编译成为一个库，所以对方人拿过来用的时候也应该有权力说我不要可视化界面（何况可视化界面也要占用不少的CPU资源）
        mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap),
        mnLastRelocFrameId(0) //恢复为0,没有进行这个过程的时候的默认值
  {
    // Load camera parameters from settings file
    // Step 1 从配置文件中加载相机参数
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    //? 参数从ymal读入，而const变量要在声明的时候指定具体数值（所以不能const???）
    const float fx = fSettings["Camera.fx"];
    const float fy = fSettings["Camera.fy"];
    const float cx = fSettings["Camera.cx"];
    const float cy = fSettings["Camera.cy"];

    //     |fx  0   cx|
    // K =  |0   fy  cy|
    //     |0   0   1|
    //构造相机内参矩阵
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK); // mK = K 赋值给成员变量mK

    // 图像矫正系数
    // [k1 k2 p1 p2 k3]
    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    //有些相机的畸变系数中会没有k3项
    if (k3 != 0)
    {
      DistCoef.resize(5);
      DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef); // mDistCoef = DistCoef 赋值给成员变量mDistCoef

    // 双目摄像头baseline * fx 50
    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if (fps == 0)
      fps = 30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    //输出
    cout << endl
         << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if (DistCoef.rows == 5)
      cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    // 1:RGB 0:BGR
    const int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if (mbRGB)
      cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
      cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    // Step 2 加载ORB特征点有关的参数,并新建特征点提取器
    //? 为什么不能const？
    // 每一帧提取的特征点数 1000
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    // 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // 尺度金字塔的层数 8
    int nLevels = fSettings["ORBextractor.nLevels"];
    // 提取fast特征点的默认阈值 20
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    // tracking过程都会用到mpORBextractorLeft作为特征点提取器
    mpORBextractorLeft =
        new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    // 如果是双目，tracking过程中还会用用到mpORBextractorRight作为右目特征点提取器
    if (sensor == System::STEREO)
      mpORBextractorRight =
          new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    // 在单目初始化的时候，会用mpIniORBextractor来作为特征点提取器 // 两倍特征点数
    if (sensor == System::MONOCULAR)
      mpIniORBextractor =
          new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    cout << endl
         << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if (sensor == System::STEREO || sensor == System::RGBD)
    {
      // 判断一个3D点远/近的阈值 mbf * (35~40) / fx
      // ThDepth其实就是表示基线长度的多少倍
      mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
      cout << endl
           << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if (sensor == System::RGBD)
    {
      // 深度相机disparity转化为depth时的因子
      mDepthMapFactor = fSettings["DepthMapFactor"];
      //? 原理是啥？
      if (fabs(mDepthMapFactor) < 1e-5)
        mDepthMapFactor = 1;
      else
        mDepthMapFactor = 1.0f / mDepthMapFactor;
    }
  }

  //设置局部建图器
  void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
  {
    mpLocalMapper = pLocalMapper;
  }

  //设置回环检测器
  void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
  {
    mpLoopClosing = pLoopClosing;
  }

  //设置可视化查看器
  void Tracking::SetViewer(Viewer *pViewer)
  {
    mpViewer = pViewer;
  }

  // 输入左右目图像，可以为RGB、BGR、RGBA、GRAY
  // 1、将图像转为mImGray和imGrayRight 并初始化mCurrentFrame
  // 2、进行tracking过程
  // 输出世界坐标系到该帧相机坐标系的变换矩阵
  cv::Mat Tracking::GrabImageStereo(
      const cv::Mat &imRectLeft,  //左侧图像
      const cv::Mat &imRectRight, //右侧图像
      const double &timestamp)    //时间戳
  {
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    // Step 1 ：将RGB或RGBA图像转为灰度图像
    if (mImGray.channels() == 3)
    {
      if (mbRGB)
      {
        cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
      }
      else
      {
        cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
      }
    }
    // 这里考虑得十分周全,甚至连四通道的图像都考虑到了
    else if (mImGray.channels() == 4)
    {
      if (mbRGB)
      {
        cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
      }
      else
      {
        cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
      }
    }

    // Step 2 ：构造Frame对象
    // add LK-Stereo: depends on whether need new keyframe or not

    // Step 3 ：跟踪
    mNeedNewKF = NeedNewKeyFrame();
    // mNeedNewKF = true;
    if (!mNeedNewKF)
    {
#ifdef COMPILEDWITHC14
      std::chrono::steady_clock::time_point t1_LK = std::chrono::steady_clock::now();
#else
      std::chrono::monotonic_clock::time_point t1_LK = std::chrono::monotonic_clock::now();
#endif
      mCurrentFrame = Frame(true);
      cv::Mat Tcw;
      last_mnMatchesInliers = mnMatchesInliers;

      mLKimg = computeMtcwUseLK(
          mpLastKeyFrame,
          imRectRight,                                           // [in] current frame RGB imgage
          (mCurrentFrame.mnId - mpLastKeyFrame->mnFrameId) == 1, // [in] if last frame is keyframe
          mK, mDistCoef,
          Tcw,               // [out] current frame pose
          mnMatchesInliers); // [out] the number of current frame Matches Inliers
      // mLKimg = flow(mpLastKeyFrame, imRGB,  mCurrentFrame.mnId - mpLastKeyFrame->mnFrameId ==1, mK, mDistCoef, mTcw);

      mpFrameDrawer->mLK = mLKimg;
      mCurrentFrame.mTcw = Tcw;
#ifdef COMPILEDWITHC14
      std::chrono::steady_clock::time_point t2_LK = std::chrono::steady_clock::now();
#else
      std::chrono::monotonic_clock::time_point t2_LK = std::chrono::monotonic_clock::now();
#endif
      cout << "------ track LK optical flow  use time: "
           << chrono::duration_cast<std::chrono::duration<double>>(t2_LK - t1_LK).count()
           << " us" << endl;
      mpFrameDrawer->Update(this);

      // no need keyframe before, but LK tracking failed, so need keyframe!
      mNeedNewKF = NeedNewKeyFrame();
    }

    // same as the case without introducing LK
    if (mNeedNewKF)
    {
#ifdef COMPILEDWITHC14
      std::chrono::steady_clock::time_point t1_ORB = std::chrono::steady_clock::now();
#else
      std::chrono::monotonic_clock::time_point t1_ORB = std::chrono::monotonic_clock::now();
#endif
      mCurrentFrame = Frame(
          mImGray,             //左目图像
          imGrayRight,         //右目图像
          timestamp,           //时间戳
          mpORBextractorLeft,  //左目特征提取器
          mpORBextractorRight, //右目特征提取器
          mpORBVocabulary,     //字典
          mK,                  //内参矩阵
          mDistCoef,           //去畸变参数
          mbf,                 //基线长度
          mThDepth);           //远点,近点的区分阈值
      // cout << "track Feature: currentFrame ID: " << mCurrentFrame.mnId << endl;
      Track();
#ifdef COMPILEDWITHC14
      std::chrono::steady_clock::time_point t2_ORB = std::chrono::steady_clock::now();
#else
      std::chrono::monotonic_clock::time_point t2_ORB = std::chrono::monotonic_clock::now();
#endif
      cout << "****** KeyFrame track feature use time: "
           << chrono::duration_cast<std::chrono::duration<double>>(t2_ORB - t1_ORB).count()
           << " us" << endl;
    }

    // cout << "currentFrame and pose: " << mCurrentFrame.mnId << endl;
    // cout << mCurrentFrame.mTcw << endl;
    // cout << "orbslam mTcw:" << endl
    //      << mCurrentFrame.mTcw << endl;
    // count++;

    //返回位姿
    // cout << "current frame pose " << endl
    //      << mCurrentFrame.mTcw << endl;
    return mCurrentFrame.mTcw.clone();
  }

  // 输入左目RGB或RGBA图像和深度图
  // 1、将图像转为mImGray和imDepth并初始化mCurrentFrame
  // 2、进行tracking过程
  // 输出世界坐标系到该帧相机坐标系的变换矩阵
  cv::Mat Tracking::GrabImageRGBD(
      const cv::Mat &imRGB,    //彩色图像
      const cv::Mat &imD,      //深度图像
      const double &timestamp) //时间戳
  {
    mImGray = imRGB;
    mImDepth = imD; // add LK-RGBD //? for what???
    cv::Mat imDepth = imD;

    // step 1 ：将RGB或RGBA图像转为灰度图像
    if (mImGray.channels() == 3)
    {
      if (mbRGB)
        cvtColor(mImGray, mImGray, CV_RGB2GRAY);
      else
        cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
    else if (mImGray.channels() == 4)
    {
      if (mbRGB)
        cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
      else
        cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    // step 2 ：将深度相机的disparity转为Depth, 也就是转换成为真正尺度下的深度
    // 还原视差图的深度信息
    //这里的判断条件感觉有些尴尬，前者和后者满足一个就可以了
    //满足前者意味着, mDepthMapFactor 相对1来讲要足够大 //? 1e-5 足够大？？
    //满足后者意味着, 如果深度图像不是浮点型? 才会执行
    if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
      //将图像转换成为另外一种数据类型,具有可选的数据大小缩放系数
      imDepth.convertTo(
          imDepth,          //输出图像
          CV_32F,           //输出图像的数据类型
          mDepthMapFactor); //缩放系数

    // step 3 ：构造Frame
    // add LK-RGBD: depends on whether need new keyframe or not

    // step 4 ：跟踪
    mNeedNewKF = NeedNewKeyFrame();
    if (!mNeedNewKF)
    {
#ifdef COMPILEDWITHC14
      std::chrono::steady_clock::time_point t1_LK = std::chrono::steady_clock::now();
#else
      std::chrono::monotonic_clock::time_point t1_LK = std::chrono::monotonic_clock::now();
#endif
      mCurrentFrame = Frame(true);
      cv::Mat Tcw;
      last_mnMatchesInliers = mnMatchesInliers;

      mLKimg = computeMtcwUseLK(
          mpLastKeyFrame,
          imRGB,                                                 // [in] current frame RGB imgage
          (mCurrentFrame.mnId - mpLastKeyFrame->mnFrameId) == 1, // [in] if last frame is keyframe
          mK, mDistCoef,
          Tcw,               // [out] current frame pose
          mnMatchesInliers); // [out] the number of current frame Matches Inliers
      // mLKimg = flow(mpLastKeyFrame, imRGB,  mCurrentFrame.mnId - mpLastKeyFrame->mnFrameId ==1, mK, mDistCoef, mTcw);

      mpFrameDrawer->mLK = mLKimg;
      mCurrentFrame.mTcw = Tcw;
#ifdef COMPILEDWITHC14
      std::chrono::steady_clock::time_point t2_LK = std::chrono::steady_clock::now();
#else
      std::chrono::monotonic_clock::time_point t2_LK = std::chrono::monotonic_clock::now();
#endif
      cout << "------ track LK optical flow  use time: "
           << chrono::duration_cast<std::chrono::duration<double>>(t2_LK - t1_LK).count()
           << " us" << endl;
      mpFrameDrawer->Update(this);

      // no need keyframe before, but LK tracking failed, so need keyframe!
      mNeedNewKF = NeedNewKeyFrame();
    }

    // same as the case without introducing LK
    if (mNeedNewKF)
    {
#ifdef COMPILEDWITHC14
      std::chrono::steady_clock::time_point t1_ORB = std::chrono::steady_clock::now();
#else
      std::chrono::monotonic_clock::time_point t1_ORB = std::chrono::monotonic_clock::now();
#endif
      mCurrentFrame = Frame(
          mImGray,            //灰度图像
          imDepth,            //深度图像
          timestamp,          //时间戳
          mpORBextractorLeft, // ORB特征提取器
          mpORBVocabulary,    //词典
          mK,                 //相机内参矩阵
          mDistCoef,          //相机的去畸变参数
          mbf,                //相机基线*相机焦距
          mThDepth);          //内外点区分深度阈值
      // cout << "track Feature: currentFrame ID: " << mCurrentFrame.mnId << endl;
      Track();
#ifdef COMPILEDWITHC14
      std::chrono::steady_clock::time_point t2_ORB = std::chrono::steady_clock::now();
#else
      std::chrono::monotonic_clock::time_point t2_ORB = std::chrono::monotonic_clock::now();
#endif
      cout << "****** KeyFrame track feature use time: "
           << chrono::duration_cast<std::chrono::duration<double>>(t2_ORB - t1_ORB).count()
           << " us" << endl;
    }

    // cout << "currentFrame and pose: " << mCurrentFrame.mnId << endl;
    // cout << mCurrentFrame.mTcw << endl;
    // cout << "orbslam mTcw:" << endl
    //      << mCurrentFrame.mTcw << endl;
    // count++;

    //返回当前帧的位姿
    // cout << "current frame pose " << endl
    //      << mCurrentFrame.mTcw << endl;
    return mCurrentFrame.mTcw.clone();
  }

  /**
   * @brief
   * 输入左目RGB或RGBA图像，输出世界坐标系到该帧相机坐标系的变换矩阵
   *
   * @param[in] im 单目图像
   * @param[in] timestamp 时间戳
   * @return cv::Mat
   *
   * Step 1 ：将彩色图像转为灰度图像
   * Step 2 ：构造Frame
   * Step 3 ：跟踪
   */
  cv::Mat Tracking::GrabImageMonocular(
      const cv::Mat &im,
      const double &timestamp)
  {
    mImGray = im;

    // Step 1 ：将彩色图像转为灰度图像
    if (mImGray.channels() == 3)
    {
      if (mbRGB)
        cvtColor(mImGray, mImGray, CV_RGB2GRAY);
      else
        cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
    else if (mImGray.channels() == 4)
    {
      if (mbRGB)
        cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
      else
        cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    // Step 2 ：构造Frame
    //判断该帧是不是初始化
    if (mState == NOT_INITIALIZED ||
        mState == NO_IMAGES_YET) //没有成功初始化的前一个状态就是NO_IMAGES_YET
      mCurrentFrame = Frame(
          mImGray,
          timestamp,
          mpIniORBextractor, //初始化ORB特征点提取器会提取2倍的指定特征点数目
          mpORBVocabulary,
          mK,
          mDistCoef,
          mbf,
          mThDepth);
    else
      mCurrentFrame = Frame(
          mImGray,
          timestamp,
          mpORBextractorLeft, //正常运行的时的ORB特征点提取器，提取指定数目特征点
          mpORBVocabulary,
          mK,
          mDistCoef,
          mbf,
          mThDepth);

    // Step 3 ：跟踪
    Track();

    //返回当前帧的位姿
    return mCurrentFrame.mTcw.clone();
  }

  /**
   * @brief
   * Main tracking function. It is independent of the input sensor.
   *
   * track包含两部分：估计运动、跟踪局部地图
   *
   * Step 1：初始化
   * Step 2：跟踪
   * Step 3：记录位姿信息，用于轨迹复现
   */
  void Tracking::Track()
  {
    // mState为tracking的状态，包括 SYSTME_NOT_READY, NO_IMAGES_YET, NOT_INITIALIZED, OK, LOST
    // 如果图像复位过、或者第一次运行，则为 NO_IMAGES_YET状态
    if (mState == NO_IMAGES_YET) //? 什么逻辑？ 状态往前一步？
    {
      mState = NOT_INITIALIZED;
    }

    // mLastProcessedState 存储了Tracking最新的状态，用于FrameDrawer中的绘制
    mLastProcessedState = mState;

    // Get Map Mutex -> Map cannot be changed
    // 地图更新时加锁。保证地图不会发生变化
    //  Question: 这样子会不会影响地图的实时更新?
    //  Ansewer:  主要耗时在构造帧//?(构造帧是啥 关键帧吗？)//中特征点的提取和匹配部分,
    //        在那个时候地图是没有被上锁的,有足够的时间更新地图
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    // Step 1：地图初始化
    if (mState == NOT_INITIALIZED)
    {
      if (mSensor == System::STEREO || mSensor == System::RGBD)
        //双目RGBD相机的初始化共用一个函数
        StereoInitialization();
      else
        //单目初始化
        MonocularInitialization();

      //更新帧绘制器中存储的最新状态
      mpFrameDrawer->Update(this);

      //这个状态量被上面的初始化函数中被更新
      if (mState != OK) //? 意味着初始化失败？
        return;
    }

    // System is initialized. Track Frame.
    // 不需要初始化 直接跟踪当前帧就行
    else
    {
      // bOK为临时变量，用于表示每个函数是否执行成功
      bool bOK;

      // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
      // mbOnlyTracking等于false表示正常SLAM模式（定位+地图更新），mbOnlyTracking等于true表示仅定位模式
      // tracking类构造时默认为false
      // 在viewer中有个开关ActivateLocalizationMode，可以控制是否开启mbOnlyTracking
      if (!mbOnlyTracking)
      {
        // Local Mapping is activated. This is the normal behaviour, unless
        // you explicitly activate the "only tracking" mode.
        // Step 2：跟踪进入正常SLAM模式，有地图更新
        // 是否正常跟踪
        if (mState == OK)
        {
          // Local Mapping might have changed some MapPoints tracked in last frame
          // Step 2.1 检查并更新上一帧被替换的MapPoints
          // 局部建图线程则可能会对原有的地图点进行替换.在这里进行检查
          CheckReplacedInLastFrame();

          // Step 2.2
          // 运动模型是空的 或 刚完成重定位： 跟踪参考关键帧；否则恒速模型跟踪
          // 第一个条件,如果运动模型为空,说明是刚初始化开始，或者已经跟丢了
          // 第二个条件,如果当前帧紧紧地跟着在重定位的帧的后面，我们将重定位帧来恢复位姿
          //       mnLastRelocFrameId 上一次重定位的那一帧
          if (mVelocity.empty() || ((mCurrentFrame.mnId - mnLastRelocFrameId) < 2))
          {
            // 用最近的关键帧来跟踪当前的普通帧
            //    通过BoW的方式在参考帧中找当前帧特征点的匹配点
            //    优化每个特征点都对应3D点重投影误差即可得到位姿
            bOK = TrackReferenceKeyFrame();
          }
          else
          {
            // 用最近的普通帧来跟踪当前的普通帧
            //    根据恒速模型设定当前帧的初始位姿
            //    通过投影的方式在参考帧中找当前帧特征点的匹配点
            //    优化每个特征点所对应3D点的投影误差即可得到位姿
            bOK = TrackWithMotionModel();
            if (!bOK)
              //根据恒速模型失败了，只能根据参考关键帧来跟踪
              bOK = TrackReferenceKeyFrame();
          }
        }
        else
        {
          // 如果跟踪状态不成功,那么就只能重定位了
          //    BOW搜索，EPnP求解位姿
          bOK = Relocalization();
        }
      }
      else
      // Step 2：只进行跟踪tracking，局部地图不工作
      // Localization Mode (mbOnlyTracking == true): Local Mapping is deactivated
      {
        // Step 2.1 如果跟丢了，只能重定位
        if (mState == LOST)
        {
          bOK = Relocalization();
        }
        else
        {
          // mbVO是mbOnlyTracking为true时的才有的一个变量
          // mbVO为false表示此帧匹配了很多的MapPoints，跟踪很正常 (注意有点反直觉)
          // mbVO为true表明此帧匹配了很少的MapPoints，少于10个，要跪的节奏
          if (!mbVO)
          {
            // Step 2.2.1 如果跟踪正常，使用恒速模型 或 参考关键帧跟踪
            // ?In last frame we tracked enough MapPoints in the map
            if (!mVelocity.empty())
            {
              bOK = TrackWithMotionModel();
              // ? 为了和前面模式统一，这个地方是不是应该加上
              // if(!bOK)
              //    bOK = TrackReferenceKeyFrame();
            }
            else
            {
              // 如果恒速模型不被满足,那么就只能够通过参考关键帧来定位
              bOK = TrackReferenceKeyFrame();
            }
          }
          else
          {
            // Step 2.2.2 跟踪不正常，mbVO为true，既做跟踪又做重定位
            //        表明此帧匹配了很少（小于10）的地图点，要跪的节奏，
            // In last frame we tracked mainly "visual odometry" points.
            // We compute two camera poses, one from motion model and one doing relocalization.
            // If relocalization is successful we choose that solution,
            // otherwise we retain the "visual odometry" solution.

            // MM=Motion Model
            //通过运动模型进行跟踪的结果
            bool bOKMM = false;
            //通过重定位方法来跟踪的结果
            bool bOKReloc = false;

            //运动模型中构造的地图点
            vector<MapPoint *> vpMPsMM;
            //在追踪运动模型后发现的外点
            vector<bool> vbOutMM;
            //运动模型得到的位姿
            cv::Mat TcwMM;

            // Step 2.3 当运动模型有效的时候,根据运动模型计算位姿
            if (!mVelocity.empty())
            {
              bOKMM = TrackWithMotionModel();

              // 将恒速模型跟踪结果暂存到这几个变量中，因为后面重定位会改变这些变量
              vpMPsMM = mCurrentFrame.mvpMapPoints;
              vbOutMM = mCurrentFrame.mvbOutlier;
              TcwMM = mCurrentFrame.mTcw.clone();
            }

            // Step 2.4 使用重定位的方法来得到当前帧的位姿
            bOKReloc = Relocalization();

            // Step 2.5 根据前面的恒速模型、重定位结果来更新状态
            if (bOKMM && !bOKReloc)
            {
              // 恒速模型成功、重定位失败，重新使用之前暂存的恒速模型结果
              mCurrentFrame.SetPose(TcwMM);
              mCurrentFrame.mvpMapPoints = vpMPsMM;
              mCurrentFrame.mvbOutlier = vbOutMM;

              //? 疑似bug！这段代码是不是重复增加了观测次数？后面 TrackLocalMap ????
              //函数中会有这些操作
              // 如果当前帧匹配的3D点很少，增加当前可视地图点的被观测次数

              if (mbVO)
              // ?是这个意思吗？ 经过恒速跟踪模型+重定位，跟踪还是不正常，也就是mbVO还是true
              // ? 为什么都还是不正常还要增加地图的观测次数？
              {
                // 更新当前帧的地图点被观测次数
                for (int i = 0; i < mCurrentFrame.N; i++) // 遍历当前帧的keypoint
                {
                  //如果这个特征点形成了地图点,并且也不是外点的时候
                  if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                  {
                    //增加能观测到该地图点的帧数
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                  }
                }
              }
            }
            else if (bOKReloc)
            {
              // 只要重定位成功整个跟踪过程正常进行（重定位与跟踪，更相信重定位）
              mbVO = false;
            }
            //有一个成功我们就认为执行成功了
            bOK = bOKReloc || bOKMM;
          }
        }
      }

      // 将最新的关键帧作为当前帧的参考关键帧
      mCurrentFrame.mpReferenceKF = mpReferenceKF;

      // If we have an initial estimation of the camera pose and matching. Track the local map.
      // Step 3：在跟踪得到当前帧初始姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
      // 竹曼理解：前面是粗略的定位，这里是精匹配
      // 前面只是跟踪一帧得到初始位姿，这里搜索局部关键帧、局部地图点，和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
      if (!mbOnlyTracking)
      // 正常SLAM模式，有地图更新
      {
        if (bOK) // 恒速模型 or 重定位 其中有一个成功
          bOK = TrackLocalMap();
      }
      else
      // 追踪模式
      {
        // mbVO true means that there are few matches to MapPoints in the map.
        // We cannot retrieve(取回) a local map and therefore we do not perform TrackLocalMap().
        // Once the system relocalizes the camera we will use the local map again.

        // 重定位成功
        if (bOK && !mbVO)
          bOK = TrackLocalMap();
      }

      //根据上面的操作来判断是否追踪成功
      if (bOK)
        mState = OK;
      else
        mState = LOST;

      // Step 4：更新显示线程中的图像、特征点、地图点等信息
      mpFrameDrawer->Update(this);
      mpFrameDrawer->mLK = mLKimg; // add LK-RGBD-Stereo

      // If tracking were good, check if we insert a keyframe
      //只有在成功追踪时才考虑生成关键帧的问题
      if (bOK)
      {
        // Update motion model
        // Step 5：跟踪成功，更新恒速运动模型
        if (!mLastFrame.mTcw.empty())
        // 上一帧已经存在定位，恒速模型才有意义
        {
          // 更新恒速运动模型 TrackWithMotionModel 中的mVelocity
          cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);                                  // (上一帧 wrt world)^-1 的SE(3)
          mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3)); // R
          mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));            // T
          // mVelocity = Tcl = Tcw * Twl,表示上一帧到当前帧的变换，
          // 其中 Twl = LastTwc
          // (当前帧 wrt 上一帧) = (当前帧 wrt world) * (world wrt 上一帧)
          // 这个结果在匀速模型里面用来作为下一帧的初值
          mVelocity = mCurrentFrame.mTcw * LastTwc;
        }
        else
        // 上一帧不存在位姿信息，无法套用恒速模型
        {
          //否则速度为空
          mVelocity = cv::Mat();
        }

        //更新显示中的位姿
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        // Clean VO matches
        // Step 6：清除观测不到的地图点
        for (int i = 0; i < mCurrentFrame.N; i++) // 遍历当前帧的特征点
        {
          MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
          // 遍历到的特征点有对应的地图点
          if (pMP)
            if (pMP->Observations() < 1) // 当前地图点的被观测次数 < 1
            {
              mCurrentFrame.mvbOutlier[i] = false;
              mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }
        }

        // Delete temporal MapPoints
        // Step 7：清除恒速模型跟踪中 UpdateLastFrame中为当前帧临时添加的MapPoints（仅双目和rgbd）
        // 注意：步骤6中只是在当前帧中将这些MapPoints剔除
        // 现在要！从MapPoints的数据库中删除
        // 临时地图点仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
        for (list<MapPoint *>::iterator
                 lit = mlpTemporalPoints.begin(),
                 lend = mlpTemporalPoints.end();
             lit != lend; lit++)
        {
          MapPoint *pMP = *lit;
          delete pMP; // 释放
        }

        // 这里不仅仅是清除mlpTemporalPoints，通过delete pMP还删除了指针指向的MapPoint
        // 不能够直接执行这个是因为其中存储的都是指针（要释放容器里面的地址）,上一步的操作是为了避免内存泄露
        mlpTemporalPoints.clear(); // clear the whole list

        // Check if we need to insert a new keyframe
        // Step 8：检测并插入关键帧，对于双目或RGB-D会产生新的地图点
        // 若跟踪成功,根据条件判定是否产生关键帧
        if (NeedNewKeyFrame())
          CreateNewKeyFrame();

        // We allow points with high innovation (considered outliers by the Huber
        // Function) pass to the new keyframe, so that bundle adjustment will
        // finally decide if they are outliers or not. We don't want next frame to
        // estimate its position with those points so we discard them in the frame.
        // 作者说 允许在BA中被Huber核判断为外点的传入新的关键帧中，让后续的BA来审判他们是不是真正的外点
        // 但是估计下一帧位姿的时候我们不想用这些外点，所以删掉
        //  Step 9 删除那些在bundle adjustment中检测为outlier的地图点
        for (int i = 0; i < mCurrentFrame.N; i++) // 遍历特征点
        {
          // 这里第一个条件还要执行判断 因为前面的操作中可能删除了其中的地图点
          if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i]) // 非空 且 为外点
            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        }
      }

      // Reset if the camera get lost soon after initialization
      // Step 10 如果初始化后不久就跟踪失败，并且relocation也没有搞定，只能重新Reset
      if (mState == LOST)
      {
        //如果地图中的关键帧信息过少的话,直接重新进行初始化了
        if (mpMap->KeyFramesInMap() <= 5)
        {
          cout << "Track lost soon after initialisation, reseting..." << endl;
          mpSystem->Reset();
          return;
        }
      }

      //确保已经设置了参考关键帧
      if (!mCurrentFrame.mpReferenceKF)              // 如果没有
        mCurrentFrame.mpReferenceKF = mpReferenceKF; // 设置

      // 保存上一帧的数据,当前帧变上一帧
      mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    // Step 11：记录位姿信息，用于最后保存所有的轨迹
    if (!mCurrentFrame.mTcw.empty())
    // 当前帧有位姿信息
    {
      // 计算相对姿态 Tcr = Tcw * Twr, Twr = Trw^-1
      // (当前帧 wrt 相对帧) = (当前帧 wrt world) * (world wrt 相对帧)，
      //        相对帧就是当前帧用来参考的关键帧
      cv::Mat Tcr =
          mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
      //保存各种状态
      mlRelativeFramePoses.push_back(Tcr);
      mlpReferences.push_back(mpReferenceKF);
      mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
      mlbLost.push_back(mState == LOST);
    }
    else
    // 当前帧没有位姿信息
    {
      // This can happen if tracking is lost
      // 如果跟踪失败，则相对位姿使用上一次值
      mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
      mlpReferences.push_back(mlpReferences.back());
      mlFrameTimes.push_back(mlFrameTimes.back());
      mlbLost.push_back(mState == LOST);
    }

  } // end Tracking

  /*
   * @brief 双目和rgbd的地图初始化，比单目简单很多
   *
   * 由于具有深度信息，直接生成MapPoints
   */
  void Tracking::StereoInitialization()
  {
    // 初始化要求当前帧的特征点超过500
    if (mCurrentFrame.N > 500)
    {
      // Set Frame pose to the origin
      // 设定初始位姿为单位，无旋转平移
      mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

      // Create initial KeyFrame
      // 将当前帧构造为初始关键帧
      // mCurrentFrame的数据类型为Frame
      // KeyFrame包含Frame、地图3D点、以及BoW
      // KeyFrame里有一个mpMap，Tracking里有一个mpMap，而KeyFrame里的mpMap都指向Tracking里的这个mpMap
      // KeyFrame里有一个mpKeyFrameDB，Tracking里有一个mpKeyFrameDB，而KeyFrame里的mpKeyFrameDB都指向Tracking里的这个mpKeyFrameDB
      //    提问: 为什么要指向Tracking中的相应的变量呢?
      //    ANSWER: 因为Tracking是主线程，是它创建和加载的这些模块
      KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

      // 地图中也包含了KeyFrame，反过来KeyFrame中包含了地图点，相互包含
      // Insert KeyFrame in the map
      // 在地图中添加该初始关键帧
      mpMap->AddKeyFrame(pKFini);

      // Create MapPoints and associate to KeyFrame
      // 为每个特征点构造MapPoint
      for (int i = 0; i < mCurrentFrame.N; i++)
      {
        float z = mCurrentFrame.mvDepth[i]; // 当前帧中当前遍历到的地图点的深度
        //只有具有正深度的点才会被构造地图点
        if (z > 0)
        {
          // 通过反投影得到该特征点的世界坐标系下3D坐标
          cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);

          // 将3D点构造为MapPoint
          MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);

          // 为该MapPoint添加属性：
          // a.观测到该MapPoint的关键帧
          // b.该MapPoint的描述子
          // c.该MapPoint的平均观测方向和深度范围
          //
          // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
          pNewMP->AddObservation(pKFini, i);
          // b.从众多观测到该MapPoint的特征点中挑选区分度最高的描述子
          pNewMP->ComputeDistinctiveDescriptors();
          // c.更新该MapPoint平均观测方向以及观测距离的范围
          pNewMP->UpdateNormalAndDepth();

          // 在地图中添加该MapPoint
          mpMap->AddMapPoint(pNewMP);
          // 表示该KeyFrame的哪个特征点可以观测到哪个3D点
          pKFini->AddMapPoint(pNewMP, i);

          // 将该MapPoint添加到当前帧的mvpMapPoints中
          // 为当前Frame的特征点与MapPoint之间建立索引
          mCurrentFrame.mvpMapPoints[i] = pNewMP;
        }
      }

      cout << "New map created with "
           << mpMap->MapPointsInMap()
           << " points" << endl;

      // 在局部地图中添加该初始关键帧
      mpLocalMapper->InsertKeyFrame(pKFini);

      // 更新当前帧为上一帧
      mLastFrame = Frame(mCurrentFrame);
      mnLastKeyFrameId = mCurrentFrame.mnId;
      mpLastKeyFrame = pKFini;

      mvpLocalKeyFrames.push_back(pKFini);
      //? 这个局部地图点竟然..不在mpLocalMapper中管理?
      // 我现在的想法是，这个点只是暂时被保存在了 Tracking 线程中，所以称之为 local
      // 初始化之后，通过双目图像生成的地图点，都应该被认为是局部地图点
      mvpLocalMapPoints = mpMap->GetAllMapPoints();
      mpReferenceKF = pKFini;
      mCurrentFrame.mpReferenceKF = pKFini;

      // 把当前（最新的）局部MapPoints作为ReferenceMapPoints
      // ReferenceMapPoints是DrawMapPoints函数画图的时候用的
      mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
      mpMap->mvpKeyFrameOrigins.push_back(pKFini); //? 保存了最初始的关键帧 mvpKeyFrameOrigins的长度只有1吗？
      mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

      //追踪成功
      mState = OK;
    }
  }

  /*
   * @brief 单目的地图初始化
   *
   * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
   * 得到初始两帧的匹配、相对运动、初始MapPoints
   *
   * Step 1：（未创建）得到用于初始化的第一帧，初始化需要两帧
   * Step 2：（已创建）如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
   * Step 3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
   * Step 4：如果初始化的两帧之间的匹配点太少，重新初始化
   * Step 5：通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
   * Step 6：删除那些无法进行三角化的匹配点
   * Step 7：将三角化得到的3D点包装成MapPoints
   */
  void Tracking::MonocularInitialization()
  {
    // Step 1 如果单目初始器还没有被创建，则创建。后面如果重新初始化时会清掉这个
    if (!mpInitializer)
    {
      // Set Reference Frame
      // 单目初始帧的特征点数必须大于100
      if (mCurrentFrame.mvKeys.size() > 100)
      {
        // 初始化需要两帧，分别是mInitialFrame，mCurrentFrame
        mInitialFrame = Frame(mCurrentFrame);
        // 用当前帧更新上一帧
        mLastFrame = Frame(mCurrentFrame);
        // mvbPrevMatched  记录"上一帧"所有特征点
        mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
        for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
          mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

        // 删除前判断一下，来避免出现段错误。不过在这里是多余的判断
        // 不过在这里是多余的判断，因为前面已经判断过了
        if (mpInitializer)
          delete mpInitializer;

        // 由当前帧构造初始器 sigma:1.0 iterations:200
        mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

        // 初始化为-1 表示没有任何匹配。这里面存储的是匹配的点的id
        fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

        return;
      }
    }
    else //如果单目初始化器已经被创建
    {
      // Try to initialize
      // Step 2 如果当前帧特征点数太少（不超过100），则重新构造初始器
      // NOTICE 只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
      if ((int)mCurrentFrame.mvKeys.size() <= 100)
      {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer *>(NULL);
        fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
        return;
      }

      // Find correspondences
      // Step 3 在mInitialFrame与mCurrentFrame中找匹配的特征点对
      ORBmatcher matcher(
          0.9,   //最佳的和次佳特征点评分的比值阈值，这里是比较宽松的，跟踪时一般是0.7
          true); //检查特征点的方向

      // 对 mInitialFrame,mCurrentFrame 进行特征点匹配
      // mvbPrevMatched为参考帧的特征点坐标，初始化存储的是mInitialFrame中特征点坐标，匹配后存储的是匹配好的当前帧的特征点坐标
      // mvIniMatches
      // 保存参考帧F1中特征点是否匹配上，index保存是F1对应特征点索引，值保存的是匹配好的F2特征点索引
      int nmatches = matcher.SearchForInitialization(
          mInitialFrame, mCurrentFrame, //初始化时的参考帧和当前帧
          mvbPrevMatched,               //在初始化参考帧中提取得到的特征点
          mvIniMatches,                 //保存匹配关系
          100);                         //搜索窗口大小

      // Check if there are enough correspondences
      // Step 4 验证匹配结果，如果初始化的两帧之间的匹配点太少，重新初始化
      if (nmatches < 100)
      {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer *>(NULL);
        return;
      }

      cv::Mat Rcw;                 // Current Camera Rotation
      cv::Mat tcw;                 // Current Camera Translation
      vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

      // Step 5 通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
      if (mpInitializer->Initialize(
              mCurrentFrame,   //当前帧
              mvIniMatches,    //当前帧和参考帧的特征点的匹配关系
              Rcw, tcw,        //初始化得到的相机的位姿
              mvIniP3D,        //进行三角化得到的空间点集合
              vbTriangulated)) //以及对应于mvIniMatches来讲,其中哪些点被三角化了
      {
        // Step 6 初始化成功后，删除那些无法进行三角化的匹配点
        for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
        {
          if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
          {
            mvIniMatches[i] = -1;
            nmatches--;
          }
        }

        // Set Frame Poses
        // Step 7 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
        mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
        // 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到相机坐标系的变换矩阵
        cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
        Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
        tcw.copyTo(Tcw.rowRange(0, 3).col(3));
        mCurrentFrame.SetPose(Tcw);

        // Step 8 创建初始化地图点MapPoints
        // Initialize函数会得到mvIniP3D，
        // mvIniP3D是cv::Point3f类型的一个容器，是个存放3D点的临时变量，
        // CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中
        CreateInitialMapMonocular();
      } //当初始化成功的时候进行
    }   //如果单目初始化器已经被创建
  }

  /**
   * @brief 单目相机成功初始化后用三角化得到的点生成MapPoints
   *
   */
  void Tracking::CreateInitialMapMonocular()
  {
    // Create KeyFrames 认为单目初始化时候的参考帧和当前帧都是关键帧
    KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB); // 第一帧
    KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB); // 第二帧

    // Step 1 将初始关键帧,当前关键帧的描述子转为BoW
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    // Step 2 将关键帧插入到地图
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    // Step 3 用初始化得到的3D点来生成地图点MapPoints
    //  mvIniMatches[i] 表示初始化两帧特征点匹配关系。
    //  具体解释：i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值,没有匹配关系的话，vMatches12[i]值为
    //  -1
    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
      // 没有匹配，跳过
      if (mvIniMatches[i] < 0)
        continue;

      // Create MapPoint.
      //  用三角化点初始化为空间点的世界坐标
      cv::Mat worldPos(mvIniP3D[i]);

      // Step 3.1 用3D点构造MapPoint
      MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

      // Step 3.2 为该MapPoint添加属性：
      // a.观测到该MapPoint的关键帧
      // b.该MapPoint的描述子
      // c.该MapPoint的平均观测方向和深度范围

      // 表示该KeyFrame的2D特征点和对应的3D地图点
      pKFini->AddMapPoint(pMP, i);
      pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

      // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
      pMP->AddObservation(pKFini, i);
      pMP->AddObservation(pKFcur, mvIniMatches[i]);

      // b.从众多观测到该MapPoint的特征点中挑选最有代表性的描述子
      pMP->ComputeDistinctiveDescriptors();
      // c.更新该MapPoint平均观测方向以及观测距离的范围
      pMP->UpdateNormalAndDepth();

      // Fill Current Frame structure
      // mvIniMatches下标i表示在初始化参考帧中的特征点的序号
      // mvIniMatches[i]是初始化当前帧中的特征点的序号
      mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
      mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

      // Add to Map
      mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    // Step 3.3 更新关键帧间的连接关系
    // 在3D点和关键帧之间建立边，每个边有一个权重，边的权重是该关键帧与当前帧公共3D点的个数
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points"
         << endl;

    // Step 4 全局BA优化，同时优化所有位姿和三维点
    Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

    // Set median depth to 1
    // Step 5 取场景的中值深度，用于尺度归一化
    // 为什么是 pKFini 而不是 pKCur ? 答：都可以的，内部做了位姿变换了
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    //两个条件,一个是平均深度要大于0,另外一个是在当前帧中被观测到的地图点的数目应该大于100
    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
    {
      cout << "Wrong initialization, reseting..." << endl;
      Reset();
      return;
    }

    // Step 6 将两帧之间的变换归一化到平均深度1的尺度下
    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    // x/z y/z 将z归一化到1
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    // Step 7 把3D点的尺度也归一化到1
    // 为什么是pKFini? 是不是就算是使用 pKFcur 得到的结果也是相同的?
    // 答：是的，因为是同样的三维点
    vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
    {
      if (vpAllMapPoints[iMP])
      {
        MapPoint *pMP = vpAllMapPoints[iMP];
        pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
      }
    }

    //  Step 8 将关键帧插入局部地图，更新归一化后的位姿、局部地图点
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    // 单目初始化之后，得到的初始地图中的所有点都是局部地图点
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    //也只能这样子设置了,毕竟是最近的关键帧
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState = OK; // 初始化成功，至此，初始化过程完成
  }

  /**
   * @brief 检查上一帧中的地图点是否需要被替换
   *
   * Local Mapping线程可能会将关键帧中某些地图点进行替换，
   * 由于tracking中需要用到上一帧地图点，所以这里检查并更新上一帧中被替换的地图点
   *
   * @see LocalMapping::SearchInNeighbors()
   */
  void Tracking::CheckReplacedInLastFrame()
  {
    for (int i = 0; i < mLastFrame.N; i++) // 遍历上一帧的特征点
    {
      MapPoint *pMP = mLastFrame.mvpMapPoints[i]; // 每个特征点对应的MapPoint

      // 上一帧的遍历到的特征点有对应的MapPoint 如果这个地图点存在
      if (pMP)
      {
        // 获取其是否被替换, 以及替换后的点
        // 这也是程序不直接删除这个地图点删除的原因
        MapPoint *pRep = pMP->GetReplaced();

        // 被替换了 此处非空
        if (pRep)
        {
          // 用被替换后的地图点 代替 上一帧对应的那个地图点
          mLastFrame.mvpMapPoints[i] = pRep;
        }
      }
    }
  }

  /**
   * @brief 用参考关键帧的地图点来对当前普通帧进行跟踪 （获得当前帧的精定位）
   *
   * Step 1：将当前普通帧的描述子转化为BoW向量
   * Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
   * Step 3：将上一帧的位姿态作为当前帧位姿的初始值
   * Step 4：通过优化3D-2D的重投影误差来获得位姿
   * Step 5：剔除优化后的匹配点中的外点
   *
   * @return 如果匹配数超10，返回true
   *
   */
  bool Tracking::TrackReferenceKeyFrame()
  {
    // Compute Bag of Words vector
    // Step 1：将当前帧的描述子转化为BoW向量
    // 作用在mCurrentFrame的这两个成员变量
    //    DBoW2::BowVector mBowVec;
    //    DBoW2::FeatureVector mFeatVec;
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7, true);
    vector<MapPoint *> vpMapPointMatches;

    // Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
    // 通过BoW匹配到的当前帧与参考关键的帧特征点数目
    int nmatches = matcher.SearchByBoW(
        mpReferenceKF,      //参考关键帧
        mCurrentFrame,      //当前帧
        vpMapPointMatches); //存储匹配关系 当前帧中MapPoints对应的匹配，NULL表示未匹配

    // 匹配数目小于15，认为跟踪失败
    if (nmatches < 15)
      return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;

    // Step 3：将上一帧的位姿态作为当前帧位姿的初始值
    mCurrentFrame.SetPose(mLastFrame.mTcw); // 用上一次的Tcw设置初值，在PoseOptimization可以收敛快一些

    // Step 4：通过优化3D-2D的重投影误差来获得位姿
    Optimizer::PoseOptimization(&mCurrentFrame); // 优化后的结果通过SetPose()写入

    // Discard outliers
    // Step 5：剔除优化后的匹配点（特征点？）中的外点
    //之所以在优化之后才剔除外点，是因为在优化的过程中就有了对这些外点的标记
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++) // 遍历特征点
    {
      if (mCurrentFrame.mvpMapPoints[i]) // 当前特征点有地图点
      //? 为什么是三角化出地图点的外特征点才需要剔除？
      //  ANSWER：参考slambook2/ch13
      //    对于被认为是异常值的特征，重置特征与路标点的对应关系（而不是重置路标点）
      //    并把它重新记作正常值，认为它只是对应关系错了，并不是所谓的噪点，可能未来有用
      {
        //如果对应到的某个特征点是外点
        if (mCurrentFrame.mvbOutlier[i])
        {
          //清除它在当前帧中存在过的痕迹
          MapPoint *pMP = mCurrentFrame.mvpMapPoints[i]; // 取出 当前帧地图点对应的地图点

          // 清除当前特征点对应的地图点（取消这个对应关系）// 清空这个外点特征点持有的地图点
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
          // maybe we can still use it in future 重新记作正常值
          mCurrentFrame.mvbOutlier[i] = false;

          pMP->mbTrackInView = false;

          // 标记这个地图点最后能被当前帧看到
          pMP->mnLastFrameSeen = mCurrentFrame.mnId;
          nmatches--; //? 这么做有什么意义？这个不是成员变量 作用域就在这个函数里面而已吧？
                      // 可能只是为了结构一样吧 这个函数和用恒速模型跟踪的类似
        }
        //对应到的某个特征点不是外点 且 这个特征点对应的地图点有被观测到
        else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        {
          //匹配的内点计数++
          nmatchesMap++;
        }
      }
    }
    // 跟踪成功的数目超过10才认为跟踪成功，否则跟踪失败
    return nmatchesMap >= 10;
  }

  /**
   * @brief 更新上一帧位姿，在上一帧中生成临时地图点 //? 为什么要更新上一帧啊..？？
   *
   * 单目情况：只计算了上一帧的世界坐标系位姿
   * 双目和RGB-D情况：选取有深度值的 并且 没有被选为地图点的点 生成新的临时地图点，提高跟踪鲁棒性
   */
  void Tracking::UpdateLastFrame()
  {
    // Update pose according to reference keyframe
    // Step 1：利用参考关键帧更新上一帧在世界坐标系下的位姿
    // 上一普通帧的参考关键帧，注意这里用的是参考关键帧（位姿准，被优化过）而不是上上一帧的普通帧
    KeyFrame *pRef = mLastFrame.mpReferenceKF;
    // ref_keyframe 到 lastframe的位姿变换
    // (上一帧 wrt 上一帧的参考关键帧) = 列表的最后一个 //注意：此时当前帧还没有加入到列表，所以最后一个是上一帧
    cv::Mat Tlr = mlRelativeFramePoses.back();

    // 将上一帧的世界坐标系下的位姿计算出来
    // l:last, r:reference, w:world
    // Tlw = Tlr*Trw
    // (上一帧 wrt world) = (上一帧 wrt 上一帧的参考关键帧) * (上一帧的参考关键帧 wrt world)
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    // 如果上一帧为关键帧，或者单目的情况，则退出 // 关键帧由不得你在这里更新，人家有统一的优化
    if (mLastFrame.mnId == mnLastKeyFrameId || mSensor == System::MONOCULAR) //? 单目的检查逻辑考虑前移？
      return;

    // Step 2：对于双目或rgbd相机，为上一帧生成新的临时地图点
    // Create "visual odometry" MapPoints (only for VO and NOT mapping)
    // 注意这些地图点只是用来跟踪，不加入到地图中，跟踪完后会删除（只有关键帧才会参与到建图）

    // We sort points according to their measured depth by the stereo/RGB-D sensor
    // Step 2.1：得到上一帧中具有有效深度值的特征点（不一定是地图点）
    //? 竹曼觉得 RGBD不一定是地图点因为本身就带有深度信息，但是双目的话有深度信息代表是被三角化的特征点（那么应该就是地图点吧？）
    vector<pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N); // 所以下面可以用empty
    /** reserve 是容器预留空间，但并不真正创建元素对象，在创建对象之前，不能引用容器内的元素，
     *      因此当加入新的元素时，需要用push_back() / insert() 函数。
     * resize 是改变容器的大小，并且创建对象，因此，调用这个函数之后，就可以引用容器内的对象了，
     *      因此当加入新的元素时，用operator[] 操作符，或者用迭代器来引用元素对象。
     */

    for (int i = 0; i < mLastFrame.N; i++) // 遍历特征点
    {
      float z = mLastFrame.mvDepth[i];
      if (z > 0) // 深度有效
      {
        // vDepthIdx第一个元素是某个点的深度, 第二个元素是对应的特征点id
        vDepthIdx.push_back(make_pair(z, i));
      }
    }

    // 如果上一帧中没有有效深度的点,那么就直接退出
    if (vDepthIdx.empty())
      return;

    // 按照深度从小到大排序
    sort(vDepthIdx.begin(), vDepthIdx.end()); // 近到远

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    // Step 2.2：从（具有有效深度值的特征点）中找出不是地图点的部分
    int nPoints = 0;
    for (size_t j = 0; j < vDepthIdx.size(); j++) // 遍历具有有效深度的pair
    {
      int i = vDepthIdx[j].second; // 有有效深度的特征点的id

      bool bCreateNew = false;

      // 如果这个点对应在上一帧中的地图点没有, 或者创建后就没有被观测到,那么就生成一个临时的地图点
      MapPoint *pMP = mLastFrame.mvpMapPoints[i];
      if (!pMP)
      {
        // 上一帧中具有有效深度信息 但是不是地图点，认为需要生成为一个临时的地图点？ //? why
        bCreateNew = true;
      }
      else if (pMP->Observations() < 1)
      {
        // 地图点被创建后就没有被观测，认为不靠谱，也需要重新创建
        bCreateNew = true;
      }

      if (bCreateNew)
      {
        // Step 2.3：需要创建的点，包装为地图点。只是为了提高双目和RGBD的跟踪成功率，并没有添加复杂属性，因为后面会扔掉
        // （only for VO NOT mapping）
        // 反投影到世界坐标系中
        cv::Mat x3D = mLastFrame.UnprojectStereo(i); // 当某个特征点的深度信息或者双目信息有效时,将它反投影到三维世界坐标系中

        MapPoint *pNewMP = new MapPoint(
            x3D,         // 世界坐标系坐标
            mpMap,       // 跟踪的全局地图
            &mLastFrame, // 存在这个特征点的帧(上一帧)
            i);          // 特征点id

        // 加入上一帧的地图点中
        mLastFrame.mvpMapPoints[i] = pNewMP; // 上一帧的第i（特征点id）个特征点持有的地图点

        // 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
        mlpTemporalPoints.push_back(pNewMP);

        // 因为从近到远排序，记录其中需要创建（临时）地图点的个数
        nPoints++;
      }
      else
      {
        // 因为从近到远排序，记录其中不需要创建（临时）地图点的个数
        nPoints++;
      }

      // Step 2.4：如果地图点质量不好，停止创建地图点
      // 停止新增临时地图点必须同时满足以下条件：
      // 1、当前的点的深度已经超过了设定的深度阈值（35倍基线）（//?太远了三角化会不准）
      // 2、nPoints已经超过100个点，说明距离比较远了，可能不准确，停掉退出
      if (vDepthIdx[j].first > mThDepth && nPoints > 100)
        break;
    }
  }

  /**
   * @brief 根据恒定速度模型用上一帧地图点来对当前帧进行跟踪
   *
   * Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
   * Step 2：根据上一帧特征点对应地图点进行投影匹配
   * Step 3：优化当前帧位姿
   * Step 4：剔除地图点中外点
   *
   * @return 如果匹配数大于10，认为跟踪成功，返回true
   */
  bool Tracking::TrackWithMotionModel()
  {
    // 最小距离 < 0.9*次小距离 匹配成功，检查旋转
    ORBmatcher matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points (NOT for mapping)
    // Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
    UpdateLastFrame();

    // Step 2：根据之前估计的速度，用恒速模型得到当前帧的初始位姿。
    // mVelocity 在上一循环中的(当前帧 wrt 上一帧) 在恒速模型中永远相等
    // (当前帧 wrt world) = (当前帧 wrt 上一帧) * (上一帧 wrt world)
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

    // 清空当前帧的地图点
    //? why 要看看这个函数什么时候被调用？竹曼猜测是一开始定位的时候 用恒速模型作为初值 所以要先清除
    // Assigns val to all the elements in the range [first,last).
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
         static_cast<MapPoint *>(NULL));

    // Project points seen in previous frame
    // Step 3：用上一帧地图点进行投影匹配，如果匹配点不够，则扩大搜索半径再来一次
    // 设置特征匹配过程中的搜索半径
    int th;
    if (mSensor != System::STEREO)
      th = 15; //单目 //? or RGBD？
    else
      th = 7; //双目

    // 基于描述子的前后帧特征点匹配
    int nmatches = matcher.SearchByProjection(
        mCurrentFrame, mLastFrame,
        th,
        mSensor == System::MONOCULAR);

    // If few matches, uses a wider window search
    // 如果匹配点太少，则扩大搜索半径再来一次
    if (nmatches < 20)
    {
      // 再次清空当前帧的地图点
      //? 为什么还要清空一次？matcher.SearchByProjection会写mCurrentFrame.mvpMapPoints吗？待验证
      fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
           static_cast<MapPoint *>(NULL));

      // 基于描述子的前后帧特征点匹配
      nmatches = matcher.SearchByProjection(
          mCurrentFrame, mLastFrame,
          2 * th, // 扩大搜索半径再来一次
          mSensor == System::MONOCULAR);
    }

    // 如果还是不能够获得足够的匹配点,那么就认为跟踪失败（前后帧的匹配失败，运动过大？）
    if (nmatches < 20)
      return false;

    // Optimize frame pose with all matches: motion-only optimization
    // Step 4：利用3D-2D投影关系，优化当前帧位姿（最小化地图点的重投影误差）
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    // Step 5：剔除地图点中外点。优化之后总是要排除外点清除关系（与TrackReferenceKeyFrame()类似）
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++) // 遍历特征点
    {
      if (mCurrentFrame.mvpMapPoints[i]) // 特征点中有地图点
      {
        if (mCurrentFrame.mvbOutlier[i])
        {
          // 如果优化后判断某个地图点是外点，清除它的所有关系
          MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
          mCurrentFrame.mvbOutlier[i] = false;

          pMP->mbTrackInView = false;

          pMP->mnLastFrameSeen = mCurrentFrame.mnId;
          nmatches--; // 基于描述子的前后帧特征点匹配数目 - 优化后的外点数
        }
        else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        {
          // 累加成功匹配到的地图点数目
          nmatchesMap++;
        }
      }
    }

    if (mbOnlyTracking)
    {
      // 纯定位模式下：如果成功追踪的地图点非常少,那么这里的mbVO标志就会置位
      mbVO = (nmatchesMap < 10); // mbVO==true表示不好
      return (nmatches > 20);    // 前后帧特征点匹配数目太少 or 优化后的外点太多
    }

    // Step 6：匹配超过10个点就认为跟踪成功
    return nmatchesMap >= 10;
  }

  /**
   * @brief 用局部地图进行跟踪，进一步优化位姿
   *
   * 1. 更新局部地图，包括局部关键帧和关键点（特征点？地图点？）
   * 2. 对局部MapPoints进行投影匹配
   * 3. 根据匹配对估计当前帧的姿态
   * 4. 根据姿态剔除误匹配
   *
   * @return true if success
   *
   * Step 1：更新局部关键帧mvpLocalKeyFrames和局部地图点mvpLocalMapPoints
   * Step 2：在局部地图中查找与当前帧匹配的MapPoints（其实也就是对局部地图点进行跟踪）
   * Step 3：更新局部所有MapPoints后对位姿再次优化（还是motion-only optimization）
   * Step 4：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
   * Step 5：决定是否跟踪成功
   */
  bool Tracking::TrackLocalMap()
  {
    // 至此已经实现的
    // 在这个函数之前，在Relocalization、TrackReferenceKeyFrame、TrackWithMotionModel中都有位姿优化
    // We (had) have an estimation of the camera pose and some mappoints tracked in the frame.

    // 这个函数即将要做的
    // We (will) retrieve the local map and try to find matches to points in the local map.

    // Update Local KeyFrames and Local Points
    // Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
    UpdateLocalMap();

    // Step 2：筛选局部地图中新增的在视野范围内的地图点，投影到当前帧搜索匹配，得到更多的匹配关系
    SearchLocalPoints();

    // Optimize Pose
    // Step 3：前面新增了更多的匹配关系，BA优化得到更准确的位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    mnMatchesInliers = 0;
    // Update MapPoints Statistics
    // Step 4：更新当前帧的地图点被观测程度，并统计跟踪局部地图后匹配数目
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
      if (mCurrentFrame.mvpMapPoints[i]) // 当前帧的特征点的地图点
      {
        if (!mCurrentFrame.mvbOutlier[i]) // 不是外点
        {
          // 由于当前帧的地图点可以被当前帧观测到，其被观测统计量加1
          // 找到该点的帧数 mnFound 加 1
          mCurrentFrame.mvpMapPoints[i]->IncreaseFound();

          //查看当前是否是在纯定位过程
          if (!mbOnlyTracking) // 正常SLAM
          {
            // 如果该地图点被相机观测数目nObs大于0，匹配内点计数+1
            // Observations()获取 nObs： 被观测到的相机数目，单目+1，双目或RGB-D则+2
            if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
              mnMatchesInliers++; // 不是外点 且 被观测到
          }
          else // 是纯定位
          {
            // 记录当前帧跟踪到的地图点数目，用于统计跟踪效果
            mnMatchesInliers++;
          }
        }
        // 如果这个地图点是外点,并且当前相机输入还是双目的时候,就删除这个点
        //? 双目的外点就直接没了？ 为何这么严格啊对双目
        // ?单目就不管吗 竹曼不理解
        else if (mSensor == System::STEREO)
        {
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        }
      }
    }

    // Decide if the tracking was successful
    // More restrictive if there was a relocalization recently
    // Step 5：根据跟踪匹配数目及重定位情况决定是否跟踪成功
    // 如果最近刚刚发生了重定位,那么至少成功匹配50个点才认为是成功跟踪
    if ((mCurrentFrame.mnId - mnLastRelocFrameId < mMaxFrames) && // 当前帧与重定位帧足够靠近
        (mnMatchesInliers < 50))
      return false;

    //如果是正常的状态话只要跟踪的地图点大于30个就认为成功了
    if (mnMatchesInliers < 30)
      return false;
    else
      return true;
  }

  /**
   * @brief 判断当前帧是否需要插入关键帧
   *
   * Step 1：纯VO模式下不插入关键帧
   * Step 2：如果局部地图被闭环检测使用，则不插入关键帧
   * Step 3：如果距离上一次重定位比较近，或者关键帧数目超出最大限制，不插入关键帧
   * Step 4：得到参考关键帧跟踪到的地图点数量
   * Step 5：查询局部地图管理器是否繁忙,也就是当前能否接受新的关键帧
   * Step 6：对于双目或RGBD摄像头，统计可以添加的有效地图点总数 和 跟踪到的地图点数量
   * Step 7：决策是否需要插入关键帧
   *
   * @return true   需要
   * @return false   不需要
   */
  // add LK-RGBD-Stereo
  // TODO 插入关键帧策略重大变更！！！
  bool Tracking::NeedNewKeyFrame()
  {
    // add LK-RGBD-Stereo
    static int frameCount = 0;
    if (frameCount < 3)
    {
      frameCount++;
      return true;
    }

    //! bug: if LK is not initied,
    //! the current frame should be tracking by original ORBSLAM
    if (mnMatchesInliers == -1)
    {
      return true;
    }

    // Step 1：纯VO模式下不插入关键帧
    if (mbOnlyTracking)
      return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // Step 2：如果局部地图线程被闭环检测使用，则不插入关键帧
    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
      return false;

    // 获取当前地图中的关键帧数目
    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalization
    // Step 3：如果距离上一次重定位比较近 且 关键帧数目超出最大限制，不插入关键帧
    // mCurrentFrame.mnId是当前帧的ID
    // mnLastRelocFrameId是最近一次重定位帧的ID
    // mMaxFrames等于图像输入的帧率
    if ((mCurrentFrame.mnId - mnLastRelocFrameId < mMaxFrames) && (nKFs > mMaxFrames))
      return false;

    /*
// add LK-RGBD-Stereo

    // Tracked MapPoints in the reference keyframe
    // Step 4：得到参考关键帧跟踪到的地图点数量
    // UpdateLocalKeyFrames()函数中会将 与当前关键帧共视程度最高的关键帧 设定为 当前帧的参考关键帧

    // 地图点的最小观测次数
    int nMinObs = 3;
    if (nKFs <= 2) // 当前地图中的关键帧数目
      nMinObs = 2;

    // 参考关键帧的地图点中观测的数目>= nMinObs 的地图点数目
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

// end add LK-RGBD-Stereo
    */

    // Local Mapping accept keyframes?
    // Step 5：查询局部地图线程是否繁忙，当前能否接受新的关键帧
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    /*
// add LK-RGBD-Stereo

    // Check how many "close" points are being tracked and how many could be potentially created.
    // Step 6：对于双目或RGBD摄像头，统计成功跟踪的近点的数量：
    // 如果跟踪到的近点太少，没有跟踪到的近点较多，可以插入关键帧 //? 竹曼的理解 需要插入关键帧，这是关键帧的选取策略吗？
    int nNonTrackedClose = 0; //双目或RGB-D中没有跟踪到的近点
    int nTrackedClose = 0;    //双目或RGB-D中成功跟踪的近点（三维点） //? 被三角化成了三维点 但是没有被选入地图点？
    if (mSensor != System::MONOCULAR)
    {
      for (int i = 0; i < mCurrentFrame.N; i++)
      {
        // 深度值在有效范围内 且 都是近点
        if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
        {
          // 是地图点 且 不是外点
          if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
            nTrackedClose++;
          // 不是地图点 或 是外点
          else
            nNonTrackedClose++;
        }
      }
    }

    //? 这里应该是orbslam2的插入关键帧策略
    // 双目或RGBD情况下：跟踪到的地图点中近点太少 且 没有跟踪到的三维点太多，可以插入关键帧了
    // 单目情况下：恒为false
    bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

    // Step 7：决策是否需要插入关键帧
    // Thresholds （阈值越小对关键帧的插入越严格）
    // Step 7.1：设定比例阈值：当前帧和参考关键帧跟踪到点的比例，比例越大，越倾向于增加关键帧
    //                        （当前帧跟踪到的多，参考关键帧跟踪到的少）
    float thRefRatio = 0.75f;

    // 关键帧只有一帧，那么插入关键帧的阈值设置的低一点，插入频率较低
    if (nKFs < 2)
      thRefRatio = 0.4f;

    //单目情况下插入关键帧的频率很高
    if (mSensor == System::MONOCULAR)
      thRefRatio = 0.9f;

// end add LK-RGBD-Stereo
    */

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // Step 7.2：很长时间没有插入关键帧，可以插入
    const bool c1a = (mCurrentFrame.mnId - mnLastKeyFrameId) >= mMaxFrames;

    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // Step 7.3：满足插入关键帧的最小间隔 且 localMapper处于空闲状态，可以插入
    const bool c1b = ((mCurrentFrame.mnId - mnLastKeyFrameId) >= mMinFrames &&
                      bLocalMappingIdle);

    /*
// add LK-RGBD-Stereo

    // Condition 1c: tracking is weak
    // Step 7.4：在双目，RGB-D的情况下：当前帧跟踪到的点比参考关键帧的0.25倍还少，或者满足bNeedToInsertClose
    const bool c1c =
        (mSensor != System::MONOCULAR) && //只考虑在双目，RGB-D的情况
        //当前帧中的进行匹配的内点 < (参考关键帧的地图点中观测的数目>= nMinObs 的地图点数目)*0.25
        (mnMatchesInliers < nRefMatches * 0.25 || // 当前帧中能和地图点匹配的数目非常少（离关键帧足够远了）
         bNeedToInsertClose);                     // 根据 成功跟踪的近点的数量 判断到的需要插入

    // Condition 2: Few tracked points compared to reference keyframe.
    //         Lots of visual odometry compared to map matches.
    // Step 7.5：和参考帧相比当前跟踪到的点太少 或 满足bNeedToInsertClose 且 跟踪到的内点还不能太少
    const bool c2 =
        ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) &&
         mnMatchesInliers > 15); // 跟踪到的内点太少：//?就是跟丢了吧？

// end add LK-RGBD-Stereo
    */

    // add LK-RGBD-Stereo
    //! debug: Floating point exception
    if (mnMatchesInliers == 0)
      mnMatchesInliers = 1;
    // the greater threshold, the more strictly, so more LK
    const bool c3 = (mSensor != System::MONOCULAR) && (mnMatchesInliers < 300 && mnMatchesInliers > 0);
    const bool c4 = (mSensor != System::MONOCULAR) && (mnMatchesInliers < 100 && mnMatchesInliers > 0);
    const bool c5 = (last_mnMatchesInliers / mnMatchesInliers) > 2; // inlier decrease fast
    // end add LK-RGBD-Stereo

    // original ORBSLAM2 Criterion
    // if ((c1a || c1b || c1c) && c2)

    // add LK-RGBD-Stereo
    // c1a: long time no keyframe inserted
    // c1b: keyframe interval larger than threshold AND LocalMapper is available
    // c3: the number of matched inlier in frame-to-frame (0, 300)
    // c4: the number of matched inlier in frame-to-frame (0, 100)
    // c5: the number of matched inlier in frame-to-frame decrease too fast
    if (((c1a || c1b) && c3) || (c4 || c5))
    {
      // If the mapping accepts keyframes, insert keyframe.
      // Otherwise send a signal to interrupt BA
      // Step 7.6：local mapping空闲时可以直接插入，不空闲的时候要根据情况插入
      if (bLocalMappingIdle)
      {
        //可以插入关键帧
        return true;
      }
      // 此时local mapping线程非空闲（意味着前面可能有要插入的关键帧在等待）
      else
      {
        // 发信号终止BA
        mpLocalMapper->InterruptBA();
        if (mSensor != System::MONOCULAR)
        {

          // 队列里不能阻塞太多关键帧
          // tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
          // 然后localmapper再逐个pop出来插入到mspKeyFrames

          // add LK-RGBD-Stereo
          // if (mpLocalMapper->KeyframesInQueue() < 3)
          if (mpLocalMapper->KeyframesInQueue() < 2)
            //队列中的关键帧数目不是很多,可以插入
            return true;
          else
            //队列中缓冲的关键帧数目太多,暂时不能插入
            return false;
        }
        else
          // 对于单目，一旦local mapping线程非空闲就直接无法插入关键帧了
          //? 为什么这里对单目情况的处理不一样?
          //? 回答：可能是单目关键帧相对比较密集（无法缓存这么多？） 竹曼待确认
          return false;
      }
    }
    else
      //不满足上面的条件,自然不能插入关键帧
      return false;
  }

  /**
   * @brief 创建新的关键帧
   * 对于非单目的情况，同时创建新的MapPoints（双目关键帧的特征点三角化出地图点用于建图）
   *
   * Step 1：将当前帧构造成关键帧
   * Step 2：将当前关键帧设置为当前帧的参考关键帧
   * Step 3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
   *
   */
  void Tracking::CreateNewKeyFrame()
  {
    // 如果局部建图线程关闭了,就无法插入关键帧 （local mapping线程用作接收关键帧）
    if (!mpLocalMapper->SetNotStop(true))
    {
      // mpLocalMapper的成员变量mbStopped == true
      return;
    }

    // Step 1：将当前帧构造成关键帧（把当前帧升级为关键帧）
    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    // Step 2：将当前关键帧 设置为 当前帧的参考关键帧
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧 设定为当前帧的参考关键帧
    mpReferenceKF = pKF;               // 对于tracking线程而言
    mCurrentFrame.mpReferenceKF = pKF; //? 对于tracking处理到的当前帧这个对象而言（处理完还会被保存）

    // 这段代码和 Tracking::UpdateLastFrame 中的那一部分代码功能相同
    // Step 3：对于双目或rgbd摄像头，为当前帧生成新的地图点；单目无操作
    if (mSensor != System::MONOCULAR)
    {
      // 根据Tcw计算mRcw、mtcw和mRwc、mOw
      // (当前帧 wrt world)的SE3 得到 (当前帧 wrt world)的旋转矩阵 mRcw
      //                            平移向量 mtcw
      // (world wrt 当前帧)的旋转矩阵mRwc 也就是mRcw的逆
      // mOw同理
      mCurrentFrame.UpdatePoseMatrices();

      // We sort points by the measured depth by the stereo/RGBD sensor.
      // We create all those MapPoints whose depth < mThDepth.
      // If there are less than 100 close points we create the 100 closest.
      // Step 3.1：得到当前帧有深度值的特征点（不一定是地图点）
      vector<pair<float, int>> vDepthIdx;
      vDepthIdx.reserve(mCurrentFrame.N);
      for (int i = 0; i < mCurrentFrame.N; i++)
      {
        float z = mCurrentFrame.mvDepth[i];
        if (z > 0)
        {
          // 第一个元素是深度,第二个元素是对应的特征点的id
          vDepthIdx.push_back(make_pair(z, i));
        }
      }

      if (!vDepthIdx.empty())
      {
        // Step 3.2：按照深度从小到大排序（近到远）
        sort(vDepthIdx.begin(), vDepthIdx.end());

        // Step 3.3：从中找出不是地图点的有深度的特征点 然后生成临时地图点
        int nPoints = 0;                              // 处理的近点的个数
        for (size_t j = 0; j < vDepthIdx.size(); j++) // 遍历符合有效深度的特征点
        {
          int i = vDepthIdx[j].second; // 特征点的id

          bool bCreateNew = false;

          // 如果这个点对应在当前帧中的地图点没有, 或者创建后就没有被观测到,
          // 那么就生成一个临时的地图点
          MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
          if (!pMP)
            bCreateNew = true;
          else if (pMP->Observations() < 1)
          {
            bCreateNew = true;
            // 当前帧的地图点在创建后就没有被观测到 //? 要删掉这个地图点 竹曼理解 删掉重建
            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
          }

          // 如果需要就新建地图点，这里的地图点不是临时的，
          // 这是全局地图中新建地图点，用于跟踪（因为现在是关键帧：用于建图+定位）
          if (bCreateNew)
          {
            cv::Mat x3D = mCurrentFrame.UnprojectStereo(i); // 得到特征点的反投影点的世界坐标

            MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
            // 这些添加属性的操作是每次创建MapPoint后都要做的
            //? 下面的i都是：MapPoint在KeyFrame中的索引，也就是特征点id
            pNewMP->AddObservation(pKF, i);          // 地图点被关键帧观测到
            pKF->AddMapPoint(pNewMP, i);             // 关键帧看到了地图点
            pNewMP->ComputeDistinctiveDescriptors(); // 计算具有代表的描述子：这个地图点对应的关键帧的描述子应为最具代表性
            pNewMP->UpdateNormalAndDepth();          // 更新平均观测方向以及观测距离范围 //?（感觉和上一个类似）
            mpMap->AddMapPoint(pNewMP);

            // 配置完的新地图点 加入到 当前帧的某个特征点对应的地图点
            mCurrentFrame.mvpMapPoints[i] = pNewMP;
            nPoints++;
          }
          else
          {
            // 因为从近到远排序，记录其中不需要创建地图点的个数
            nPoints++;
          }

          // Step 3.4：停止新建地图点必须同时满足以下条件：
          // 1、当前的点的深度已经超过了设定的深度阈值（35倍基线）
          // 2、nPoints已经超过100个点，说明距离比较远了，可能不准确，停掉退出
          if (vDepthIdx[j].first > mThDepth && nPoints > 100)
            break;
        }
      }
    }

    // Step 4：插入关键帧
    // 关键帧插入到列表 mlNewKeyFrames中，并传给LocalMapping线程，发出中断BA信号
    // 等待local mapping线程临幸 //? 竹曼理解 local mapping会查询 mlNewKeyFrames容器里面有没有新东西
    mpLocalMapper->InsertKeyFrame(pKF);

    // 插入好了，允许局部建图停止
    mpLocalMapper->SetNotStop(false);

    // 当前帧成为新的关键帧，更新
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
  }

  /**
   * @brief 用局部地图点进行投影匹配，得到更多的匹配关系
   *
   * 注意：局部地图点中已经是当前帧地图点的不需要再投影，
   *    只需要将此外的并且在视野范围内的点和当前帧进行投影匹配（投影到当前帧）
   */
  void Tracking::SearchLocalPoints()
  {
    // Do not search map points already matched
    // Step 1：遍历当前帧的地图点，标记这些地图点不参与之后的投影搜索匹配（自己投影到自己没有意义）
    for (vector<MapPoint *>::iterator
             vit = mCurrentFrame.mvpMapPoints.begin(),
             vend = mCurrentFrame.mvpMapPoints.end();
         vit != vend; vit++)
    {
      MapPoint *pMP = *vit; // 当前帧持有的地图点
      if (pMP)
      {
        if (pMP->isBad())
        {
          // 删除坏点
          *vit = static_cast<MapPoint *>(NULL);
        }
        else
        {
          // 更新能观测到该点的帧数加1(这个地图点被当前帧观测了，因为当前帧持有了这个地图点)
          pMP->IncreaseVisible();
          // 标记该点被当前帧观测到
          pMP->mnLastFrameSeen = mCurrentFrame.mnId;
          // 标记该点在后面搜索匹配时不被投影，因为已经有匹配了
          pMP->mbTrackInView = false;
        }
      }
    }

    // 准备进行投影匹配的点的数目
    int nToMatch = 0;

    // Project points in frame and check its visibility
    // Step 2：判断所有局部地图点中除当前帧地图点外的点，是否在当前帧视野范围内
    for (vector<MapPoint *>::iterator
             vit = mvpLocalMapPoints.begin(),
             vend = mvpLocalMapPoints.end();
         vit != vend; vit++)
    {
      MapPoint *pMP = *vit;

      // 已经被当前帧观测到的地图点肯定在视野范围内，跳过
      if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
        continue;

      // 跳过坏点
      if (pMP->isBad())
        continue;

      //? Project (this fills MapPoint variables for matching)
      // 判断地图点是否在在当前帧视野内
      if (mCurrentFrame.isInFrustum(pMP, 0.5))
      {
        // 观测到该点的帧数加1
        pMP->IncreaseVisible();
        // 只有在视野范围内的地图点才参与之后的投影匹配
        nToMatch++;
      }
    }

    // Step 3：如果需要进行投影匹配的点的数目大于0，就进行投影匹配，增加更多的匹配关系
    if (nToMatch > 0)
    {
      ORBmatcher matcher(0.8);

      //? 越大越宽松？
      int th = 1;

      if (mSensor == System::RGBD) // RGBD相机输入的时候,搜索的阈值会变得稍微大一些
        th = 3;

      // If the camera has been relocalised recently, perform a coarser search
      // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
      if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
        th = 5;

      // 投影匹配得到更多的匹配关系
      // 通过投影地图点到当前帧，对Local MapPoint进行跟踪
      matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
    }
  }

  /**
   * @brief 更新LocalMap
   *
   * 局部地图包括：
   * 1、K1个关键帧、K2个临近关键帧和参考关键帧 //?
   * 2、由这些关键帧观测到的MapPoints
   *
   */
  void Tracking::UpdateLocalMap()
  {
    // This is for visualization
    // 设置参考地图点用于绘图显示局部地图点（红色）
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    // 用共视图来更新局部关键帧和局部地图点
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
  }

  /**
   * @brief 更新局部关键点。
   *
   * 先把局部地图清空，
   * 然后将局部关键帧的有效地图点添加到局部地图中
   *
   */
  void Tracking::UpdateLocalPoints()
  {
    // Step 1：清空局部地图点
    mvpLocalMapPoints.clear(); // 清空vector

    // Step 2：遍历局部关键帧 mvpLocalKeyFrames
    for (vector<KeyFrame *>::const_iterator // const迭代器 不能修改里面的值
             itKF = mvpLocalKeyFrames.begin(),
             itEndKF = mvpLocalKeyFrames.end();
         itKF != itEndKF; itKF++)
    {
      KeyFrame *pKF = *itKF;

      // 获取该关键帧的MapPoints
      const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

      // step 3：将局部关键帧的地图点添加到mvpLocalMapPoints
      for (vector<MapPoint *>::const_iterator
               itMP = vpMPs.begin(),
               itEndMP = vpMPs.end();
           itMP != itEndMP; itMP++)
      {
        MapPoint *pMP = *itMP;

        if (!pMP)
          continue;

        // 用该地图点的成员变量mnTrackReferenceForFrame 记录当前帧的id
        // 表示它已经是当前帧的局部地图点了（已经添加过），可以防止重复添加局部地图点
        //? 局部关键帧有很多个 他们有可能同时观测到同一个地图点 所以会有重复 这样子吗？
        if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
          continue;

        if (!pMP->isBad()) // 不是坏点
        {
          mvpLocalMapPoints.push_back(pMP);

          // 标记这个局部关键帧的地图点的id（防止重复）
          pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }
      }
    }
  }

  /**
   * @brief //? 跟踪局部地图函数里，更新局部关键帧
   * @brief 更新局部关键帧
   *
   * @details
   * 方法是遍历当前帧的地图点，
   * 将观测到这些地图点的 关键帧 和 相邻的关键帧 及其父子关键帧，作为mvpLocalKeyFrames
   *
   * Step 1：遍历当前帧的地图点，记录所有能观测到当前帧地图点的关键帧
   * Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧包括以下3种类型
   *      类型1：能观测到当前帧地图点的关键帧，也称一级共视关键帧
   *      // 先产生一级 随后的都是base on一级
   *      类型2：一级共视关键帧的共视关键帧，称为二级共视关键帧
   *      类型3：一级共视关键帧的子关键帧、父关键帧
   * Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
   *
   */
  void Tracking::UpdateLocalKeyFrames()
  {
    // Each map point vote for the keyframes in which it has been observed
    // Step 1：遍历当前帧的地图点（出发点），记录所有能观测到当前帧地图点的关键帧（作为候选人）
    map<KeyFrame *, int> keyframeCounter; // 对关键帧候选人投票，看看哪个关键帧持有更多的地图点 更关键

    for (int i = 0; i < mCurrentFrame.N; i++) // 遍历当前帧的特征点（经三角化的特征点才有可能是地图点）
    {
      if (mCurrentFrame.mvpMapPoints[i]) // 这个特征点有生成地图点
      {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

        // 不是坏点
        if (!pMP->isBad())
        {
          // 观测到该地图点的 关键帧 和 该地图点在关键帧中的索引
          //                （该地图点在该关键帧的特征点的访问id）
          const map<KeyFrame *, size_t> observations = pMP->GetObservations();

          // 由于一个地图点可以被多个关键帧观测到,
          // 因此对于每一次观测,都对观测到这个地图点的关键帧进行累计投票
          // （对关键帧投票，看看哪个关键帧更关键）
          for (map<KeyFrame *, size_t>::const_iterator
                   it = observations.begin(),
                   itend = observations.end();
               it != itend; it++)
            /*
             * 这里的操作非常精彩！
             *
             * map[key] = value
             * 当要插入的键存在时，会覆盖键对应的原来的值；如果键不存在，则添加一组键值对
             *
             * it->first是地图点看到的关键帧，
             * 同一个关键帧持有的地图点会累加到该关键帧计数，所以最后keyframeCounter
             *
             * 第一个参数表示某个关键帧；
             * 第二个参数表示该关键帧看到了多少当前帧(mCurrentFrame)的地图点，也就是共视程度
             */
            keyframeCounter[it->first]++; // 关键帧的共视次数（同一个关键帧被不同的地图点看到）
        }
        // 是坏点
        else
        {
          mCurrentFrame.mvpMapPoints[i] = NULL; // 删掉地图点
        }
      }
    }

    // 没有当前帧没有共视关键帧，返回
    //? observations是空的才能满足吧？但是observations不可能是空的啊
    //? 因为地图点必定是由关键帧产生的 除非都是坏点？
    if (keyframeCounter.empty())
      return;

    // 存储具有最多观测次数（max）的关键帧 //临时变量 选出最叻的帧
    int max = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    // Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有3种类型
    // 先清空局部关键帧
    mvpLocalKeyFrames.clear();
    // 先申请3倍内存，不够后面再加 // keyframeCounter 相当于关键帧候选人
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. （候选人名单）
    // Also check which keyframe shares most points （查共视次数最多的）
    // Step 2.1
    // 类型1：能观测到当前帧地图点的关键帧（候选人名单）作为局部关键帧
    // （将邻居拉拢入伙）（一级共视关键帧）
    for (map<KeyFrame *, int>::const_iterator // 遍历候选人
             it = keyframeCounter.begin(),    // map<KeyFrame *, int> keyframeCounter;
         itEnd = keyframeCounter.end();       // 候选关键帧，候选关键帧持有的地图点数目
         it != itEnd; it++)
    {
      KeyFrame *pKF = it->first;

      // 如果设定为要删除的，跳过
      if (pKF->isBad())
        continue;

      // 寻找具有最大观测数目的关键帧
      if (it->second > max)
      {
        max = it->second;
        pKFmax = pKF;
      }

      // 添加到局部关键帧的列表里
      mvpLocalKeyFrames.push_back(it->first);

      // 用该关键帧的成员变量mnTrackReferenceForFrame 记录当前帧的id
      // 表示它已经是当前帧的局部关键帧了（已经添加过），可以防止重复添加局部关键帧
      pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to
    // already-included keyframes
    // Step 2.2 遍历一级共视关键帧，寻找更多的局部关键帧
    for (vector<KeyFrame *>::const_iterator
             itKF = mvpLocalKeyFrames.begin(),
             itEndKF = mvpLocalKeyFrames.end();
         itKF != itEndKF; itKF++)
    {
      // Limit the number of keyframes
      // 处理的局部关键帧不超过80帧
      if (mvpLocalKeyFrames.size() > 80)
        break;

      KeyFrame *pKF = *itKF; // 遍历的 一级共视关键帧 对2.2 2.3有效

      // 类型2：一级共视关键帧的共视（前10个）关键帧，称为二级共视关键帧（将邻居的邻居拉拢入伙）
      // 如果共视帧不足10帧，那么就返回所有具有共视关系的关键帧
      const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

      // vNeighs 是按照共视程度从大到小排列 现在遍历它
      for (vector<KeyFrame *>::const_iterator
               itNeighKF = vNeighs.begin(),
               itEndNeighKF = vNeighs.end();
           itNeighKF != itEndNeighKF; itNeighKF++)
      {
        KeyFrame *pNeighKF = *itNeighKF;
        if (!pNeighKF->isBad())
        {
          // mnTrackReferenceForFrame防止重复添加局部关键帧
          if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
          {
            mvpLocalKeyFrames.push_back(pNeighKF);
            pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;

            //? 找到一个就直接跳出for循环？
            // 对于一级出来的二级 只要共视程度最高的那个
            break;
          }
        }
      }

      // Step 2.3.1
      // 类型3.1：将一级共视关键帧的子关键帧作为局部关键帧（将邻居的孩子们拉拢入伙）
      //? GetChilds()的返回类型是set 因为没有重复吗？
      const set<KeyFrame *> spChilds = pKF->GetChilds();
      for (set<KeyFrame *>::const_iterator
               sit = spChilds.begin(),
               send = spChilds.end();
           sit != send; sit++)
      {
        KeyFrame *pChildKF = *sit;
        if (!pChildKF->isBad())
        {
          // mnTrackReferenceForFrame防止重复添加局部关键帧
          if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
          {
            mvpLocalKeyFrames.push_back(pChildKF);
            pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;

            //? 找到一个就直接跳出for循环？
            break;
          }
        }
      }

      // Step 2.3.2
      // 类型3.2：将一级共视关键帧的父关键帧（将邻居的父母们拉拢入伙）
      KeyFrame *pParent = pKF->GetParent();
      if (pParent)
      {
        // mnTrackReferenceForFrame防止重复添加局部关键帧
        if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
        {
          mvpLocalKeyFrames.push_back(pParent);
          pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;

          //! 感觉是个bug！如果找到父关键帧会直接跳出整个循环
          //! 竹曼也觉得啊 所以竹曼注释掉这个break
          // break;
        }
      }
    } // end 遍历一级共视关键帧

    // Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
    //? 竹曼觉得这个step3 可以放到前面吧 pKFmax很早就已经确定了啊
    if (pKFmax)
    {
      mpReferenceKF = pKFmax;
      mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
  }

  /**
   * @brief 重定位过程
   *
   * @details
   * Step 1：计算当前帧特征点的词袋向量
   * Step 2：找到与当前帧相似的候选关键帧
   * Step 3：通过BoW进行匹配
   * Step 4：通过EPnP算法估计姿态
   * Step 5：通过PoseOptimization对姿态进行优化求解
   * Step 6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
   *
   * @return bool
   */
  bool Tracking::Relocalization()
  {
    // Compute Bag of Words Vector
    // Step 1：计算当前帧特征点的词袋向量
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalization
    // Step 2：用词袋找到与当前帧相似的候选关键帧 （后面运算都是基于这里的候选关键帧）
    vector<KeyFrame *> vpCandidateKFs =
        mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    // 如果没有候选关键帧，则退出
    if (vpCandidateKFs.empty())
      return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75, true);
    //每个关键帧的解算器：每个关键帧都需要一个独立的求解器
    // 记录具体的任务要匹配的当前帧 和 要匹配的关键帧候选人
    vector<PnPsolver *> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs); // resize会初始化里面的元素

    //每个关键帧和当前帧中特征点的匹配关系
    // 为每一个候选关键帧生成一个vector<MapPoint *>：每个候选关键帧都有自己持有的地图点
    vector<vector<MapPoint *>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    //放弃某个关键帧的标记
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    //有效的候选关键帧数目
    int nCandidates = 0; // 坏关键帧无效

    // Step 3：遍历所有的候选关键帧，通过词袋进行快速匹配（粗匹配），用匹配结果初始化PnP
    // Solver
    for (int i = 0; i < nKFs; i++)
    {
      KeyFrame *pKF = vpCandidateKFs[i]; // 候选关键帧
      if (pKF->isBad())
        vbDiscarded[i] = true;
      else
      {
        // 当前帧和候选关键帧用BoW进行快速匹配，匹配结果记录在vvpMapPointMatches
        //? BoW粗匹配？ 粗略匹配的结果vvpMapPointMatches 表示候选关键帧与当前帧能匹配上的地图点
        int nmatches = // nmatches表示匹配的数目
            matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);

        // 如果和当前帧的匹配数小于15，那么只能放弃这个关键帧
        if (nmatches < 15)
        {
          vbDiscarded[i] = true;
          continue;
        }
        // 如果匹配数目够用，用匹配结果初始化EPnPsolver
        else
        {
          // 为什么用EPnP? 因为计算复杂度低，精度高
          // 配置EPnP求解器：当前帧 和 BoW粗匹配的地图点
          PnPsolver *pSolver =
              new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
          pSolver->SetRansacParameters(
              0.99,   //用于计算RANSAC迭代次数理论值的概率
              10,     //最小内点数,
                      //但是要注意在程序中实际上是min(给定最小内点数,最小集,内点数理论值),不一定使用这个
              300,    //最大迭代次数
              4,      //最小集(求解这个问题在一次采样中所需要采样的最少的点的个数,对于Sim3是3,EPnP是4),参与到最小内点数的确定过程中
              0.5,    //这个是表示(最小内点数/样本总数);实际上的RANSAC正常退出的时候所需要的最小内点数其实是根据这个量来计算得到的
              5.991); // 自由度为2的卡方检验的阈值,程序中还会根据特征点所在的图层对这个阈值进行缩放
          vpPnPsolvers[i] = pSolver;
          nCandidates++;
        }
      }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    // 这里的 P4P RANSAC是EPnP，每次迭代需要4个点

    bool bMatch = false; // 是否已经找到相匹配的关键帧的标志
    ORBmatcher matcher2(0.9, true);

    // Step 4: 通过一系列操作（精匹配），直到找到能够匹配上的关键帧
    // 为什么搞这么复杂？答：是担心误闭环

    // 外围的终止条件：每次处理完一个候选关键帧 都要看看是不是可以结束了
    //有效的候选关键帧数目（还没逐个检查完） 且 没有找到匹配
    while (nCandidates > 0 && !bMatch)
    {
      //遍历所有的候选关键帧
      for (int i = 0; i < nKFs; i++) // nKFs 总是大于 nCandidates
      {
        // 忽略放弃的
        if (vbDiscarded[i])
          continue;

        //内点标记
        vector<bool> vbInliers;

        //内点数
        int nInliers;

        // 表示RANSAC已经没有更多的迭代次数可用 --
        // 也就是说数据不够好，但是RANSAC也已经尽力了。。。
        bool bNoMore;

        // Step 4.1：通过EPnP算法估计姿态，迭代5次
        PnPsolver *pSolver = vpPnPsolvers[i];
        // EPnP估计位姿结果
        cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

        // If Ransac reaches max. iterations discard keyframe
        // bNoMore 为true ：表示已经超过了RANSAC最大迭代次数，就放弃当前关键帧
        if (bNoMore)
        {
          vbDiscarded[i] = true;
          nCandidates--;
        }

        // If a Camera Pose is computed, then optimize
        if (!Tcw.empty())
        {
          //  Step 4.2：如果EPnP 计算出了位姿，对内点进行BA优化
          Tcw.copyTo(mCurrentFrame.mTcw); // 当前帧的位姿 = EPnP估计位姿结果

          // EPnP 里RANSAC后的内点的集合
          set<MapPoint *> sFound;

          const int np = vbInliers.size(); // EPnP结果的内点标记 的数目
          //遍历所有内点 //?（内点：匹配情况很好的地图点？）
          for (int j = 0; j < np; j++) //? 这里的j不是特征点id吗
          {
            if (vbInliers[j])
            {
              // 重定位的目的就是给当前帧指派位姿，现在给当前帧赋予匹配情况好的地图点
              mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j]; // 第i关键帧里面的第j地图点
              sFound.insert(vvpMapPointMatches[i][j]);
            }
            else
            {
              // 不是内点的匹配 就把当前帧对应的地图点清空
              mCurrentFrame.mvpMapPoints[j] = NULL;
            }
          }

          // 只优化位姿,不优化地图点的坐标，返回的是内点的数量
          // 前面的 BoW粗匹配+EPnP精匹配 已经赋值了 当前帧的 位姿+地图点，现在进行motion-only BA
          int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

          // 如果优化之后的内点数目不多，跳过了当前候选关键帧，
          // 但是却没有放弃当前帧的重定位
          if (nGood < 10)
            continue;

          // 删除外点对应的地图点
          for (int io = 0; io < mCurrentFrame.N; io++)
          {
            // motion-only BA 优化之后会写入的成员变量
            if (mCurrentFrame.mvbOutlier[io])
            {
              mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);
            }
          }

          // If few inliers, search by projection in a coarse（粗） window and optimize again
          // Step 4.3：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
          // 前面的匹配关系是用词袋匹配过程得到的
          if (nGood < 50)
          {
            // 通过投影的方式将关键帧中未匹配的地图点投影到当前帧中, 生成新的匹配
            int nadditional = matcher2.SearchByProjection(
                mCurrentFrame,     //当前帧
                vpCandidateKFs[i], //关键帧
                sFound,            //已经找到的地图点集合，不会用于PNP
                10,                //窗口阈值，会乘以金字塔尺度
                100);              //匹配的ORB描述子距离应该小于这个阈值

            // 如果通过投影过程新增了比较多的匹配特征点对
            if (nadditional + nGood >= 50)
            {
              // 根据投影匹配的结果，再次采用3D-2D pnp BA优化位姿
              nGood = Optimizer::PoseOptimization(&mCurrentFrame);

              // If many inliers but still not enough, search by projection again in
              // a narrower window the camera has been already optimized with many points
              // Step 4.4：如果BA后内点数还是比较少(<50)但是还不至于太少(>30)，可以挽救一下,
              // 最后垂死挣扎 重新执行上一步 4.3的过程，只不过使用更小的搜索窗口
              // 这里的位姿已经使用了更多的点进行了优化,应该更准，所以使用更小的窗口搜索
              if (nGood > 30 && nGood < 50)
              {
                // 用更小窗口、更严格的描述子阈值，重新进行投影搜索匹配
                sFound.clear();
                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                {
                  if (mCurrentFrame.mvpMapPoints[ip])
                  {
                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                  }
                }
                nadditional = matcher2.SearchByProjection(
                    mCurrentFrame,     //当前帧
                    vpCandidateKFs[i], //候选的关键帧
                    sFound,            //已经找到的地图点，不会用于PNP
                    3,                 //新的窗口阈值，会乘以金字塔尺度
                    64);               //匹配的ORB描述子距离应该小于这个阈值

                // Final optimization
                // 如果成功挽救回来，匹配数目达到要求，最后BA优化一下
                if (nGood + nadditional >= 50)
                {
                  nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                  //更新地图点
                  for (int io = 0; io < mCurrentFrame.N; io++)
                  {
                    if (mCurrentFrame.mvbOutlier[io])
                    {
                      mCurrentFrame.mvpMapPoints[io] = NULL;
                    }
                  }
                }
                //如果还是不能够满足就放弃了
              }
            }
          }

          // If the pose is supported by enough inliers stop ransacs and continue
          // 如果对于当前的候选关键帧已经有足够的内点(50个)了,那么就认为重定位成功
          if (nGood >= 50)
          {
            bMatch = true;

            // 只要有一个候选关键帧重定位成功，就退出循环，不考虑其他候选关键帧了
            break;
          }
        }
      } // end 遍历所有的候选关键帧
    }   //一直运行，直到已经没有候选关键帧，或者已经有成功匹配上的候选关键帧

    // 折腾了这么久还是没有匹配上，重定位失败
    if (!bMatch)
    {
      return false;
    }
    else
    {
      // 如果匹配上了,说明当前帧重定位成功了(当前帧已经有了自己的位姿)
      // 记录成功重定位帧的id，防止短时间多次重定位
      mnLastRelocFrameId = mCurrentFrame.mnId; // 当前帧触发过重定位
      return true;
    }
  }

  /**
   * @brief 整个追踪线程执行复位操作
   *
   * @details 基本上是挨个请求各个线程终止
   */
  void Tracking::Reset()
  {
    if (mpViewer)
    {
      mpViewer->RequestStop();
      while (!mpViewer->isStopped())
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    cout << "System Reseting" << endl;

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    //然后复位各种变量
    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if (mpInitializer)
    {
      delete mpInitializer; // 释放mpInitializer指向的东西
      mpInitializer = static_cast<Initializer *>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if (mpViewer)
      mpViewer->Release();
  }

  //目测是 根据配置文件中的参数重新改变已经设置在系统中的参数,但是当前文件中没有找到对它的调用
  //?
  void Tracking::ChangeCalibration(const string &strSettingPath)
  {
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
      DistCoef.resize(5);
      DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    //做标记,表示在初始化帧的时候将会是第一个帧,要对它进行一些特殊的初始化操作
    Frame::mbInitialComputations = true;
  }

  void Tracking::InformOnlyTracking(const bool &flag) { mbOnlyTracking = flag; }

} // namespace ORB_SLAM2
