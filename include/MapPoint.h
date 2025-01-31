/**
 * @file MapPoint.h
 * @author guoqing (1337841346@qq.com)
 * @brief 地图点
 * @version 0.1
 * @date 2019-02-26
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

/*
- 创建MapPoint的时机:
1.  Tracking线程中初始化过程(Tracking::MonocularInitialization()和Tracking::StereoInitialization())
2.  Tracking线程中创建新的关键帧(Tracking::CreateNewKeyFrame())
3.  Tracking线程中恒速运动模型跟踪(Tracking::TrackWithMotionModel())也会产生临时地图点,
    但这些临时地图点在跟踪成功后会被马上删除(那跟踪失败怎么办?跟踪失败的话不会产生关键帧,这些地图点也不会被注册进地图).
4.  LocalMapping线程中创建新地图点的步骤(LocalMapping::CreateNewMapPoints())
    会将当前关键帧与前一关键帧进行匹配,生成新地图点.

- 删除MapPoint的时机:
1.  LocalMapping线程中删除恶劣地图点的步骤(LocalMapping::MapPointCulling()).
2.  删除关键帧的函数KeyFrame::SetBadFlag()会调用函数MapPoint::EraseObservation()
    删除地图点对关键帧的观测,若地图点对关键帧的观测少于2,则地图点无法被三角化,就删除该地图点.

- 替换MapPoint的时机:
1.  LoopClosing线程中闭环矫正(LoopClosing::CorrectLoop())时当前关键帧和闭环关键帧上的地图点发生冲突时,
    会使用闭环关键帧的地图点替换当前关键帧的地图点.
2.  LoopClosing线程中闭环矫正函数LoopClosing::CorrectLoop()会调用LoopClosing::SearchAndFuse()
    将闭环关键帧的共视关键帧组中所有地图点投影到当前关键帧的共视关键帧组中,发生冲突时就会替换.
*/

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <mutex>
#include <opencv2/core/core.hpp>

#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"

namespace ORB_SLAM2
{

    class KeyFrame;
    class Map;
    class Frame;

    /**
     * @brief MapPoint是一个地图点
     */
    class MapPoint
    {
    public:
        /**
         * @brief 给定坐标与keyframe构造MapPoint
         * @details 被调用:
         * 双目：StereoInitialization()，CreateNewKeyFrame()，LocalMapping::CreateNewMapPoints()
         * 单目：CreateInitialMapMonocular()，LocalMapping::CreateNewMapPoints()
         *
         * @param[in] Pos       MapPoint的坐标（wrt世界坐标系）
         * @param[in] pRefKF    生成地图点的关键帧
         * @param[in] pMap      地图点所存在的地图
         */
        MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap);

        /**
         * @brief 给定坐标与frame构造MapPoint
         * @details 被双目：UpdateLastFrame()调用
         * @param[in] Pos       MapPoint的坐标（wrt世界坐标系）
         * @param[in] pMap      Map
         * @param[in] pFrame    Frame
         * @param[in] idxF      MapPoint在Frame中的索引，即对应的特征点的编号
         */
        MapPoint(const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF);

        /**
         * @brief 设置世界坐标系下地图点的位姿
         * @param[in] Pos 世界坐标系下地图点的位姿
         */
        void SetWorldPos(const cv::Mat &Pos);
        /**
         * @brief 获取当前地图点在世界坐标系下的位置
         * @return cv::Mat 位置
         */
        cv::Mat GetWorldPos();

        /**
         * @brief 获取当前地图点的平均观测方向
         * @return cv::Mat 一个向量
         */
        cv::Mat GetNormal();

        /**
         * @brief 获取生成当前地图点的参考关键帧
         * @return KeyFrame*
         */
        KeyFrame *GetReferenceKeyFrame();

        /**
         * @brief 获取观测到当前地图点的关键帧
         * @return std::map<KeyFrame*, size_t>
         *       观测到当前地图点的关键帧序列；
         *       size_t 这个对象对应为该地图点在该关键帧的特征点的访问id（特征点索引号，上限是N-1）
         */
        std::map<KeyFrame *, size_t> GetObservations();

        // 获取当前地图点的被观测次数
        int Observations();

        /**
         * @brief 添加观测
         * @details 记录哪些KeyFrame的哪个特征点能观测到该MapPoint
         *          并增加观测的相机数目nObs，单目+1，双目或者RGBD+2
         *          这个函数是建立关键帧共视关系的核心函数，
         *          能共同观测到某些MapPoints的关键帧是共视关键帧
         *
         * @param[in] pKF KeyFrame,观测到当前地图点的关键帧
         * @param[in] idx MapPoint在KeyFrame中的索引
         */
        void AddObservation(KeyFrame *pKF, size_t idx);

        /**
         * @brief   取消某个关键帧对当前地图点的观测
         * @details 如果某个关键帧要被删除，那么会发生这个操作
         * @param[in] pKF
         */
        void EraseObservation(KeyFrame *pKF);

        /**
         * @brief 获取观测到当前地图点的关键帧,在观测数据中的索引
         *        当前地图点是给定的KF的第几号地图点
         * @param[in] pKF   关键帧
         * @return int      索引
         */
        int GetIndexInKeyFrame(KeyFrame *pKF);

        /**
         * @brief 查看某个关键帧是否看到了当前的地图点
         *
         * @param[in] pKF   关键帧
         * @return bool
         */
        bool IsInKeyFrame(KeyFrame *pKF);

        /**
         * @brief 删除当前地图点 告知可以观测到该MapPoint的Frame，该MapPoint已被删除
         */
        void SetBadFlag();

        /**
         * @brief   判断当前地图点是否是bad
         * @details 查询当前地图点是否被删除(本质上就是查询mbBad)
         * @return bool
         */
        bool isBad();

        /**
         * @brief   使用地图点pMP替换当前地图点
         * @details 在形成闭环的时候，会更新 KeyFrame 与 MapPoint 之间的关系
         *          其实也就是相互替换?
         * @param[in] pMP 地图点
         */
        void Replace(MapPoint *pMP);

        /**
         * @brief 获取取代当前地图点的点?
         * @return MapPoint*
         */
        MapPoint *GetReplaced();

        /**
         * @brief 增加可视次数
         * @details Visible表示：
         * \n 1. 该MapPoint在某些帧的视野范围内，通过Frame::isInFrustum()函数判断
         * \n 2. 该MapPoint被这些帧观测到，但并不一定能和这些帧的特征点匹配上
         * \n   例如：有一个MapPoint（记为M），在某一帧F的视野范围内，
         *      但并不表明该点M可以和F这一帧的某个特征点能匹配上
         * @param[in] n 要增加的次数
         */
        void IncreaseVisible(int n = 1);

        /**
         * @brief Increase Found
         * @details 能找到该点的帧数+n，n默认为1
         * @param[in] n 增加的个数
         * @see Tracking::TrackLocalMap()
         */
        void IncreaseFound(int n = 1);

        //? 这个比例是?
        // 计算被找到的比例 mnFound / mnVisible
        float GetFoundRatio();

        /**
         * @brief 获取被找到的次数
         * @return int 被找到的次数
         */
        inline int GetFound() { return mnFound; }

        /**
         * @brief 计算地图点最具代表性的描述子
         * @details
         * 由于一个地图点会被许多相机观测到，因此在插入关键帧后，需要判断是否更新代表当前点的描述子
         * 先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
         * @see III - C3 .3
         *
         * 特征描述子的更新时机:
         * 一旦某地图点对关键帧的观测mObservations发生改变,
         * 就调用函数MapPoint::ComputeDistinctiveDescriptors()更新该地图点的特征描述子.
         *
         * 特征描述子的用途:
         * 在函数ORBmatcher::SearchByProjection()和ORBmatcher::Fuse()中,
         * 通过比较地图点的特征描述子与图片特征点描述子,实现将地图点与图像特征点的匹配(3D-2D匹配).
         */
        void ComputeDistinctiveDescriptors();

        /**
         * @brief 获取当前地图点的描述子
         * @return cv::Mat
         */
        cv::Mat GetDescriptor();

        /**
         * @brief 更新平均观测方向以及观测距离范围
         *
         * 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要更新相应变量
         * @see III - C2.2 c2.4
         */
        void UpdateNormalAndDepth();

        //? 获取最小距离不变性
        float GetMinDistanceInvariance();
        float GetMaxDistanceInvariance();

        //? 尺度预测?
        /**
         * @brief 预测地图点对应特征点所在的图像金字塔尺度层数
         *
         * @param[in] currentDist   相机光心距离地图点距离
         * @param[in] pKF           关键帧
         * @return int              预测的金字塔层数
         */
        // 根据某地图点到某帧的观测深度 估计其在该帧图片上的层级,是UpdateNormalAndDepth()后半段的逆运算.
        int PredictScale(const float &currentDist, KeyFrame *pKF);
        int PredictScale(const float &currentDist, Frame *pF);

    public:
        long unsigned int mnId; ///< Global ID for MapPoint
        static long unsigned int nNextId;
        const long int mnFirstKFid; ///< 创建该MapPoint的关键帧ID
        //呐,如果是从帧中创建的话,会将普通帧的id存放于这里
        const long int mnFirstFrame; ///< 创建该MapPoint的帧ID（即每一关键帧有一个帧ID）

        // 当前地图点被多少个关键帧相机观测到了，单目+1，双目或RGB-D则+2
        // TODO: 考虑改为私有变量：因为存在一个公有接口读取这个成员变量，而且加锁了
        int nObs;

        // Variables used by the tracking
        float mTrackProjX;     ///< 当前地图点投影到某帧上后的坐标
        float mTrackProjY;     ///< 当前地图点投影到某帧上后的坐标
        float mTrackProjXR;    ///< 当前地图点投影到某帧上后的坐标(右目)
        int mnTrackScaleLevel; ///< 所处的尺度, 由其他的类进行操作 //?
        float mTrackViewCos;   ///< 被追踪到时,那帧相机看到当前地图点的视角
        // TrackLocalMap - SearchByProjection 中决定是否对该点进行投影的变量
        // NOTICE mbTrackInView==false的点有几种：
        // a
        // 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
        // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影   //?
        // 为什么已经是内点了之后就不需要再进行投影了呢?
        // c
        // 不在当前相机视野中的点（即未通过isInFrustum判断）     //?
        bool mbTrackInView;
        // TrackLocalMap - UpdateLocalPoints中
        // 防止将MapPoints重复添加至mvpLocalMapPoints的标记
        long unsigned int mnTrackReferenceForFrame;

        // TrackLocalMap - SearchLocalPoints 中决定是否进行isInFrustum判断的变量
        // NOTICE mnLastFrameSeen==mCurrentFrame.mnId的点有几种：
        // a
        // 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
        // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
        long unsigned int mnLastFrameSeen;

        // REVIEW 下面的....都没看明白
        //  Variables used by local mapping
        //  local mapping中记录地图点对应当前局部BA的关键帧的mnId。mnBALocalForKF
        //  在map point.h里面也有同名的变量。
        long unsigned int mnBALocalForKF;
        long unsigned int
            mnFuseCandidateForKF; ///< 在局部建图线程中使用,表示被用来进行地图点融合的关键帧(存储的是这个关键帧的id)

        // Variables used by loop closing -- 一般都是为了避免重复操作
        /// 标记当前地图点是作为哪个"当前关键帧"的回环地图点(即回环关键帧上的地图点),在回环检测线程中被调用
        long unsigned int mnLoopPointForKF;
        // 如果这个地图点对应的关键帧参与到了回环检测的过程中,那么在回环检测过程中已经使用了这个关键帧修正只有的位姿来修正了这个地图点,那么这个标志位置位
        long unsigned int mnCorrectedByKF;
        long unsigned int mnCorrectedReference;
        // 全局BA优化后(如果当前地图点参加了的话),这里记录优化后的位姿
        cv::Mat mPosGBA;
        // 如果当前点的位姿参与到了全局BA优化,那么这个变量记录了那个引起全局BA的"当前关键帧"的id
        long unsigned int mnBAGlobalForKF;

        ///全局BA中对当前点进行操作的时候使用的互斥量
        static std::mutex mGlobalMutex;

    protected:
        // Position in absolute coordinates
        cv::Mat mWorldPos; ///< MapPoint在世界坐标系下的坐标

        // Keyframes observing the point and associated index in keyframe
        // 观测到该MapPoint的KF和该MapPoint在KF中的索引
        // std::map是一个key-value结构, key为某个关键帧, value为当前地图点在该关键帧中的索引
        // (是在该关键帧成员变量std::vector<MapPoint*> mvpMapPoints中的索引)
        std::map<KeyFrame *, size_t> mObservations;

        // Mean viewing direction
        // 该MapPoint平均观测方向
        // 用于判断点是否在可视范围内
        cv::Mat mNormalVector;

        // Best descriptor to fast matching
        // 每个3D点也有一个描述子，但是这个3D点可以观测多个二维特征点，从中选择一个最有代表性的
        //通过ComputeDistinctiveDescriptors()得到的最有代表性描述子,距离其它描述子的平均距离最小
        cv::Mat mDescriptor;

        /// Reference KeyFrame
        // 1. 通常情况下MapPoint的参考关键帧就是创建该MapPoint的那个关键帧（构造函数中）
        // 2. 若当前地图点对参考关键帧的观测被删除(EraseObservation(KeyFrame* pKF)),
        //    则取第一个观测到当前地图点的关键帧做参考关键帧.
        KeyFrame *mpRefKF;

        /// Tracking counters
        int mnVisible;
        int mnFound;

        /// Bad flag (we do not currently erase MapPoint from memory)
        bool mbBad;

        // 用来替换当前地图点的新地图点
        MapPoint *mpReplaced;

        /// Scale invariance distances
        // 平均观测距离的下限 / 上限
        // 特征点的观测距离与其在图像金字塔中的图层呈线性关系.
        // 直观上理解, 如果一个图像区域被放大后才能识别出来, 说明该区域的观测深度较深(远).
        float mfMinDistance;
        float mfMaxDistance;

        ///所属的地图
        Map *mpMap;

        ///对当前地图点位姿进行操作的时候的互斥量
        std::mutex mMutexPos;
        ///对当前地图点的特征信息进行操作的时候的互斥量
        std::mutex mMutexFeatures;
    };

} // namespace ORB_SLAM2

#endif // MAPPOINT_H
