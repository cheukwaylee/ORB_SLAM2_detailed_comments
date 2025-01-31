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

#include "MapPoint.h"

#include <mutex>

#include "ORBmatcher.h"

namespace ORB_SLAM2
{

    long unsigned int MapPoint::nNextId = 0;
    mutex MapPoint::mGlobalMutex;

    /**
     * @brief Construct a new Map Point:: Map Point object
     *
     * @param[in] Pos           MapPoint的坐标（世界坐标系）
     * @param[in] pRefKF        关键帧
     * @param[in] pMap          地图
     */
    MapPoint::MapPoint(const cv::Mat &Pos,           //地图点的世界坐标
                       KeyFrame *pRefKF,             //生成地图点的关键帧
                       Map *pMap)                    //地图点所存在的地图
        : mnFirstKFid(pRefKF->mnId),                 //第一次观测/生成它的关键帧 id
          mnFirstFrame(pRefKF->mnFrameId),           //创建该地图点的帧ID(因为关键帧也是帧啊)
          nObs(0),                                   //被观测次数
          mnTrackReferenceForFrame(0),               //放置被重复添加到局部地图点的标记
          mnLastFrameSeen(0),                        //是否决定判断在某个帧视野中的变量
          mnBALocalForKF(0),                         //
          mnFuseCandidateForKF(0),                   //
          mnLoopPointForKF(0),                       //
          mnCorrectedByKF(0),                        //
          mnCorrectedReference(0),                   //
          mnBAGlobalForKF(0),                        //
          mpRefKF(pRefKF),                           //
          mnVisible(1),                              //在帧中的可视次数
          mnFound(1),                                //被找到的次数 和上面的相比要求能够匹配上
          mbBad(false),                              //坏点标记
          mpReplaced(static_cast<MapPoint *>(NULL)), //替换掉当前地图点的点
          mfMinDistance(0),                          //当前地图点在某帧下,可信赖的被找到时其到关键帧光心距离的下界
          mfMaxDistance(0),                          //上界
          mpMap(pMap)                                //从属地图
    {
        Pos.copyTo(mWorldPos);
        //平均观测方向初始化为0
        mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

        // MapPoints can be created from Tracking and Local Mapping. This mutex avoid
        // conflicts with id.
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++; // nNextId 是静态变量
    }

    /**
     * @brief 给定坐标与frame构造MapPoint
     *
     * 双目：UpdateLastFrame()
     * @param Pos    MapPoint的坐标（世界坐标系）
     * @param pMap   Map
     * @param pFrame  Frame
     * @param idxF   MapPoint在Frame中的索引，即对应的特征点的编号
     */
    MapPoint::MapPoint(
        const cv::Mat &Pos,
        Map *pMap,
        Frame *pFrame,
        const int &idxF)
        : mnFirstKFid(-1),
          mnFirstFrame(pFrame->mnId),
          nObs(0),
          mnTrackReferenceForFrame(0),
          mnLastFrameSeen(0),
          mnBALocalForKF(0),
          mnFuseCandidateForKF(0),
          mnLoopPointForKF(0),
          mnCorrectedByKF(0),
          mnCorrectedReference(0),
          mnBAGlobalForKF(0),
          mpRefKF(static_cast<KeyFrame *>(NULL)),
          mnVisible(1),
          mnFound(1),
          mbBad(false),
          mpReplaced(NULL),
          mpMap(pMap)
    {
        Pos.copyTo(mWorldPos);
        cv::Mat Ow = pFrame->GetCameraCenter();
        mNormalVector = mWorldPos - Ow;                          // 世界坐标系下相机到3D点的向量 (当前关键帧的观测方向)
        mNormalVector = mNormalVector / cv::norm(mNormalVector); // 单位化

        //这个算重了吧
        cv::Mat PC = Pos - Ow;
        const float dist = cv::norm(PC); //到相机的距离
        const int level = pFrame->mvKeysUn[idxF].octave;
        const float levelScaleFactor = pFrame->mvScaleFactors[level];
        const int nLevels = pFrame->mnScaleLevels;

        // 另见 PredictScale 函数前的注释
        /* 666,因为在提取特征点的时候,考虑到了图像的尺度问题,
           因此在不同图层上提取得到的特征点,对应着特征点距离相机的远近不同,
           所以在这里生成地图点的时候,也要再对其进行确认
           虽然我们拿不到每个图层之间确定的尺度信息,但是我们有缩放比例这个相对的信息哇
        */
        mfMaxDistance = dist * levelScaleFactor;                             //当前图层的"深度"
        mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1]; //该特征点上一个图层的"深度"

        // 见 mDescriptor 在MapPoint.h中的注释 ==> 其实就是获取这个地图点的描述子
        pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

        // MapPoints can be created from Tracking and Local Mapping.
        // This mutex avoid conflicts with id.
        //? 不太懂,怎么个冲突法?
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++; // nNextId 是静态变量
    }

    //设置地图点在世界坐标系下的坐标
    void MapPoint::SetWorldPos(const cv::Mat &Pos)
    {
        //? 为什么这里多了个线程锁
        unique_lock<mutex> lock2(mGlobalMutex);
        unique_lock<mutex> lock(mMutexPos);
        Pos.copyTo(mWorldPos);
    }
    //获取地图点在世界坐标系下的坐标
    cv::Mat MapPoint::GetWorldPos()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mWorldPos.clone();
    }

    //世界坐标系下地图点被多个相机观测的平均观测方向
    cv::Mat MapPoint::GetNormal()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mNormalVector.clone();
    }

    //获取地图点的参考关键帧
    KeyFrame *MapPoint::GetReferenceKeyFrame()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    /**
     * @brief 添加观测
     * @details 记录哪些KeyFrame的哪个特征点能观测到该MapPoint
     *          并增加观测的相机数目nObs，单目+1，双目或者RGBD+2
     *          这个函数是建立关键帧共视关系的核心函数，
     *          能共同观测到某些MapPoints的关键帧是共视关键帧
     *
     * @param[in] pKF KeyFrame,观测到当前地图点的关键帧
     * @param[in] idx MapPoint在KeyFrame中的索引 这是KF的第几个地图点
     */
    void MapPoint::AddObservation(KeyFrame *pKF, size_t idx)
    {
        // 当前地图点的特征信息进行操作的时候
        unique_lock<mutex> lock(mMutexFeatures);

        // mObservations:观测到该MapPoint的关键帧KF和该MapPoint在KF中的索引
        // 如果已经添加过观测，返回
        /*
         * map::count()是C++ STL中的内置函数，如果在映射容器中存在带有键K的元素，则该函数返回1。
         * 如果容器中不存在键为K的元素，则返回0。
         * 用法: map_name.count(key k)
         */
        if (mObservations.count(pKF))
            return;

        // 如果没有添加过观测，记录下能观测到该MapPoint的KF和该MapPoint在KF中的索引
        mObservations[pKF] = idx;

        // 根据观测形式是单目还是双目更新观测计数变量nObs
        if (pKF->mvuRight[idx] >= 0)
            nObs += 2; // 双目或者rgbd
        else
            nObs++; // 单目
    }

    // 删除某个关键帧对当前地图点的观测
    void MapPoint::EraseObservation(KeyFrame *pKF)
    {
        bool bBad = false;
        { // 这个括号是为了数据锁的作用域
            unique_lock<mutex> lock(mMutexFeatures);
            // 查找这个要删除的观测,根据单目和双目类型的不同从其中删除当前地图点的被观测次数
            if (mObservations.count(pKF))
            {
                int idx = mObservations[pKF];
                if (pKF->mvuRight[idx] >= 0)
                    nObs -= 2;
                else
                    nObs--;

                mObservations.erase(pKF);

                // 如果要删除的keyFrame是参考关键帧，
                // 该key Frame被删除后重新指定RefFrame，把观测到当前地图点的第一个KF作为refKF
                if (mpRefKF == pKF)
                    mpRefKF = mObservations.begin()->first; // ????参考帧指定得这么草率真的好么?

                // If only 2 observations or less, discard point
                // 当观测到该点的相机数目少于2时，丢弃该点(至少需要两个观测才能三角化)
                if (nObs <= 2)
                    bBad = true;
            }
        }

        if (bBad)
            // 告知可以观测到该MapPoint的Frame，该MapPoint已被删除
            SetBadFlag();
    }

    // 能够观测到当前地图点的所有关键帧及该地图点在KF中的索引
    map<KeyFrame *, size_t> MapPoint::GetObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    // 获取被观测到的相机数目，单目+1，双目或RGB-D则+2
    int MapPoint::Observations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }

    /**
     * @brief 告知可以观测到该MapPoint的Frame，该MapPoint已被删除
     *
     * @details 删除地图点的各成员变量是一个较耗时的过程,
     *  因此函数SetBadFlag()删除关键点时采取先标记再清除的方式,具体的删除过程分为以下两步:
     *  1. 先将坏点标记mbBad置为true,逻辑上删除该地图点.(地图点的社会性死亡)
     *  2. 再依次清空当前地图点的各成员变量,物理上删除该地图点.(地图点的肉体死亡)
     */
    void MapPoint::SetBadFlag()
    {
        map<KeyFrame *, size_t> obs;
        { // 数据锁的作用域
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad = true; // 标记mbBad,逻辑上删除当前地图点：轻操作

            // 把mObservations转存到obs，obs和mObservations里存的是指针，赋值过程为浅拷贝
            obs = mObservations;

            // 当前mp的 把mObservations指向的内存释放，obs作为局部变量之后自动删除
            mObservations.clear();
        }

        // 下面就不用加锁了，因为每次用地图点的时候都要检查mbBad
        // 删除关键帧对当前地图点的观测
        for (map<KeyFrame *, size_t>::iterator
                 mit = obs.begin(),
                 mend = obs.end();
             mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            // 告诉可以观测到该MapPoint的KeyFrame，该MapPoint被删了
            pKF->EraseMapPointMatch(mit->second);
        }

        // ? 在地图类上注册删除当前地图点,这里会发生内存泄漏
        // ? （地图类指向地图点的指针被删除了，但是创建的地图点对象没有被释放）
        // 擦除该MapPoint申请的内存 // this 代表当前地图点
        mpMap->EraseMapPoint(this);
    }

    MapPoint *MapPoint::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    /**
     * @brief 替换地图点，更新观测关系
     * @param[in] pMP       用该地图点来替换当前地图点
     */
    void MapPoint::Replace(MapPoint *pMP)
    {
        // 同一个地图点 则跳过
        if (pMP->mnId == this->mnId)
            return;

        //要替换当前地图点,有两个工作:
        // 1. 将当前地图点的观测数据等其他数据都"叠加"到新的地图点上
        // 2. 将观测到当前地图点的关键帧的信息进行更新

        // Step 1 逻辑上删除当前地图点
        // 清除当前地图点的信息，这一段和SetBadFlag函数相同
        int nvisible, nfound;
        map<KeyFrame *, size_t> obs;
        { // 数据锁作用域：轻操作
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            obs = mObservations;
            //清除当前地图点的原有观测
            mObservations.clear();
            //当前的地图点被删除了
            mbBad = true;
            //暂存当前地图点的可视次数和被找到的次数
            nvisible = mnVisible;
            nfound = mnFound;
            //指明当前地图点已经被指定的地图点替换了
            mpReplaced = pMP;
        }

        // Step 2 将当地图点的数据叠加到新地图点上
        // 所有能观测到原地图点的关键帧都要复制到替换的地图点上
        // 将观测到当前地图的的关键帧的信息进行更新 遍历与当前mp关联的所有KF
        for (map<KeyFrame *, size_t>::iterator
                 mit = obs.begin(),
                 mend = obs.end();
             mit != mend; mit++)
        {
            // Replace measurement in keyframe
            KeyFrame *pKF = mit->first;

            if (!pMP->IsInKeyFrame(pKF))
            {
                // 该关键帧中没有对"要替换本地图点的地图点"的观测

                // mit->second 当前mp是KF的第几个mp，把KF的第mit->second号mp换新
                pKF->ReplaceMapPointMatch(mit->second, pMP); // 让KeyFrame用pMP替换掉原来的MapPoint
                // 给新的mp增加观测，新mp被某个KF观测到，是其中第mit->second个mp
                pMP->AddObservation(pKF, mit->second); // 让MapPoint替换掉对应的KeyFrame
            }
            else
            {
                // 这个关键帧对当前的地图点和"要替换本地图点的地图点"都具有观测

                // 产生冲突，即pKF中有两个特征点a,b（这两个特征点的描述子是近似相同的），
                // 这两个特征点对应两个MapPoint 为this,pMP
                // 然而在fuse的过程中pMP的观测更多，需要替换this，因此保留b与pMP的联系，去掉a与this的联系
                //说白了,既然是让对方的那个地图点来代替当前的地图点,
                //就是说明对方更好,所以删除这个关键帧对当前帧的观测
                pKF->EraseMapPointMatch(mit->second); // 把旧的擦掉
            }
        }

        // 将当前地图点的观测数据等其他数据都"叠加"到新的地图点上
        //? 为什么？因为新点必定更好？所以更多
        pMP->IncreaseFound(nfound);
        pMP->IncreaseVisible(nvisible);

        //描述子更新
        pMP->ComputeDistinctiveDescriptors();

        // Step 3 删除当前地图点
        //告知地图,删掉我
        mpMap->EraseMapPoint(this);
    }

    // 没有经过 MapPointCulling 检测的MapPoints, 认为是坏掉的点
    bool MapPoint::isBad()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mbBad;
    }

    /**
     * @brief Increase Visible
     *
     * Visible表示：
     * 1. 该MapPoint在某些帧的视野范围内，通过Frame::isInFrustum()函数判断
     * 2. 该MapPoint被这些帧观测到，但并不一定能和这些帧的特征点匹配上
     *    例如：有一个MapPoint（记为M），在某一帧F的视野范围内，
     *    但并不表明该点M可以和F这一帧的某个特征点能匹配上
     * //?  所以说，found 就是表示匹配上了嘛？
     */
    void MapPoint::IncreaseVisible(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    /**
     * @brief Increase Found
     *
     * 能找到该点的帧数+n，n默认为1
     * @see Tracking::TrackLocalMap()
     */
    void MapPoint::IncreaseFound(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound += n;
    }

    // 计算被找到的比例
    float MapPoint::GetFoundRatio()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound) / mnVisible;
    }

    /**
     * @brief 计算地图点最具代表性的描述子
     * @details
     * 由于一个地图点会被许多相机观测到，因此在插入关键帧后，需要判断是否更新代表当前点的描述子
     * 先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
     */
    void MapPoint::ComputeDistinctiveDescriptors()
    {
        // Retrieve all observed descriptors
        vector<cv::Mat> vDescriptors;

        map<KeyFrame *, size_t> observations;

        // Step 1 获取该地图点所有有效的观测关键帧信息
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            if (mbBad) // 当前地图点被即将被物理删除（当前为软删除状态）
                return;
            observations = mObservations;
        }

        if (observations.empty())
            return;

        // 地图点包含的描述子数目等于地图点的kp数目
        vDescriptors.reserve(observations.size());

        // Step 2
        // 遍历观测到该地图点的所有关键帧，对应的orb描述子，放到向量vDescriptors中
        for (map<KeyFrame *, size_t>::iterator
                 mit = observations.begin(),
                 mend = observations.end();
             mit != mend; mit++)
        {
            // mit->first取观测到该地图点的关键帧
            // mit->second取该地图点在关键帧中的索引 当前mp是KF的第几个mp（也就是第几个kp）
            KeyFrame *pKF = mit->first;

            if (!pKF->isBad())
                // 取对应的描述子向量
                // mDescriptors的第mit->second行 取出KF对应的那个kp的描述子
                vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
        }

        if (vDescriptors.empty()) // observations里面的KF全都是bad
            return;

        // Compute distances between them
        // Step 3 计算这些描述子两两之间的距离
        // N表示为一共多少个描述子
        const size_t N = vDescriptors.size();

        // 将Distances表述成一个对称的矩阵
        // float Distances[N][N];
        std::vector<std::vector<float>> Distances;
        // 第一维N个元素 第二维vector<float>(N, 0)
        Distances.resize(N, vector<float>(N, 0));

        for (size_t i = 0; i < N; i++)
        {
            // 和自己的距离当然是0
            Distances[i][i] = 0;
            // 计算并记录不同描述子距离
            for (size_t j = i + 1; j < N; j++)
            {
                int distij =
                    ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
                Distances[i][j] = distij;
                Distances[j][i] = distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        // Step 4 选择最有代表性的描述子，它与其他描述子应该具有最小的距离中值
        int BestMedian = INT_MAX; // 记录最小的中值
        int BestIdx = 0;          // 最小中值对应的索引
        for (size_t i = 0; i < N; i++)
        {
            // 第i个描述子到其它所有描述子之间的距离
            // vector<int> vDists(Distances[i],Distances[i]+N);
            vector<int> vDists(Distances[i].begin(), Distances[i].end());
            sort(vDists.begin(), vDists.end());

            // 获得中值
            int median = vDists[0.5 * (N - 1)];

            // 寻找最小的中值
            if (median < BestMedian)
            {
                BestMedian = median;
                BestIdx = i;
            }
        }

        {
            unique_lock<mutex> lock(mMutexFeatures);
            mDescriptor = vDescriptors[BestIdx].clone();
        }
    }

    // 获取当前地图点的描述子
    cv::Mat MapPoint::GetDescriptor()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mDescriptor.clone();
    }

    //获取当前地图点在某个关键帧的观测中，对应的特征点的ID
    //当前mp是给定KF里面的第几个mp
    int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return mObservations[pKF];
        else
            return -1;
    }

    /**
     * @brief check MapPoint is in keyframe
     * @param  pKF KeyFrame
     * @return     true if in pKF
     */

    /**
     * @brief 检查该地图点是否在关键帧中（有对应的二维特征点）
     *
     * @param[in] pKF       关键帧
     * @return true         如果能够观测到，返回true
     * @return false        如果观测不到，返回false
     */
    bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        // 存在返回true，不存在返回false
        // std::map.count 用法见：http://www.cplusplus.com/reference/map/map/count/
        return (mObservations.count(pKF));
    }

    /**
     * @brief 更新地图点的平均观测方向、观测距离范围
     * @details 其中平均观测方向是根据mObservations中所有观测到本地图点的关键帧取平均得到的;
     *          平均观测距离是根据参考关键帧得到的.
     * 调用时机:
     * 1. 创建地图点时调用UpdateNormalAndDepth()初始化其观测信息.
     * 2. 地图点对关键帧的观测mObservations更新时(跟踪局部地图添加或删除对关键帧的观测时、
     *    LocalMapping线程删除冗余关键帧时或**LoopClosing线程闭环矫正**时),
     *    调用UpdateNormalAndDepth()初始化其观测信息.
     * 3. 地图点世界坐标mWorldPos发生变化时(BA优化之后),调用UpdateNormalAndDepth()初始化其观测信息.
     *
     * 总结成一句话: 只要地图点本身或关键帧对该地图点的观测发生变化,就应该调用函数
     */
    void MapPoint::UpdateNormalAndDepth()
    {
        // Step 1 获得观测到该地图点的所有关键帧、坐标等信息
        map<KeyFrame *, size_t> observations;
        KeyFrame *pRefKF;
        cv::Mat Pos;

        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            if (mbBad)
                return;

            observations = mObservations; // 获得观测到该地图点的所有关键帧
            pRefKF = mpRefKF;             // 观测到该点的参考关键帧（第一次创建时的关键帧）
            Pos = mWorldPos.clone();      // 地图点在世界坐标系中的位置
        }

        // 当前地图点没有被观测到
        if (observations.empty())
            return;

        // Step 2 计算该地图点的平均观测方向
        // 能观测到该地图点的所有关键帧，对该点的观测方向归一化为单位向量，然后进行求和得到该地图点的朝向
        // 初始值为0向量，累加为归一化向量，最后除以总数n
        cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
        int n = 0;
        for (map<KeyFrame *, size_t>::iterator
                 mit = observations.begin(),
                 mend = observations.end();
             mit != mend; mit++)
        {
            // 观测到当前地图点的KF
            KeyFrame *pKF = mit->first;
            // 获取(左目)相机的中心在世界坐标系下的坐标，KF的相机坐标wrt world
            cv::Mat Owi = pKF->GetCameraCenter();
            // 获得地图点和观测到它关键帧的向量并归一化
            // (地图点wrt KF相机坐标) =  (地图点wrt world) - (KF相机坐标 wrt world)
            cv::Mat normali = mWorldPos - Owi;
            // 累加求和：当前地图点被多个KF观测到
            normal = normal + normali / cv::norm(normali);
            n++;
        }
        // refKF的camera 指向mappoint
        cv::Mat PC = Pos - pRefKF->GetCameraCenter(); // 参考关键帧相机指向地图点的向量（在世界坐标系下的表示）
        const float dist = cv::norm(PC);              // 该点到参考关键帧相机的距离

        // 观测到该地图点的当前帧的特征点在金字塔的第几层
        // observations[pRefKF] 返回当前地图点是refKF的第几个地图点（也就是第几个keypoint，只有kp可以mp）
        // 取出refKF的这个keypoint 然后获取它的金字塔层数
        const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
        // 当前金字塔层对应的尺度因子，scale^n，scale=1.2，n为层数
        const float levelScaleFactor = pRefKF->mvScaleFactors[level];
        // 金字塔总层数，默认为8
        const int nLevels = pRefKF->mnScaleLevels;

        { // 最终目的
            unique_lock<mutex> lock3(mMutexPos);
            // 使用方法见PredictScale函数前的注释

            // mfMaxDistance表示若地图点匹配在某特征提取器图像金字塔第7层上的某特征点,观测距离值
            // mfMinDistance表示若地图点匹配在某特征提取器图像金字塔第0层上的某特征点,观测距离值
            // refKF相机到mp的距离 * refKF提取到当前mp的kp的金字塔层的缩放系数
            mfMaxDistance = dist * levelScaleFactor;                             // 观测到该点的距离上限 最深
            mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1]; // 观测到该点的距离下限 最浅
            // 获得地图点平均的观测方向
            mNormalVector = normal / n;
        }
    }

    float MapPoint::GetMinDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 0.8f * mfMinDistance;
    }

    float MapPoint::GetMaxDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 1.2f * mfMaxDistance;
    }

    // 下图中横线的大小表示不同图层图像上的一个像素表示的真实物理空间中的大小
    //              ____
    // Nearer      /____\     level:n-1 --> dmin
    //            /______\                       d/dmin = 1.2^(n-1-m)
    //           /________\   level:m   --> d
    //          /__________\                     dmax/d = 1.2^m
    // Farther /____________\ level:0   --> dmax
    //
    //           log(dmax/d)
    // m = ceil(------------)
    //            log(1.2)
    // 这个函数的作用:
    // 在进行投影匹配的时候会给定特征点的搜索范围,考虑到处于不同尺度
    // (也就是距离相机远近,位于图像金字塔中不同图层)的特征点受到相机旋转的影响不同,
    // 因此会希望距离相机近的点的搜索范围更大一点,距离相机更远的点的搜索范围更小一点,
    // 所以要在这里,根据点到关键帧/帧的距离来估计它在当前的关键帧/帧中,
    // 会大概处于哪个尺度
    /**
     * @brief 预测地图点对应特征点所在的图像金字塔尺度层数
     *
     * @param[in] currentDist   相机光心距离地图点距离
     * @param[in] pKF           关键帧
     * @return int              预测的金字塔尺度
     */
    int MapPoint::PredictScale(const float &currentDist, KeyFrame *pKF)
    {
        float ratio;
        {
            unique_lock<mutex> lock(mMutexPos);
            // mfMaxDistance = ref_dist*levelScaleFactor 为参考帧考虑上尺度后的距离
            // ratio = mfMaxDistance/currentDist = ref_dist/cur_dist
            ratio = mfMaxDistance / currentDist;
        }

        // 取对数
        // mfMaxDistance = currentDist * ScaleFactor^level
        int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);

        // 限制范围
        if (nScale < 0)
            nScale = 0;
        else if (nScale >= pKF->mnScaleLevels)
            nScale = pKF->mnScaleLevels - 1;

        return nScale;
    }

    /**
     * @brief 同上 根据地图点到光心的距离来预测一个类似特征金字塔的尺度
     *
     * @param[in] currentDist       地图点到光心的距离
     * @param[in] pF                当前帧
     * @return int                  尺度
     */
    int MapPoint::PredictScale(const float &currentDist, Frame *pF)
    {
        float ratio;
        {
            unique_lock<mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }

        int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
        if (nScale < 0)
            nScale = 0;
        else if (nScale >= pF->mnScaleLevels)
            nScale = pF->mnScaleLevels - 1;

        return nScale;
    }

} // namespace ORB_SLAM2
