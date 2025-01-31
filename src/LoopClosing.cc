/**
 * @file LoopClosing.cc
 * @author guoqing (1337841346@qq.com)
 * @brief 回环检测线程
 * @version 0.1
 * @date 2019-05-05
 *
 * @copyright Copyright (c) 2019
 *
 */

/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
 * For more information see <https://github.com/raulmur/ORB_SLAM2>
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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include <mutex>
#include <thread>

namespace ORB_SLAM2
{

    /**
     * @brief 构造函数
     * @param[in] pMap    地图指针
     * @param[in] pDB     词袋数据库
     * @param[in] pVoc    词典
     * @param[in] bFixScale  表示sim3中的尺度是否要计算,
     *              对于双目和RGBD情况尺度是固定的,s=1,bFixScale=true;
     *              而单目下尺度是不确定的,此时bFixScale=false,sim3中的s需要被计算
     */
    LoopClosing::LoopClosing(
        Map *pMap,
        KeyFrameDatabase *pDB,
        ORBVocabulary *pVoc,
        const bool bFixScale)
        : mbResetRequested(false),
          mbFinishRequested(false), mbFinished(true),
          mpMap(pMap),
          mpKeyFrameDB(pDB), mpORBVocabulary(pVoc),
          mpMatchedKF(NULL), mLastLoopKFid(0),
          mbRunningGBA(false), mbFinishedGBA(true),
          mbStopGBA(false),
          mpThreadGBA(NULL),
          mbFixScale(bFixScale),
          mnFullBAIdx(0)
    {
        // 连续性阈值
        mnCovisibilityConsistencyTh = 3;
    }

    // 设置追踪线程句柄
    void LoopClosing::SetTracker(Tracking *pTracker)
    {
        mpTracker = pTracker;
    }
    // 设置局部建图线程的句柄
    void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
    {
        mpLocalMapper = pLocalMapper;
    }

    // LoopClosing回环线程主函数
    void LoopClosing::Run()
    {
        mbFinished = false;

        // 线程主循环
        while (1)
        {
            // Check if there are keyframes in the queue
            // LoopClosing中的关键帧是LocalMapping发送过来的，LocalMapping是Tracking中发过来的
            // 在LocalMapping中通过 InsertKeyFrame 将关键帧插入闭环检测队列 mlpLoopKeyFrameQueue
            // Step 1 查看闭环检测队列mlpLoopKeyFrameQueue中有没有关键帧进来
            if (CheckNewKeyFrames())
            {
                // Detect loop candidates and check covisibility consistency
                if (DetectLoop())
                {
                    // Compute similarity transformation [sR|t]
                    // In the stereo/RGBD case s=1
                    if (ComputeSim3())
                    {
                        // Perform loop fusion and pose graph optimization
                        CorrectLoop();
                    }
                }
            }

            // 查看是否有外部线程请求复位当前线程
            ResetIfRequested();

            // 查看外部线程是否有终止当前线程的请求,如果有的话就跳出这个线程的主函数的主循环
            if (CheckFinish())
                break;

            // 线程暂停5毫秒,5毫秒结束后再从while(1)循环首部运行 usleep(5000);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        // 运行到这里说明有外部线程请求终止当前线程,在这个函数中执行终止当前线程的一些操作
        SetFinish();
    }

    // 将某个关键帧加入到回环检测的过程中,由局部建图线程（上一级）调用
    void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        // 注意：这里第0个关键帧不能够参与到回环检测的过程中,因为第0关键帧定义了整个地图的世界坐标系
        if (pKF->mnId != 0)
            mlpLoopKeyFrameQueue.push_back(pKF);
    }

    /**
     * @brief  查看列表中是否有等待被插入的关键帧
     * @return  如果存在，返回true
     */
    bool LoopClosing::CheckNewKeyFrames()
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        return (!mlpLoopKeyFrameQueue.empty());
    }

    /**
     * @brief  闭环检测
     * @details 闭环检测原理: 若连续4个关键帧都能在数据库中找到对应的闭环匹配关键帧组,且这些闭环匹配关键帧组间是连续的,则认为实现闭环
     *
     * @return true   成功检测到闭环
     * @return false   未检测到闭环
     */
    bool LoopClosing::DetectLoop()
    {
        // Step 1：取出缓冲队列头部的关键帧,作为当前检测闭环关键帧,设置其不被优化删除
        {
            unique_lock<mutex> lock(mMutexLoopQueue);

            // 从队列头开始取，也就是先取早进来的关键帧 FIFO
            mpCurrentKF = mlpLoopKeyFrameQueue.front();
            // 取出关键帧后从队列里弹出该关键帧
            mlpLoopKeyFrameQueue.pop_front();

            // Avoid that a keyframe can be erased while it is being process by this thread
            // 设置当前关键帧不要在优化的过程中被删除
            mpCurrentKF->SetNotErase();
        }

        // If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
        // Step 2：如果距离上次闭环时间太短（小于10帧），或者map中关键帧总共还没有10帧，则不进行闭环检测
        // （后者的体现是当mLastLoopKFid为0的时候）
        if (mpCurrentKF->mnId < mLastLoopKFid + 10)
        {
            mpKeyFrameDB->add(mpCurrentKF);
            mpCurrentKF->SetErase();
            return false;
        }

        // Compute reference BoW similarity score
        // This is the lowest score to a connected keyframe in the covisibility graph
        // We will impose loop candidates to have a higher similarity than this
        // Step 3：遍历当前回环关键帧所有连接（>15个共视地图点）关键帧，计算当前关键帧与每个共视关键的bow相似度得分，并得到最低得分minScore（作为后续的下限基准）
        const vector<KeyFrame *> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames(); // 取出当前要检测回环的当前KF的（直接连接的）共视关键帧
        const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;                               // KF的字典（单词及其权重）
        float minScore = 1;
        for (size_t i = 0; i < vpConnectedKeyFrames.size(); i++) // 遍历共视关键帧
        {
            KeyFrame *pKF = vpConnectedKeyFrames[i];
            if (pKF->isBad())
                continue;
            const DBoW2::BowVector &BowVec = pKF->mBowVec;

            // 计算两个关键帧的相似度得分；得分越低,相似度越低
            float score = mpORBVocabulary->score(CurrentBowVec, BowVec);
            // 更新最低得分
            if (score < minScore)
                minScore = score;
        }

        // Query the database imposing the minimum score
        // Step 4：基于当前的关键帧 在所有关键帧数据库中找出 当前关键帧的闭环候选关键帧[vpCandidateKFs]（注意不和当前帧连接）
        // 闭环候选关键帧[vpCandidateKFs] 取自于 与当前关键帧具有相同的BOW向量 && 但不存在直接连接的关键帧.
        // minScore的作用：认为和当前关键帧具有回环关系的关键帧：
        //         不应该低于当前关键帧的相邻关键帧们（直接相连）的最低的相似度minScore得到的这些关键帧,
        //         （和当前关键帧具有较多的公共单词,并且相似度评分都挺高：才有资格入围成为候选回环关键帧）
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

        // If there are no loop candidates, just add new keyframe and return false
        // 如果没有闭环候选帧，返回false
        if (vpCandidateKFs.empty())
        {
            mpKeyFrameDB->add(mpCurrentKF);
            mvConsistentGroups.clear();
            mpCurrentKF->SetErase();
            return false;
        }

        // For each loop candidate [vpCandidateKFs], check consistency with previous loop candidates
        // Each candidate [each KF in vpCandidateKFs] expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
        // A group is consistent with a previous group if they share at least a keyframe // group一致的条件是：有共同的KF
        // We must detect a consistent loop in several consecutive keyframes to accept it // 只有在连续的KF检测到一致的回环才能认为产生了回环
        // Step 5：在候选帧中检测具有连续性的候选帧 //在当前关键帧组 和 之前的连续关键帧组之间寻找匹配
        // 1、每个候选帧[each KF in vpCandidateKFs] 将与自己相连的关键帧[->GetConnectedKeyFrames()] 构成一个“子候选组[spCandidateGroup]”
        // 2、检测“子候选组[spCandidateGroup]”中每一个关键帧是否存在于“连续组”，如果存在
        //    则nCurrentConsistency++，并将该“子候选组[spCandidateGroup]”放入“当前连续组[vCurrentConsistentGroups]”
        // 3、如果[nCurrentConsistency]大于等于3，那么该“子候选组[spCandidateGroup]”代表的候选帧过关，进入[mvpEnoughConsistentCandidates]

        /** 相关的概念说明:（为方便理解，见视频里的图示）
         * 组(group): 对于某个关键帧KF, 它和它具有共视关系的（直接相连）关键帧组成了一个"组";
         * 子候选组(CandidateGroup): 对于某个候选的回环关键帧[each KF in vpCandidateKFs], 它和它具有共视关系的关键帧组成的一个"组";
         * 连续(Consistent): 不同的组之间如果共同拥有一个及以上的关键帧KF,那么称这两个组之间具有连续关系
         * 连续性(Consistency): 称之为连续长度可能更合适,表示累计的连续的链的长度:A--B 为1, A--B--C--D 为3 等;
         *            具体反映在数据类型[ConsistentGroup.second]上
         * 连续组(Consistent group): [mvConsistentGroups] 它的数据类型是[std::vector<ConsistentGroup>]
         *               存储了上次执行回环检测时, 新的被检测出来的 具有连续性的多个组的集合.
         *               由于组之间的连续关系是个网状结构,
         *               因此可能存在一个组：因为和不同的连续组链都具有连续关系,而被添加两次的情况(当然连续性度量是不相同的)
         *               也就是[ConsistentGroup][pair<set<KeyFrame *>, int>]里面的 set可能出现多次，但是它对应的int有不同的数值
         * 连续组链：自造的称呼,类似于菊花链A--B--C--D这样形成了一条连续组链.
         *      对于这个例子中,由于可能E,F都和D有连续关系,因此连续组链会产生分叉;
         *      为了简化计算,连续组中将只会保存 最后形成连续关系的连续组们(见下面的连续组的更新)
         * 子连续组：上面的连续组中的一个组[each ConsistentGroup in mvConsistentGroups]
         *      数据类型是[std::vector<ConsistentGroup>]的[mvConsistentGroups]里面的一个元素
         * 连续组的初始值： 在遍历某个候选帧[each KF in vpCandidateKFs]的过程中,如果该子候选组[spCandidateGroup]没有能够和任何一个上次的子连续组产生连续关系,
         *          那么就将添加自己组为连续组,并且连续性为0 (相当于新开了一个连续链)
         * 连续组的更新： 当前次回环检测过程中, 所有被检测到和之前的连续组链有连续的关系的组, 都将在对应的连续组链后面+1,
         *         这些子候选组(可能有重复,见上)都将会成为新的连续组; 换而言之 连续组[mvConsistentGroups]中只保存连续组链中末尾的组
         *                                        （链的 最后一个元素（组） 及其 这条链连续的长度）
         */

        // 最终筛选后得到的闭环帧
        mvpEnoughConsistentCandidates.clear();

        // [ConsistentGroup]数据类型为 pair< set<KeyFrame*>, int >
        // [ConsistentGroup.first]对应每个“连续组”中的关键帧集合，
        // [ConsistentGroup.second]为每个“连续组”已连续几个的序号 // 这个组的连续的链的长度
        // 将 闭环候选关键帧（当前KF不直接相连的高分） 和 其共视关键帧（直接相连） 组合成为关键帧组： 当前关键帧的闭环候选关键帧组
        vector<ConsistentGroup> vCurrentConsistentGroups;

        // 这个下标是每个"子连续组[ConsistentGroup]"的下标, （和[mvConsistentGroups]的顺序对齐）
        // bool表示当前的候选组[vpCandidateKFs]中是否有和该组相同的一个关键帧（候选闭环KF是否存在于某个组里面）
        vector<bool> vbConsistentGroup(mvConsistentGroups.size(), false);

        // Step 5.1：遍历刚才得到的每一个候选关键帧
        for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++)
        {
            KeyFrame *pCandidateKF = vpCandidateKFs[i];

            // Step 5.2：将 自己（其中一个候选KF）和自己直接相连的关键帧 构成一个“子候选组”
            set<KeyFrame *> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
            // 把自己也加进去
            spCandidateGroup.insert(pCandidateKF);

            // 连续性达标的标志
            bool bEnoughConsistent = false;
            bool bConsistentForSomeGroup = false;

            // Step 5.3：遍历前一次闭环检测到的连续组链
            // 上一次闭环的连续组链 std::vector<ConsistentGroup> mvConsistentGroups
            // 其中[ConsistentGroup]的定义：typedef pair<set<KeyFrame*>,int> ConsistentGroup
            // 其中 ConsistentGroup.first对应每个“连续组”中的关键帧集合，ConsistentGroup.second为每个“连续组”的连续长度
            for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; iG++)
            {
                // 取出之前的一个 子连续组中的关键帧集合
                set<KeyFrame *> sPreviousGroup = mvConsistentGroups[iG].first;

                // Step 5.4：遍历每个“子候选组”，检测“子候选组”中每一个关键帧在“子连续组”中是否存在
                // 如果有一帧共同存在于“子候选组[spCandidateGroup]”与之前的“子连续组[each ConsistentGroup in mvConsistentGroups]”，
                // 那么“子候选组”（新东西）与该“子连续组”（旧东西）连续
                bool bConsistent = false;
                for (set<KeyFrame *>::iterator
                         sit = spCandidateGroup.begin(),
                         send = spCandidateGroup.end();
                     sit != send; sit++) // 遍历spCandidateGroup集合里面的KF
                {
                    if (sPreviousGroup.count(*sit)) // spCandidateGroup集合里面的某个KF 存在于 “子连续组[mvConsistentGroups]”
                    {
                        // 如果存在，该“子候选组”与该“子连续组”相连
                        bConsistent = true;
                        // 该“子候选组”至少与一个“子连续组”相连（初始化在遍历“连续组链”外面）
                        bConsistentForSomeGroup = true;

                        // 跳出遍历“子候选组”的循环
                        break;
                    }
                }

                if (bConsistent) // 该“子候选组”（新东西）与该“子连续组”（旧东西）
                {
                    // Step 5.5：如果判定为连续，接下来判断是否达到连续的条件
                    // 取出和当前的候选组发生"连续"关系的子连续组的"已连续次数"
                    int nPreviousConsistency = mvConsistentGroups[iG].second;
                    // 将当前候选组连续长度在原子连续组的基础上 +1，
                    int nCurrentConsistency = nPreviousConsistency + 1;

                    // 如果上述连续关系还未记录到[vCurrentConsistentGroups]，那么记录一下
                    // 注意这里[spCandidateGroup]可能放置在[vbConsistentGroup]中其他索引(iG)下
                    //? 因为一对多的网状结构？同一个的链尾元素，不同的链长
                    if (!vbConsistentGroup[iG])
                    {
                        // 将该“子候选组”的该关键帧打上连续编号加入到“当前连续组”
                        ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
                        // 放入本次闭环检测的连续组[vCurrentConsistentGroups]里
                        vCurrentConsistentGroups.push_back(cg);

                        // this avoid to include the same group more than once
                        // 标记一下，防止重复添加到同一个索引iG
                        // 但是[spCandidateGroup]可能重复添加到不同的索引iG对应的[vbConsistentGroup]中
                        vbConsistentGroup[iG] = true;
                    }

                    // 如果连续长度满足要求，那么当前的这个候选关键帧[pCandidateKF]是足够靠谱的
                    // 连续性阈值 mnCovisibilityConsistencyTh=3
                    // 足够连续的标记 bEnoughConsistent
                    if (nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
                    {
                        // 记录为达到连续条件了
                        mvpEnoughConsistentCandidates.push_back(pCandidateKF); // 最终筛选后得到的闭环帧
                        // this avoid to insert the same candidate more than once
                        //  标记一下，防止重复添加
                        bEnoughConsistent = true;

                        // ? 这里可以break掉结束当前for循环吗？
                        // 回答：不行。因为虽然[pCandidateKF]达到了连续性要求
                        // 但是“子候选组[spCandidateGroup]”（新东西） 还可以和
                        //   “子连续组[mvConsistentGroups]”（旧东西）中其他的子连续组进行连接
                        //   为的应该是更新连续组 [vCurrentConsistentGroups] 为下次做好准备
                    }
                }
            }

            // If the group is not consistent with any previous group, insert with consistency counter set to zero
            // Step 5.6：如果 该“子候选组”的所有关键帧 都和 上次闭环“子连续组” 无关（不连续），即[vCurrentConsistentGroups]没有新添加连续关系
            // 于是就把“子候选组”全部拷贝到[vCurrentConsistentGroups]，连续性计数器设为0。用于更新[mvConsistentGroups]（为下次做好准备）
            if (!bConsistentForSomeGroup)
            {
                ConsistentGroup cg = make_pair(spCandidateGroup, 0); // 初值
                vCurrentConsistentGroups.push_back(cg);
            }
        } // 遍历得到的初级的候选关键帧

        // Update Covisibility Consistent Groups
        // 更新连续组
        mvConsistentGroups = vCurrentConsistentGroups;

        // Add Current Keyframe to database
        // 当前闭环检测的关键帧添加到关键帧数据库中
        mpKeyFrameDB->add(mpCurrentKF);

        if (mvpEnoughConsistentCandidates.empty())
        {
            // 未检测到闭环，返回false
            mpCurrentKF->SetErase();
            return false;
        }
        else
        {
            // 成功检测到闭环，返回true
            return true;
        }

        // 多余的代码,执行不到
        mpCurrentKF->SetErase();
        return false;
    }

    /**
     * @brief 计算当前关键帧和上一步闭环候选帧的Sim3变换
     *
     * 找出可能的闭环关键帧[mpMatchedKF]
     * 1. 遍历闭环候选帧集[mvpEnoughConsistentCandidates]，里面的元素叫做[闭环帧]
     *   筛选出与当前帧[mpCurrentKF]的匹配特征点数大于20的候选帧集合，并为每一个候选帧构造一个Sim3Solver
     * 2. 对每一个候选帧进行 Sim3Solver 迭代匹配，直到有一个候选帧匹配成功，或者全部失败
     * 根据求得的闭环关键帧[mpMatchedKF]进行验证
     * 3. 取出闭环匹配上关键帧的相连关键帧，得到它们的地图点放入 mvpLoopMapPoints
     * 4. 将闭环匹配上关键帧以及相连关键帧的地图点投影到当前关键帧进行投影匹配
     * 5. 判断当前帧与检测出的所有闭环关键帧是否有足够多的地图点匹配
     * 6. 清空mvpEnoughConsistentCandidates
     *
     * @return true  只要有一个候选关键帧通过Sim3的求解与优化，就返回true
     * @return false  所有候选关键帧与当前关键帧都没有有效Sim3变换
     */
    bool LoopClosing::ComputeSim3()
    {
        // Sim3 计算流程说明：
        // 1. 通过Bow加速描述子的匹配，利用RANSAC粗略地计算出当前帧与闭环帧的Sim3（当前帧---闭环帧）
        // 2. 根据估计的Sim3，对3D点进行投影找到更多匹配，通过优化的方法计算更精确的Sim3（当前帧---闭环帧）
        // 3. 将闭环帧以及闭环帧相连的关键帧的地图点与当前帧的点进行匹配（当前帧---闭环帧+相连关键帧）
        // 注意以上匹配的结果均都存在成员变量[mvpCurrentMatchedPoints]中，实际的更新步骤见CorrectLoop()步骤3
        // 对于双目或者是RGBD输入的情况,计算得到的尺度=1

        //  准备工作
        // For each consistent loop candidate we try to compute a Sim3
        // 对每个（上一步DetectLoop得到的具有足够连续关系的）闭环候选帧都准备算一个Sim3
        const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

        // We compute first ORB matches for each candidate
        // If enough matches are found, we setup a Sim3Solver
        ORBmatcher matcher(0.75, true); // 要检查方向

        // 存储每一个候选帧的Sim3Solver求解器
        vector<Sim3Solver *> vpSim3Solvers;
        vpSim3Solvers.resize(nInitialCandidates);

        // 存储每个候选帧的匹配地图点信息
        vector<vector<MapPoint *>> vvpMapPointMatches;
        vvpMapPointMatches.resize(nInitialCandidates);

        // 存储每个候选帧应该被放弃(True）或者保留(False)
        vector<bool> vbDiscarded;
        vbDiscarded.resize(nInitialCandidates);

        // 完成 Step 1 的匹配后，被保留的候选帧数量
        int nCandidates = 0;

        // Step 1. 遍历闭环候选帧集，初步筛选出与当前关键帧的匹配特征点数大于20的候选帧集合，并为每一个候选帧构造一个Sim3Solver
        for (int i = 0; i < nInitialCandidates; i++)
        {
            // Step 1.1 从筛选的闭环候选帧中取出一帧有效关键帧[pKF]
            KeyFrame *pKF = mvpEnoughConsistentCandidates[i];

            // 避免在LocalMapping中KeyFrameCulling函数将此关键帧作为冗余帧剔除
            pKF->SetNotErase();

            // 如果候选帧质量不高，直接PASS
            if (pKF->isBad())
            {
                vbDiscarded[i] = true;
                continue;
            }

            // Step 1.2 将当前帧[mpCurrentKF]与闭环候选关键帧[pKF]匹配
            // 通过Bow加速得到[mpCurrentKF]与[pKF]之间的匹配特征点
            // [vvpMapPointMatches]是匹配特征点对应的地图点,本质上来自于候选闭环帧[pKF]
            // pKF2中与pKF1匹配的MapPoint，vpMatches12[i]表示匹配的地图点，null表示没有匹配，i表示匹配的pKF1特征点索引
            int nmatches = matcher.SearchByBoW(mpCurrentKF, pKF, vvpMapPointMatches[i]);

            // 粗筛：匹配的特征点数太少，该候选帧剔除
            if (nmatches < 20)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                // Step 1.3 为保留的候选帧构造Sim3求解器
                // 如果 mbFixScale（是否固定尺度） 为 true，则是6 自由度优化（双目 RGBD）
                // 如果是false，则是7 自由度优化（单目） 尺度性导致自由度+1
                Sim3Solver *pSolver = new Sim3Solver(mpCurrentKF, pKF, vvpMapPointMatches[i], mbFixScale);

                // Sim3Solver Ransac 过程置信度0.99，至少20个inliers 最多300次迭代
                pSolver->SetRansacParameters(0.99, 20, 300);
                vpSim3Solvers[i] = pSolver;
            }

            // 保留的候选帧数量 能运行到这里没有被continue 也就是说这个候选帧叻
            nCandidates++;
        }

        // 用于标记是否至少有一个候选帧通过Sim3Solver的求解与优化 （一个就好）
        bool bMatch = false;

        // Step 2 对每一个候选帧用Sim3Solver迭代匹配，直到有一个候选帧匹配成功，或者全部失败
        while (nCandidates > 0 && !bMatch)
        {
            // 遍历每一个候选帧
            for (int i = 0; i < nInitialCandidates; i++)
            {
                if (vbDiscarded[i])
                    continue;

                KeyFrame *pKF = mvpEnoughConsistentCandidates[i];

                // 内点（Inliers）标志
                // 即标记经过RANSAC sim3 求解后, [vvpMapPointMatches]中的哪些作为内点
                vector<bool> vbInliers;
                // 内点（Inliers）数量
                int nInliers;
                // 是否到达了最优解
                bool bNoMore; // 为true表示穷尽迭代还没有找到好的结果，说明求解失败

                // Step 2.1 取出从 Step 1.3 中为当前候选帧构建的 Sim3Solver 并开始迭代
                Sim3Solver *pSolver = vpSim3Solvers[i];

                // 最多迭代5次，返回的[Scm]是[候选帧pKF]到[当前帧mpCurrentKF]的Sim3变换（T12）
                cv::Mat Scm = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reaches max. iterations discard keyframe
                // 总迭代次数达到最大限制还没有求出合格的Sim3变换，该候选帧剔除
                if (bNoMore)
                {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
                // 如果计算出了Sim3变换，继续匹配出更多点并优化。因为之前 SearchByBoW 匹配可能会有遗漏
                if (!Scm.empty())
                {
                    // 取出经过Sim3Solver后匹配点中的内点集合
                    vector<MapPoint *> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint *>(NULL));
                    for (size_t j = 0, jend = vbInliers.size(); j < jend; j++)
                    {
                        // 保存内点
                        if (vbInliers[j])
                            vpMapPointMatches[j] = vvpMapPointMatches[i][j];
                        // 其余的 不是内点 就还是<MapPoint *>(NULL)
                    }

                    // Step 2.2 通过上面求取的Sim3变换引导关键帧匹配，弥补Step 1中的漏匹配
                    // [候选帧pKF]到[当前帧mpCurrentKF]的 R（R12）， t（t12）， 变换尺度s（s12）
                    cv::Mat R = pSolver->GetEstimatedRotation();
                    cv::Mat t = pSolver->GetEstimatedTranslation();
                    const float s = pSolver->GetEstimatedScale();

                    // 查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数，之前使用SearchByBoW进行特征点匹配时会有漏匹配）
                    // 通过Sim3变换，投影搜索pKF1的特征点在pKF2中的匹配，同理，投影搜索pKF2的特征点在pKF1中的匹配
                    // 只有互相都成功匹配的才认为是可靠的匹配
                    matcher.SearchBySim3(mpCurrentKF, pKF, vpMapPointMatches, s, R, t, 7.5);

                    // Step 2.3 用新的匹配来优化 Sim3，只要有一个候选帧通过Sim3的求解与优化，就跳出停止对其它候选帧的判断

                    // OpenCV的Mat矩阵转成Eigen的Matrix类型
                    // gScm：[候选关键帧m]到[当前帧c]的Sim3变换
                    g2o::Sim3 gScm(Converter::toMatrix3d(R), Converter::toVector3d(t), s);

                    // 如果mbFixScale为true，则是6 自由度优化（双目 RGBD），如果是false，则是7 自由度优化（单目）
                    // 优化[mpCurrentKF]与[pKF]对应的MapPoints间的[Sim3]，得到优化后的量[gScm]
                    const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                    // 如果优化成功，则停止while循环遍历闭环候选
                    if (nInliers >= 20)
                    {
                        // 为True时将不再进入 while循环
                        bMatch = true; // 至少有一个候选帧通过Sim3Solver的求解与优化
                        // [mpMatchedKF]就是最终闭环检测出来与当前帧[mpCurrentKF]形成闭环的关键帧
                        mpMatchedKF = pKF;

                        // gSmw：从[世界坐标系w]到[该候选帧m]的Sim3变换，都在一个坐标系下，所以尺度 Scale=1
                        // 只是构造了这个g2o::Sim3数据类型，并不需要执行优化
                        g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()), Converter::toVector3d(pKF->GetTranslation()), 1.0);

                        // 得到g2o优化后从[世界坐标系w]到[当前帧c]的Sim3变换
                        mg2oScw = gScm * gSmw;
                        mScw = Converter::toCvMat(mg2oScw);

                        mvpCurrentMatchedPoints = vpMapPointMatches;

                        // 只要有一个候选帧通过Sim3的求解与优化，就跳出停止对其它候选帧的判断 // 停止遍历每一个候选帧
                        break;
                    }
                }
            }
        }

        // 退出上面while循环的原因有两种,一种是求解到了（bMatch置位后）, 另外一种是nCandidates耗尽为0（全部失败）
        if (!bMatch)
        {
            // 如果没有一个闭环匹配候选帧通过Sim3的求解与优化
            // 清空[mvpEnoughConsistentCandidates]，这些候选关键帧以后都不会在再参加回环检测过程了
            for (int i = 0; i < nInitialCandidates; i++)
                mvpEnoughConsistentCandidates[i]->SetErase();

            // 当前关键帧也将不会再参加回环检测了
            mpCurrentKF->SetErase();

            // Sim3 计算失败，退出了
            return false;
        }

        // Step 3：取出与当前帧闭环匹配上的关键帧[mpMatchedKF]及其共视关键帧，以及这些共视关键帧的地图点
        // 注意是闭环检测出来 与当前帧形成闭环的关键帧 mpMatchedKF
        // 将mpMatchedKF共视的关键帧全部取出来放入 vpLoopConnectedKFs
        // 将vpLoopConnectedKFs的地图点取出来放入mvpLoopMapPoints
        vector<KeyFrame *> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();

        // 包含闭环匹配关键帧本身,形成一个“闭环关键帧小组”
        vpLoopConnectedKFs.push_back(mpMatchedKF);

        // 闭环关键帧[mpMatchedKF]上的所有相连关键帧+自身 的地图点
        mvpLoopMapPoints.clear();

        // 遍历“闭环关键帧小组”中的每一个关键帧
        //? 为什么不用vend = vpLoopConnectedKFs.end()
        for (vector<KeyFrame *>::iterator vit = vpLoopConnectedKFs.begin(); vit != vpLoopConnectedKFs.end(); vit++)
        {
            KeyFrame *pKF = *vit;
            vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches(); // 获取当前关键帧的具体的地图点

            // 遍历其中一个关键帧的有效地图点
            for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
            {
                MapPoint *pMP = vpMapPoints[i];
                if (pMP)
                {
                    // mnLoopPointForKF 用于标记，避免重复添加
                    if (!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId)
                    {
                        mvpLoopMapPoints.push_back(pMP);
                        // 标记一下
                        pMP->mnLoopPointForKF = mpCurrentKF->mnId;
                    }
                }
            }
        }

        // Find more matches projecting with the computed Sim3
        // Step 4：将闭环关键帧[mpMatchedKF]及其连接关键帧（闭环关键帧组）的所有地图点 投影到 当前关键帧[mpCurrentKF]进行投影匹配
        // 根据投影查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数）
        // 根据Sim3变换，将每个[mvpLoopMapPoints]投影到[mpCurrentKF]上，搜索新的匹配对，更新结果输出在[mvpCurrentMatchedPoints]
        // [mvpCurrentMatchedPoints]是前面经过SearchBySim3得到的已经匹配的点对，这里就忽略不再匹配了
        // 搜索范围系数为10
        matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints, 10);

        // If enough matches accept Loop
        // Step 5: 统计当前帧[mpCurrentKF]与闭环关键帧[mpMatchedKF]的匹配地图点数目，超过40个说明成功闭环，否则失败
        int nTotalMatches = 0;
        for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++)
        {
            if (mvpCurrentMatchedPoints[i])
                nTotalMatches++;
        }

        if (nTotalMatches >= 40)
        {
            // 如果当前回环可靠,保留当前待闭环关键帧[mpMatchedKF]，其他闭环候选全部删掉以后不用了
            for (int i = 0; i < nInitialCandidates; i++)
                if (mvpEnoughConsistentCandidates[i] != mpMatchedKF) // 留下来做闭环校正？
                    mvpEnoughConsistentCandidates[i]->SetErase();

            return true;
        }
        else
        {
            // 闭环不可靠，闭环候选[mvpEnoughConsistentCandidates]及当前待闭环帧[mpCurrentKF]全部删除
            for (int i = 0; i < nInitialCandidates; i++)
                mvpEnoughConsistentCandidates[i]->SetErase();
            mpCurrentKF->SetErase();

            return false;
        }
    }

    /**
     * @brief 闭环矫正
     *
     * 1. 通过求解的Sim3以及相对姿态关系，调整 与当前帧相连的关键帧位姿 以及 这些关键帧观测到的地图点位置（相连关键帧---当前帧）
     * 2. 将 闭环帧以及闭环帧相连的关键帧的地图点 和 与当前帧相连的关键帧的点进行匹配（当前帧+相连关键帧---闭环帧+相连关键帧）
     * 3. 通过MapPoints的匹配关系更新这些帧之间的连接关系，即更新covisibility graph
     * 4. 对Essential Graph（Pose Graph）进行优化，MapPoints的位置则根据优化后的位姿（关键帧的位姿）做相对应的调整
     * 5. 创建线程进行全局Bundle Adjustment
     */
    void LoopClosing::CorrectLoop()
    {
        cout << "Loop detected!" << endl;
        // Step 0：结束局部地图线程、全局BA，为闭环矫正做准备
        // Step 1：根据共视关系更新当前帧与其它关键帧之间的连接
        // Step 2：通过位姿传播，得到Sim3优化后，与当前帧相连的关键帧的位姿，以及它们的MapPoints
        // Step 3：检查当前帧的MapPoints与闭环匹配帧的MapPoints是否存在冲突，对冲突的MapPoints进行替换或填补
        // Step 4：通过将闭环时相连关键帧的[mvpLoopMapPoints]投影到这些关键帧中，进行MapPoints检查与替换
        // Step 5：更新当前关键帧之间的共视相连关系，得到因闭环时MapPoints融合而新得到的连接关系
        // Step 6：进行EssentialGraph优化，LoopConnections是形成闭环后新生成的连接关系，不包括步骤7中当前帧与闭环匹配帧之间的连接关系
        // Step 7：添加当前帧与闭环匹配帧之间的边（这个连接关系不优化）
        // Step 8：新建一个线程用于全局BA优化

        // g2oSic： 当前关键帧[mpCurrentKF]到其共视关键帧[pKFi]的Sim3相对变换
        // mg2oScw: [世界坐标系w]到[当前关键帧c]的Sim3变换
        // g2oCorrectedSiw：[世界坐标系w]到[当前关键帧共视关键帧]的Sim3 变换

        // Send a stop signal to Local Mapping
        // Avoid new keyframes are inserted while correcting the loop
        // Step 0：结束局部地图线程、全局BA，为闭环矫正做准备
        // 请求局部地图停止，防止在回环矫正时局部地图线程中InsertKeyFrame函数插入新的关键帧
        mpLocalMapper->RequestStop();

        if (isRunningGBA())
        {
            // 如果有全局BA在运行，终止掉，迎接新的全局BA
            unique_lock<mutex> lock(mMutexGBA);
            mbStopGBA = true;
            // 记录全局BA次数
            mnFullBAIdx++;
            if (mpThreadGBA)
            {
                // 停止正在运行的全局BA线程
                mpThreadGBA->detach();
                delete mpThreadGBA;
            }
        }

        // Wait until Local Mapping has effectively stopped
        // 一直等到局部地图线程结束再继续
        while (!mpLocalMapper->isStopped())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Ensure current keyframe is updated
        // Step 1：根据共视关系更新当前关键帧与其它关键帧之间的连接关系
        // 因为之前闭环检测、计算Sim3中改变了该关键帧的地图点，所以需要更新
        mpCurrentKF->UpdateConnections();

        // Retrieve keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
        // Step 2：通过位姿传播，得到Sim3优化后，与当前帧相连的关键帧的位姿，以及它们的地图点
        // 当前帧与世界坐标系之间的Sim变换在ComputeSim3函数中已经确定并优化，
        // 通过相对位姿关系，可以确定这些相连的关键帧与世界坐标系之间的Sim3变换

        // 取出当前关键帧及其共视关键帧，称为“当前关键帧组”
        mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
        mvpCurrentConnectedKFs.push_back(mpCurrentKF);

        // [键 KeyFrame *] [值 g2o::Sim3]
        // CorrectedSim3 ：存放闭环g2o优化后[当前关键帧的共视关键帧]的世界坐标系下Sim3变换 // 经过位姿传播后
        // NonCorrectedSim3 ：存放没有矫正的[当前关键帧的共视关键帧]的世界坐标系下Sim3变换 // 未经过位姿传播
        KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
        // 先将mpCurrentKF的Sim3变换存入，认为是准的，所以固定不动（放在优化后的变量里）
        CorrectedSim3[mpCurrentKF] = mg2oScw;

        // [当前关键帧c]到[世界坐标系w]下的变换矩阵
        cv::Mat Twc = mpCurrentKF->GetPoseInverse();

        // 对地图点操作
        {
            // Get Map Mutex
            // 锁定地图点
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Step 2.1：通过[mg2oScw]（认为是准的）来进行位姿传播，得到[当前关键帧的共视关键帧]的世界坐标系下Sim3位姿

            for (vector<KeyFrame *>::iterator // 遍历“当前关键帧组”: 当前关键帧及其共视关键帧
                     vit = mvpCurrentConnectedKFs.begin(),
                     vend = mvpCurrentConnectedKFs.end();
                 vit != vend; vit++)
            {
                KeyFrame *pKFi = *vit;

                cv::Mat Tiw = pKFi->GetPose();
                if (pKFi != mpCurrentKF) // 跳过当前关键帧，因为当前关键帧的位姿已经在前面优化过了，在这里是参考基准（以当前关键帧的位作为基准进行传播）
                {
                    // 得到[当前关键帧mpCurrentKF]到[其共视关键帧pKFi]的相对变换
                    cv::Mat Tic = Tiw * Twc;
                    cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
                    cv::Mat tic = Tic.rowRange(0, 3).col(3);

                    // 构造[g2oSic]： [当前关键帧mpCurrentKF]到[其共视关键帧pKFi]的Sim3相对变换
                    //? 这里是non-correct, 所以scale=1.0
                    g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0); //? 这个是不准的吧？
                    // 当前帧的位姿[mg2oScw]固定不动，其它的关键帧根据相对关系得到Sim3调整的位姿
                    g2o::Sim3 g2oCorrectedSiw = g2oSic * mg2oScw;

                    // Pose corrected with the Sim3 of the loop closure
                    // 存放闭环g2o优化后[当前关键帧的共视关键帧]的Sim3位姿
                    CorrectedSim3[pKFi] = g2oCorrectedSiw; // 经过位姿传播后
                }
                // 得到[世界坐标系w]到[当前关键帧mpCurrentKF+其共视关键帧pKFi]的相对变换
                cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
                cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
                g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);

                // Pose without correction
                // 存放没有矫正的[当前关键帧mpCurrentKF+其共视关键帧pKFi]的Sim3变换
                NonCorrectedSim3[pKFi] = g2oSiw; // 未经过位姿传播
            }

            // Correct all MapPoints observed by current keyframe and neighbors,
            // so that they align with the other side of the loop
            // Step 2.2：得到矫正的当前关键帧的共视关键帧位姿后，修正这些共视关键帧的地图点
            for (KeyFrameAndPose::iterator //? 遍历待矫正的共视关键帧（不包括当前关键帧）
                     mit = CorrectedSim3.begin(),
                     mend = CorrectedSim3.end();
                 mit != mend; mit++)
            {
                // 取出当前关键帧连接关键帧
                KeyFrame *pKFi = mit->first;
                //? 取出经过位姿传播后的Sim3变换
                g2o::Sim3 g2oCorrectedSiw = mit->second;
                g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();
                //? 取出未经过位姿传播的Sim3变换
                g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

                vector<MapPoint *> vpMPsi = pKFi->GetMapPointMatches();
                // 遍历待矫正共视关键帧中的每一个地图点
                for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP < endMPi; iMP++)
                {
                    MapPoint *pMPi = vpMPsi[iMP];

                    // 跳过无效的地图点
                    if (!pMPi)
                        continue;
                    if (pMPi->isBad())
                        continue;
                    // 标记，防止重复矫正：这个地图点已经被当前关键帧校正过了
                    if (pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
                        continue;

                    // 矫正过程本质上也是 基于当前关键帧的优化后的位姿展开的
                    // Project with non-corrected pose and project back with corrected pose
                    // 将该未校正的[eigP3Dw]先从[世界坐标系w]映射到[未校正的pKFi相机坐标系]，然后再反映射到[校正后的世界坐标系]下
                    cv::Mat P3Dw = pMPi->GetWorldPos();
                    // 地图点世界坐标系下坐标
                    Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
                    // g2o::Sim3 的 map(3dPoint) 内部做了相似变换： s * (R*P) + t
                    // 下面变换是： eigP3Dw: world → g2oSiw → i → g2oCorrectedSwi → world
                    Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                    cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                    pMPi->SetWorldPos(cvCorrectedP3Dw);

                    // 记录矫正该地图点的关键帧id，防止重复
                    pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                    // 记录该地图点所在的关键帧id
                    pMPi->mnCorrectedReference = pKFi->mnId;
                    // 因为地图点更新了，需要更新其平均观测方向以及观测距离范围
                    pMPi->UpdateNormalAndDepth();
                }

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                // Step 2.3：将共视关键帧的Sim3转换为SE3，根据更新的Sim3，更新关键帧的位姿
                // 其实是现在已经有了更新后的关键帧组[mvpCurrentConnectedKFs]中关键帧的位姿,
                // 但是在上面的操作时只是暂时存储到了[KeyFrameAndPose类型]的变量[CorrectedSim3, NonCorrectedSim3]中,还没有写回到关键帧对象中
                // 调用[toRotationMatrix]可以自动归一化旋转矩阵
                Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
                Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
                double s = g2oCorrectedSiw.scale();
                // 平移向量中包含有尺度信息，还需要用尺度归一化
                eigt *= (1. / s); // [R t/s;0 1]

                cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);
                // 设置矫正后的新的pose
                pKFi->SetPose(correctedTiw);

                // Make sure connections are updated
                // Step 2.4：根据共视关系更新当前帧与其它关键帧之间的连接
                // 地图点的位置改变了,可能会引起共视关系\权值的改变
                pKFi->UpdateConnections();
            }

            // Start Loop Fusion
            // Update matched map points and replace if duplicated
            // Step 3：检查[当前帧的地图点]与[经过闭环匹配后该帧的地图点]是否存在冲突，对冲突的进行替换或填补
            // [mvpCurrentMatchedPoints]是[当前关键帧]和[闭环关键帧组的所有地图点]进行投影得到的匹配点
            for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++) // i 当前关键帧的特征点的索引号
            {
                if (mvpCurrentMatchedPoints[i])
                { // 有可能存在冲突
                    //取出同一个索引对应的两种地图点，决定是否要替换
                    // 匹配投影得到的地图点（[matcher.SearchByProjection]得到的新的）
                    MapPoint *pLoopMP = mvpCurrentMatchedPoints[i];
                    // 原来的地图点（原来的）
                    MapPoint *pCurMP = mpCurrentKF->GetMapPoint(i);

                    if (pCurMP)
                        // 如果有重复的MapPoint，则用匹配的地图点代替现有的
                        // 因为匹配的地图点是经过一系列操作后比较精确的，现有的地图点很可能有累计误差
                        pCurMP->Replace(pLoopMP);
                    else
                    {
                        // 如果当前帧没有该MapPoint，则直接添加
                        mpCurrentKF->AddMapPoint(pLoopMP, i);
                        pLoopMP->AddObservation(mpCurrentKF, i);
                        pLoopMP->ComputeDistinctiveDescriptors();
                    }
                }
            }
        }

        // Project MapPoints observed in [the neighborhood of the loop keyframe]
        // into [the current keyframe and neighbors] using corrected poses.
        // Fuse duplications.
        // Step 4：将闭环相连关键帧组[mvpLoopMapPoints] 投影到[当前关键帧组]中，进行匹配，融合，新增或替换当前关键帧组中KF的地图点
        // 因为 闭环相连关键帧组[mvpLoopMapPoints] 在地图中时间比较久经历了多次优化，认为是准确的
        // 而当前关键帧组中的关键帧的地图点是最近新计算的，可能有累积误差
        // CorrectedSim3 ：存放矫正后[key当前关键帧的共视关键帧]及其[value世界坐标系下Sim3变换]
        SearchAndFuse(CorrectedSim3);

        // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
        // Step 5：更新[当前关键帧组]之间的两级共视相连关系，得到因闭环时地图点融合而新得到的连接关系 // 强调！闭环新得到
        // LoopConnections：存储因为闭环地图点调整而新生成的连接关系
        map<KeyFrame *, set<KeyFrame *>> LoopConnections;

        // Step 5.1：遍历当前帧相连关键帧组（一级相连）
        for (vector<KeyFrame *>::iterator
                 vit = mvpCurrentConnectedKFs.begin(),
                 vend = mvpCurrentConnectedKFs.end();
             vit != vend; vit++)
        {
            KeyFrame *pKFi = *vit;

            // Step 5.2：得到与当前帧相连关键帧的相连关键帧（二级相连）
            vector<KeyFrame *> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames(); // 更新前

            // Update connections. Detect new links.
            // Step 5.3：更新一级相连关键帧的连接关系(会把当前关键帧添加进去,因为地图点已经更新和替换了)
            pKFi->UpdateConnections();
            // Step 5.4：取出该帧更新后的连接关系
            LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames(); // 更新后

            // Step 5.5：从连接关系中去除闭环之前的二级连接关系，剩下的连接就是由闭环得到的连接关系 // 删掉更新前的
            for (vector<KeyFrame *>::iterator
                     vit_prev = vpPreviousNeighbors.begin(),
                     vend_prev = vpPreviousNeighbors.end();
                 vit_prev != vend_prev; vit_prev++)
            {
                LoopConnections[pKFi].erase(*vit_prev); // [erase]作用在[set]上
            }
            // Step 5.6：从连接关系中去除闭环之前的一级连接关系，剩下的连接就是由闭环得到的连接关系 // 删掉更新前的
            for (vector<KeyFrame *>::iterator
                     vit2 = mvpCurrentConnectedKFs.begin(),
                     vend2 = mvpCurrentConnectedKFs.end();
                 vit2 != vend2; vit2++)
            {
                LoopConnections[pKFi].erase(*vit2);
            }
        }

        // Optimize graph
        // Step 6：进行本质图优化，优化本质图中[所有关键帧的位姿和地图点]
        // [LoopConnections]是形成闭环后新生成的连接关系，优化不包括步骤7中[当前帧与闭环匹配帧之间]的连接关系
        Optimizer::OptimizeEssentialGraph(
            mpMap,
            mpMatchedKF, mpCurrentKF,
            NonCorrectedSim3, CorrectedSim3,
            LoopConnections,
            mbFixScale);

        // Add loop edge
        // Step 7：添加当前帧与闭环匹配帧之间的边（这个连接关系不优化）
        // !这两句话应该放在[OptimizeEssentialGraph]之前，因为[OptimizeEssentialGraph]的步骤4.2中有优化
        mpMatchedKF->AddLoopEdge(mpCurrentKF);
        mpCurrentKF->AddLoopEdge(mpMatchedKF);

        // Launch a new thread to perform Global Bundle Adjustment
        // Step 8：新建一个线程用于全局BA优化
        // [OptimizeEssentialGraph]只是优化了一些主要关键帧的位姿，这里进行全局BA可以全局优化[所有位姿和MapPoints]
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;

        mpThreadGBA = new thread(
            &LoopClosing::RunGlobalBundleAdjustment,
            this, // 类的成员函数要传入this指针
            mpCurrentKF->mnId);

        // Loop closed. Release Local Mapping.
        mpLocalMapper->Release(); // 释放当前还在缓冲区中的关键帧指针

        cout << "Loop Closed!" << endl;

        mLastLoopKFid = mpCurrentKF->mnId;
    }

    /**
     * @brief 将[闭环相连关键帧组的地图点mvpLoopMapPoints]投影到[当前关键帧组]中，进行匹配，新增或替换当前关键帧组中KF的地图点
     *
     * 因为[闭环相连关键帧组的地图点mvpLoopMapPoints]在地图中时间比较久经历了多次优化，认为是准确的
     * 而当前关键帧组中的关键帧的地图点是最近新计算的，可能有累积误差
     *
     * @param[in] CorrectedPosesMap  矫正的[当前KF对应的共视关键帧]及[Sim3变换]
     */
    void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
    {
        // 定义ORB匹配器
        ORBmatcher matcher(0.8);

        //? Step 1 遍历待矫正的当前KF的相连关键帧 // 待校正还是已校正？？？？
        for (KeyFrameAndPose::const_iterator
                 mit = CorrectedPosesMap.begin(),
                 mend = CorrectedPosesMap.end();
             mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            // 矫正过的Sim 变换
            g2o::Sim3 g2oScw = mit->second;
            cv::Mat cvScw = Converter::toCvMat(g2oScw);

            // Step 2 将mvpLoopMapPoints投影到pKF帧匹配，检查地图点冲突并融合
            // mvpLoopMapPoints ：与当前关键帧闭环匹配上的关键帧及其共视关键帧组成的地图点
            // vpReplacePoints ：存储[mvpLoopMapPoints]投影到[pKF]匹配后需要替换掉的新增地图点,索引和[mvpLoopMapPoints]一致，初始化为空
            vector<MapPoint *> vpReplacePoints(mvpLoopMapPoints.size(), static_cast<MapPoint *>(NULL));
            // 搜索区域系数为4
            matcher.Fuse(pKF, cvScw, mvpLoopMapPoints, 4, vpReplacePoints);

            // Get Map Mutex
            // 之所以不在上面[Fuse]函数中进行地图点融合更新 的原因是需要对地图加锁
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
            const int nLP = mvpLoopMapPoints.size();
            // Step 3 遍历闭环帧组的所有的地图点，替换掉需要替换的地图点
            for (int i = 0; i < nLP; i++)
            {
                MapPoint *pRep = vpReplacePoints[i];
                if (pRep)
                {
                    // 如果记录了需要替换的地图点
                    // 用[mvpLoopMapPoints]替换掉[vpReplacePoints]里记录的要替换的地图点
                    pRep->Replace(mvpLoopMapPoints[i]);
                }
            }
        }
    }

    // 由外部线程调用,请求复位当前线程
    void LoopClosing::RequestReset()
    {
        // 标志置位
        {
            unique_lock<mutex> lock(mMutexReset);
            mbResetRequested = true;
        }

        // 堵塞,直到回环检测线程复位完成
        while (1)
        {
            {
                unique_lock<mutex> lock2(mMutexReset);
                if (!mbResetRequested)
                    break;
            }
            // usleep(5000);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    // 当前线程调用,检查是否有外部线程请求复位当前线程,如果有的话就复位回环检测线程
    void LoopClosing::ResetIfRequested()
    {
        unique_lock<mutex> lock(mMutexReset);
        // 如果有来自于外部的线程的复位请求,那么就复位当前线程
        if (mbResetRequested)
        {
            mlpLoopKeyFrameQueue.clear(); // 清空参与进行回环检测的关键帧队列
            mLastLoopKFid = 0;            // 上一次没有和任何关键帧形成闭环关系
            mbResetRequested = false;     // 复位请求标志复位
        }
    }

    /**
     * @brief 全局BA线程,这是这个线程的主函数
     *
     * @param[in] nLoopKF 看上去是闭环关键帧id,但是在调用的时候给的其实是[当前关键帧的id]
     */
    void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
    {
        cout << "Starting Global Bundle Adjustment" << endl;

        // 记录GBA已经迭代次数,
        // 用来检查全局BA过程是否是因为意外结束的（意外的时候[mnFullBAIdx]会++）
        int idx = mnFullBAIdx;

        // Step 1 执行全局BA，优化所有的关键帧位姿和地图中地图点
        // [mbStopGBA]直接传引用过去了,这样当有外部请求的时候这个优化函数能够及时响应并且结束掉
        Optimizer::GlobalBundleAdjustemnt(
            mpMap,      // 地图点对象
            10,         // 迭代次数
            &mbStopGBA, // 外界控制 GBA 停止的标志
            nLoopKF,    // 形成了闭环的当前关键帧的id
            false);     // 不使用鲁棒核函数
        // 提问：进行完这个过程后我们能够获得哪些信息?
        // 回答：能够得到[全部关键帧优化后的位姿],以及[优化后的地图点]

        // Update all MapPoints and KeyFrames
        // Local Mapping was active during BA, that means that
        // there might be new keyframes not included in the Global BA and they are not consistent with the updated map.
        // We need to propagate the correction through the spanning tree
        // 更新所有的地图点和关键帧
        // 在global BA过程中local mapping线程仍然在工作，这意味着在global BA时可能有新的关键帧产生，但是并未包括在GBA里，
        // 所以和更新后的地图并不连续。需要通过spanning tree来传播
        {
            unique_lock<mutex> lock(mMutexGBA);
            // 如果GBA过程是因为意外结束的,那么直接退出GBA
            if (idx != mnFullBAIdx)
                return;

            // 如果当前GBA没有中断请求[mbStopGBA==false]，更新[位姿和地图点]
            // 这里和上面那句话的功能还有些不同：
            // 因为如果一次GBA被中断,往往意味又要重新开启一个新的GBA;
            // 为了中断当前正在执行的优化过程[mbStopGBA]将会被置位=true,同时会有一定的时间使得该线程进行响应;
            // 而在开启一个新的GBA进程（mpThreadGBA = new thread...）之前[mbStopGBA]将会被置为=false
            // 因此,如果被强行中断的线程退出时已经有新的线程启动了,[mbStopGBA]=false,为了避免进行后面的程序,所以有了上面的程序;
            // 而如果被强行中断的线程退出时新的线程还没有启动,那么上面的条件就不起作用了 //? 意思是 idx == mnFullBAIdx 吗？
            // (虽然概率很小,前面的程序中[mbStopGBA]置位后很快[mnFullBAIdx]就++了,保险起见),所以这里要再判断一次
            if (!mbStopGBA)
            {
                // 保证[mbStopGBA==false] //? 保险起见：且[idx == mnFullBAIdx]才继续

                cout << "Global Bundle Adjustment finished" << endl;
                cout << "Updating map ..." << endl;

                // 避免在进行全局优化的过程中局部建图[localmapping线程]添加新的关键帧
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                // 等待直到[localmapping线程]结束才会继续后续操作
                while (!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
                {
                    // usleep(1000);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                // Get Map Mutex
                // 后续要更新地图所以要上锁
                unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

                // Correct keyframes starting at map first keyframe
                // 从[第一个关键帧]开始矫正关键帧。 刚开始只保存了初始化的那个关键帧
                list<KeyFrame *> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(), mpMap->mvpKeyFrameOrigins.end());

                // 提问：GBA里锁住[第一个关键帧位姿]没有优化，其对应的[pKF->mTcwGBA]是不变的吧？那后面调整位姿的意义何在？
                // 回答：注意在前面[essential graph BA]里只锁住了[回环帧]，没有锁定[第一个初始化关键帧位姿]。
                //    所以[第一个初始化关键帧位姿]已经更新了，
                //    在GBA里锁住[第一个关键帧位姿]没有优化，其对应的[pKF->mTcwGBA]应该是[essential graph BA]结果，在这里统一更新了

                // Step 2 遍历并更新全局地图中的所有spanning tree中的关键帧
                while (!lpKFtoCheck.empty())
                {
                    KeyFrame *pKF = lpKFtoCheck.front(); // [第一个初始化关键帧位姿]也就是后面说的[父关键帧]

                    const set<KeyFrame *> sChilds = pKF->GetChilds(); // 获取[当前关键帧]的[子关键帧]
                    cv::Mat Twc = pKF->GetPoseInverse();

                    // 遍历当前关键帧的子关键帧
                    for (set<KeyFrame *>::const_iterator
                             sit = sChilds.begin();
                         sit != sChilds.end(); sit++)
                    {
                        KeyFrame *pChild = *sit;

                        // 记录避免重复
                        if (pChild->mnBAGlobalForKF != nLoopKF)
                        {
                            // 从[父关键帧c]到[当前子关键帧child]的位姿变换 T_child_father
                            cv::Mat Tchildc = pChild->GetPose() * Twc; // T_child_w(原来child的位姿) * T_w_c(原来father的逆位姿)

                            // 再利用优化后的[父关键帧]的位姿，转换到世界坐标系下，相当于更新了[子关键帧]的位姿
                            // T_child_w(原来child的位姿) * T_w_c(原来father的逆位姿) * T_c_w(优化后father的位姿) //? 相当于修正量
                            pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;
                            // 做个标记，避免重复
                            pChild->mnBAGlobalForKF = nLoopKF;
                        }
                        // 这种最小生成树中除了根节点，其他的节点都会作为其他关键帧的子节点，这样做可以使得最终所有的关键帧都得到了优化
                        // [父关键帧]的[子关键帧]作为下次循环的[父关键帧]：遍历整颗最小生成树
                        lpKFtoCheck.push_back(pChild);
                    }

                    // 记录未矫正的关键帧的位姿
                    pKF->mTcwBefGBA = pKF->GetPose();
                    // 记录已经矫正的关键帧的位姿
                    pKF->SetPose(pKF->mTcwGBA);

                    // 从列表中移除
                    lpKFtoCheck.pop_front();
                }

                // Correct MapPoints
                const vector<MapPoint *> vpMPs = mpMap->GetAllMapPoints();

                // Step 3 遍历每一个地图点并用[更新的关键帧位姿]来更新[地图点位置]
                for (size_t i = 0; i < vpMPs.size(); i++)
                {
                    MapPoint *pMP = vpMPs[i];

                    if (pMP->isBad())
                        continue;

                    if (pMP->mnBAGlobalForKF == nLoopKF)
                    {
                        // If optimized by Global BA, just update
                        // 如果这个地图点直接参与到了全局BA优化的过程,那么就直接重新设置器位姿即可
                        pMP->SetWorldPos(pMP->mPosGBA);
                    }
                    else
                    {
                        // Update according to the correction of its reference keyframe
                        // 如这个地图点并没有直接参与到全局BA优化的过程中,那么就使用其[参考关键帧]的新位姿来优化自己的坐标
                        KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();

                        // 如果[参考关键帧]并没有经过此次全局BA优化，就跳过
                        if (pRefKF->mnBAGlobalForKF != nLoopKF)
                            continue;

                        // [地图点wrt相机]的位姿认为是正确的，现在[相机wrt世界]发生了优化，传递一下，[地图点wrt世界]优化了
                        // (MP wrt world)_优化后 =
                        // (MP wrt world)_优化前 * (world wrt cam/pRefKF)_优化前 * (cam/pRefKF wrt world)_优化后

                        // 获取[参考关键帧]进行全局BA优化之前的位姿 (world wrt cam/pRefKF)
                        cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
                        cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);
                        // Map to non-corrected camera
                        // 转换到其[参考关键帧]在[相机坐标系cam/pRefKF]下的坐标
                        // (MP wrt cam/pRefKF) = (MP wrt world) * (world wrt cam/pRefKF)_优化前
                        cv::Mat Xc = Rcw * pMP->GetWorldPos() + tcw;

                        // Back project using corrected camera
                        // 然后使用已经纠正过的[参考关键帧]的位姿,再将该地图点变换到世界坐标系下
                        cv::Mat Twc = pRefKF->GetPoseInverse(); // (cam/pRefKF wrt world)_优化后
                        cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
                        cv::Mat twc = Twc.rowRange(0, 3).col(3);
                        // (MP wrt world) = (MP wrt cam/pRefKF) * (cam/pRefKF wrt world)_优化后
                        pMP->SetWorldPos(Rwc * Xc + twc);
                    }
                }

                // 释放当前还在缓冲区中的关键帧指针
                mpLocalMapper->Release();

                cout << "Map updated!" << endl;
            }

            mbFinishedGBA = true;
            mbRunningGBA = false;
        }
    }

    // 由外部线程调用,请求终止当前线程
    void LoopClosing::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    // 当前线程调用,查看是否有外部线程请求当前线程
    bool LoopClosing::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    // 有当前线程调用,执行完成该函数之后线程主函数退出,线程销毁
    void LoopClosing::SetFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
    }

    // 由外部线程调用,判断当前回环检测线程是否已经正确终止了
    bool LoopClosing::isFinished()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }

} // namespace ORB_SLAM2
