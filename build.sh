# cd build

../Examples/Stereo/stereo_kitti ../Vocabulary/ORBvoc.txt ../Examples/Stereo/KITTI03.yaml /mnt/hgfs/code/data_odometry_gray/00/
# for tum rgbd
../Examples/RGB-D/rgbd_tum ../Vocabulary/ORBvoc.txt ../Examples/RGB-D/TUM1.yaml /mnt/hgfs/code/data_tum_rgbd/rgbd_dataset_freiburg1_xyz/ ../Examples/RGB-D/associations/fr1_xyz.txt 

../Examples/RGB-D/rgbd_tum ../Vocabulary/ORBvoc.txt ../Examples/RGB-D/TUM1.yaml /mnt/hgfs/code/data_tum_rgbd/rgbd_dataset_freiburg1_room/ ../Examples/RGB-D/associations/fr1_room.txt 



echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "Configuring and building ORB_SLAM2 ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
