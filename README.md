# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

# FP.1 Match 3D Objects
Implement the method "matchBoundingBoxes", which takes as input both the previous and the 
current data frames and provides as output the ids of the matched regions of interest (i.e., the 
boxID property). Matches must be the ones with the highest number of keypoint 
correspondences.
To go through all the matches a for loop is used. Inside it, we take the coordinates of both 
keypoint of the match (previous and current frame). For each of thes keypoints, we check in 
which bounding box of the corresponding frame they are placed and in case that both are placed
in some bounding box, we store the id of the last bounding box in which the kpts are found (1 
for keypoint). Then we increment the value in a matrix corresponding to that combination of 
bounding boxes, so at the end we know which are the best combinations (higher counter).

# FP.2 Compute Lidar-based TTC
Compute the time-to-collision in second for all matched 3D objects using only Lidar 
measurements from the matched bounding boxes between current and previous frame.
For this purpose, some auxiliary functions have been added to facilitate the cleanliness of the 
code. The function computeTTCLidar will make use of three functions, directly or indirectly, 
which are ???proximity???, ???euclideanCluster??? and ???removeLidarOutliers???. The main function of the 
process, computeTTCLidar, do the following: First of all, the function ???removeLidarOutliers??? is 
called to remove the Lidar points that are outliers for the actual set of points. Then, it checksthe 
3D Lidar points that are within the lane of the ego vehicle, for both the previous and the current 
frames. The minimum distance point gives us the TTC, since it is the first point of collision.
Inside the function ???removeLidarOutliers???, we calculate all the clusters of Lidar points, by making 
use of the function ???euclideanCluster??? to determine the Euclidean distances between points to 
properly form the clusters. The function ???proximity??? is called inside this last one, processing all 
the Lidar points to include all of them in a cluster.

# FP.3 Associate Keypoint Correspondences with Bounding Boxes
Prepare the TTC computation based on camera measurements by associating keypoint 
correspondences to the bounding boxes which enclose them. All matches which satisfy this 
condition must be added to a vector in the respective bounding box.
It is implemented in the function clusterKptMatchesWithROI. A for loop is used to assign every 
match in the vector kptMatches to the corresponding Bounding Box. Once it is done, a second 
for loop is used to remove outlier matches based on the euclidean distance between them in 
relation to all the matches in the Bounding Box. Finally, using the mean of the distance between 
matches inside the Bounding Box, we determine if the distance between the previous and the 
current keypoint (coordinates in space) is lower or higher to the means time the defined ratio. 
We will keep only the match indexes for those matches with a distance lower to the mean times 
the ratio, in order to have better results.

# FP.3 Associate Keypoint Correspondences with Bounding Boxes
Prepare the TTC computation based on camera measurements by associating keypoint 
correspondences to the bounding boxes which enclose them. All matches which satisfy this 
condition must be added to a vector in the respective bounding box.
It is implemented in the function clusterKptMatchesWithROI. A for loop is used to assign every 
match in the vector kptMatches to the corresponding Bounding Box. Once it is done, a second 
for loop is used to remove outlier matches based on the euclidean distance between them in 
relation to all the matches in the Bounding Box. Finally, using the mean of the distance between 
matches inside the Bounding Box, we determine if the distance between the previous and the 
current keypoint (coordinates in space) is lower or higher to the means time the defined ratio. 
We will keep only the match indexes for those matches with a distance lower to the mean times 
the ratio, in order to have better results.