# CMake基础
## 1.find_package(Mylib REQUIRED)
&emsp;&emsp;首先需要明确，要使用 C++ 的库，需要有这个库对应的***头文件和库文件***。头文件存放一些函数或者类的声明；库文件是二进制文件，存放这些函数或者类的具体实现。库文件的好处是不需要再次编译，可以直接使用，而且可以隐藏具体实现，只在头文件中留出接口。所以在编写 CMakeLists.txt 的时候要确保找到库的头文件路径以及库文件路径，并且让可执行程序链接到这个库文件。  
&emsp;&emsp;特别地，如果是纯头文件组成的库（声明和实现都在头文件里），就只需要告诉编译器头文件路径就行。  
`find_package(Mylib REQUIRED)` 这个命令用于找到Mylib这个库的头文件目录以及库文件目录(静态库 .a 或共享库 .so )。找到之后，它会设置两个变量： `Mylib_INCLUDE_DIRS` 和 
 `Mylib_LIBRARIES`，分别对应头文件路径和库文件路径。也就是说，只需要使用 `find_package(Mylib REQUIRED)` 命令就可以为编译器指定头文件和库文件路径。  
&emsp;&emsp;在此之后，只需要将可执行文件链接到库文件就行。使用 `target_link_libraries(executable_pro ${Mylib_LIBRARIES})` 即可完成链接。  
&emsp;&emsp;`find_package` 有两种方式根据库名找到对应的头文件和库文件：
  1. 模块模式：CMake 会查找一个名为 Find<PackageName>.cmake 的模块文件(如FindMylib.cmake)，该文件定义了如何查找头文件和库文件.
  2. 配置模式：CMake 会查找一个配置文件（通常是 MylibConfig.cmake），该文件提供了库的路径和其他相关信息.  
如果一些库没有提供CMake支持，则没有办法通过 `find_package` 找到，此时可以手动指定头文件和库文件路径，例如：
```cmake
# 指定头文件路径
target_include_directories(gridnn_test PRIVATE /usr/include/glog)
# 指定库文件路径
link_directories(/usr/lib/x86_64-linux-gnu)
# 链接到库
target_link_libraries(gridnn_test glog)
```
## 2.一个完整的CMakeLists.txt：
```cmake
# 最低cmake版本需求
cmake_minimum_required(VERSION 3.16)
# 项目名称
project(gridnn_test_project)
# 设置C++标准
set(CMAKE_CXX_STANDARD 17)

# 使用find_package()，通过.cmake配置文件找到所使用的库的头文件和库文件路径
find_package(PCL 1.10 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

# 指定可执行文件
add_executable(gridnn_test src/test.cpp)

# 包含头文件
# 以下两个头文件能够通过find_package()找到
# target_include_directories(gridnn_test PRIVATE ${PCL_INCLUDE_DIRS})
# target_include_directories(gridnn_test PRIVATE ${EIGEN3_INCLUDE_DIR})
# 自己编写的纯头文件
target_include_directories(gridnn_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
# 因为0.4.0的glog没有办法用find_package找到，所以只能手动指定
target_include_directories(gridnn_test PRIVATE /usr/include/glog) # 头文件路径
link_directories(/usr/lib/x86_64-linux-gnu) # 库文件路径
target_link_libraries(gridnn_test glog) # 链接到库

# 链接到库
target_link_libraries(gridnn_test ${PCL_LIBRARIES})
target_link_libraries(gridnn_test glog)
target_link_libraries(gridnn_test TBB::tbb)
```
# 内存对齐
## 为什么需要内存对齐
对于64位的cpu和内存，cpu访问物理内存时，可以一次性获取8个字节的数据，但是只能以8的整数倍作为起始地址，例如0x0000、0x0008。如果一个4字节的数据存放在0x0006到0x0009，那么cpu要访问它需要分别读取0x0000到0x0007的内存区域和0x0008到0x000E的内存区域，增大了时间开销；正确的做法是把这个变量放在以4的整数倍为起始的内存地址，也就是**内存对齐**，这样能节省CPU的访问时间。  
## 结构体内存对齐的机制
1. 数据成员对齐规则：结构（struct或联合union）的数据成员，第一个数据成员放在offset为0的地方，以后每个数据成员obj存储的起始位置要从该成员obj大小的整数倍开始（比如int在32位机为４字节，则要从4的整数倍地址开始存储，short是2字节，就要从2的整数倍开始存储）。
2. 结构体作为成员：如果一个结构里有某些结构体成员，则结构体成员要从其内部最大元素大小的整数倍地址开始存储。（struct a里存有struct b，b里有char，int，double等元素，那b应该从8的整数倍开始存储。）
3. 收尾工作：结构体的总大小，也就是sizeof的结果，必须是其内部最大成员的整数倍，不足的要补齐。如果是结构体B包含了结构体A对象a，判断最大成员时并不是a，而是a结构体的最大成员。
（补：上述取最大成员的大小后，实际上应该取[#pragma pack指定的数值]与[最大成员的数值]比较小的那个为准）  
一个例子：
```cpp
struct A{
    int a; // 4字节
    char b; // 1字节
    short c; // 2字节
}; // sizeof(A)的结果是8。内存布局：1111,1*，11.其中*表示占位补齐。
struct B{
    char b; // 1字节
    int a; // 4字节
    short c; // 2字节
}; // sizeof(B)的结果是12。内存布局：1***,1111,11**.
```
# FAST关键点
## 关键点检测方法
1. 在图像中选取像素 p，假设它的亮度为 I<sub>p</sub>。
2. 设置一个阈值 T (比如 I<sub>p</sub>的 20%)。
3. 以像素 p 为中心, 选取半径为 3 的圆上的 16 个像素点。
4. 假如选取的圆上，有连续的 N 个点的亮度大于 I<sub>p</sub> + T 或小于 I<sub>p</sub> - T ，那么像素 p 可以被认为是关键点 (N 通常取 12，即为 FAST-12。其它常用的 N 取值为 9 和 11， 他们分别被称为 FAST-9，FAST-11)。
5. 循环以上四步，对每一个像素执行相同的操作。
## FAST的预测试
在执行关键点检测之前可以先排除掉大部分不是关键点的像素点。做法是：对于每个像素点，直接检测其邻域（半径为3）圆上的第 1，5，9，13 个像素的亮度。只有当这四个像素中有三个同时大于 I<sub>p</sub> + T 或同时小于 I<sub>p</sub> - T 时，当前像素才有可能是一个角点，否则应该直接排除。
## FAST的后处理
原始的 FAST 角点经常出现“扎堆”的现象。所以在第一遍检测之后，还需要用非极大值抑制，在一定区域内仅保留响应极大值的角点，避免角点集中的问题。
## FAST存在的问题
- FAST 关键点数量很大且不确定。
- FAST关键点不具有尺度不变性和旋转不变性。
- FAST算法仅仅提取关键点，没有关键点的描述子，所以不能称为FAST特征。
# ORB特征
## 构建图像金字塔让FAST关键点具有缩放不变性
## 连接图像块的质心和几何中心构成方向向量，使得描述子具有旋转不变性
可以这样理解：这里的方向向量会随着图像的旋转而旋转（因为质心随着图像一起旋转，在图像中的相对位置不变），所以就把这个方向向量作为主方向（相当于坐标轴），把关键点周围的图像块旋转到与这个主方向一致（假设主方向与x轴夹角为$`\theta`$，那么就将图像块旋转$`\theta`$），然后再生成描述子，使得关描述子具有方向不变性。
## 改进BRIEF描述子-rBRIEF
经过上面那一步操作，描述子已经具备了方向不变性，ORB还对BRIEF做了些改进，主要是为了增加描述子的可区分度，具体细节没仔细看。
## 使用ORB算法的流程
``` cpp
// 初始化
cv::Mat descriptors1, descriptors2;
cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
cv::BFMatcher matcher(cv::NORM_HAMMING);
// 第一步:检测Oriented FAST角点位置
detector->detect(img1, keypoints1);
detector->detect(img2, keypoints2);

// 第二步:根据角点位置计算BRIEF描述子
descriptor->compute(img1, keypoints1, descriptors1);
descriptor->compute(img2, keypoints2, descriptors2);

// 第三步:对两幅图像中的BRIEF描述子进行匹配,使用Hamming距离
std::vector<cv::DMatch> matches_all;
matcher.match(descriptors1, descriptors2, matches_all);

// 第四步：匹配点对筛选
double min_dist = 10000, max_dist = 0;

// 找出所有匹配之间的最小距离和最大距离,即是最相似的和最不相似的两组点之间的距离
for (int i = 0; i < descriptors1.rows; i++)
{
  double dist = matches_all[i].distance;
  if (dist < min_dist)
      min_dist = dist;
  if (dist > max_dist)
      max_dist = dist;
}
printf("-- Max dist : %f \n", max_dist);
printf("-- Min dist : %f \n", min_dist);

// 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误
for (int i = 0; i < descriptors1.rows; i++)
{
  if (matches_all[i].distance <= std::max(2 * min_dist, 20.0))
  {
      matches.push_back(matches_all[i]);
  }
}
```
# 关于OpenCV
## 关键点的存储
关键点的数据结构是```cv::KeyPoint```。一般会创建一个vector来存储关键点：```std::vector<cv::KeyPoint> keypoints```。一个KeyPoint包括：
``` java
@param pt x & y coordinates of the keypoint
@param size 关键点的邻域范围
@param angle 关键点邻域内的主方向
@param response 关键点检测器计算的响应值，例如Harris响应值
@param octave 关键点所处的金字塔层级
@param class_id object id
```
## 描述子的存储
关键点描述子是一个向量，所以直接用```cv::Mat```来存储描述子。矩阵的每一行代表一个关键点的描述子向量，如果有m个描述子，描述子向量的长度为n，则一张图像的关键点描述子矩阵的维数是$`m \times n`$。
## 匹配结果的存储
两张图片的关键点匹配结果存储在```cv::DMatch```中，它的成员变量有：
```cpp
CV_PROP_RW int queryIdx; // 关键点在第一个描述子向量中的索引
CV_PROP_RW int trainIdx; // 关键点在第二个描述子向量中的索引
CV_PROP_RW int imgIdx;   // 多源图像匹配的参数，用不上
CV_PROP_RW float distance; // 当前两个关键点的距离，在ORB中是汉明距离
```
假设有两个准备好的描述子，对他们俩进行匹配，匹配结果存在一个```std::vector<cv::DMatch> matches_all```中：
```cpp
std::vector<cv::DMatch> matches_all;
matcher.match(descriptors1, descriptors2, matches_all);
```
那么，可以通过以下方式查询匹配点在两张图像中的索引（x、y索引）：
```cpp
// 获取第0个匹配结果在第一张图片中的关键点坐标
float x_1 = keypoints1[matches_all[0].queryIdx].pt.x;
float y_1 = keypoints1[matches_all[0].queryIdx].pt.y;
// 获取第0个匹配结果在第二张图片中的关键点坐标
float x_2 = keypoints2[matches_all[0].trainIdx].pt.x;
float y_2 = keypoints2[matches_all[0].trainIdx].pt.y;
```
# 一些术语
## 归一化平面
假设相机坐标系中有一个点$`P_c=(X_c,Y_c,Z_c)`$，那么它在归一化平面的坐标为：$`P=(\frac{{{X_c}}}{{{Z_c}}},\frac{{{Y_c}}}{{{Z_c}}},1)`$。也就是说，归一化平面是与相机光心所在平面距离为1的平面。这里要注意，针对的是相机坐标系下的某个点，不是世界坐标系。
## 超定方程组
方程组个数大于未知量个数，方程不一定有精确解。处理方法是：
- 最小二乘法求最优的系数。
- RANSAC求一个较好的系数。
# 视觉里程计中的相机运动估计
## 单目视觉：根据2D-2D匹配点求解相机运动
### 对极几何约束
![image](https://github.com/user-attachments/assets/110fd007-b209-44e8-9292-8013ad2cc3d6)  
对极几何约束如上图所示。其中P是关键点在**第一帧图像中的相机坐标**，$`p_1`$和$`p_2`$分别是该关键点在两帧图像中的**像素坐标**。下面推导对极几何约束的数学表达式。  
两帧图像中的相机数学模型为：
```math
\left\{ {\begin{array}{*{20}{l}}
  {{s_1}{p_1} = KP} \\ 
  {{s_2}{p_2} = K\left( {RP + t} \right)} 
\end{array}} \right.
```
其中$`s_1`$和$`s_2`$分别表示相机坐标系下的关键点在两帧图像中的深度，也就是z坐标。定义关键点在两帧图像的归一化坐标系下的坐标$`x_1`$和$`x_2`$：
```math
\left\{ {\begin{array}{*{20}{l}}
{{x_1} = \frac{P}{{{s_1}}} = {K^{ - 1}}{p_1}}\\
{{x_2} = \frac{{RP + t}}{{{s_2}}} = {K^{ - 1}}{p_2}}
\end{array}} \right.
```
那么，$`x_1`$和$`x_2`$有如下关系：
```math
{s_2}{x_2} = {s_1}R{x_1} + t
```
将上式两边左乘个$`{t^ \wedge }`$，得到：
```math
{s_2}{t^ \wedge }{x_2} = {s_1}{t^ \wedge }R{x_1} + {t^ \wedge }t
```
其中$`{t^ \wedge }t = t \times t = 0`$，上式两边再同时左乘个$`x_2^T`$：
```math
{s_2}x_2^T{t^ \wedge }{x_2} = {s_1}x_2^T{t^ \wedge }R{x_1}
```
根据叉乘的定义，$`{t^ \wedge }{x_2}`$是一个与$`t`$和$`x_2`$都垂直的向量，所以$`{x_2}^T{t^ \wedge }{x_2}=0`$，等式左边为0向量，所以有：
```math
x_2^T{t^ \wedge }R{x_1} = 0
```
把$`x_1`$和$`x_2`$用像素坐标系表示，并代入上述式子，可以得到：
```math
p_2^T{K^{ - T}}{t^ \wedge }R{K^{ - 1}}{p_1} = 0
```
分别定义本质矩阵E和基础矩阵F：
```math
\left\{ {\begin{array}{*{20}{l}}
  {E = {t^ \wedge }R} \\ 
  {F = {K^{ - T}}{t^ \wedge }R{K^{ - 1}} = {K^{ - T}}E{K^{ - 1}}} 
\end{array}} \right.
```
最终得到对极几何约束的简洁形式：
```math
x_2^TE{x_1} = p_2^TF{p_1} = 0
```
因此，相机运动估计问题转化为了：如何根据多对匹配的关键点求解本质矩阵E或者基础矩阵F，从而根据E或F得到相机的运动参数。利用对极几何约束估计相机运动的算法流程为：
- 根据配对点的像素位置，采用**8点法**求出本质矩阵E或者基础矩阵F.
- 对E或F做SVD分解得到相机运动参数$`t`$和$`R`$.
需要注意的是，从E分解到$`t`$和$`R`$会有四种情况，但可以根据深度是否为正得到正确的结果。
### 单应矩阵H
如果某些关键点在同一个平面内，此时直接用对极几何约束来解E是不可行的，需要求单应矩阵H，根据平面上的四个点做运动估计。具体写程序的时候，往往会把H矩阵和E矩阵都算出来，通过一些评价方法确定究竟是使用E矩阵还是H矩阵。
### 三角测量求深度
如果有同一个点的两个不同视角的图像，并且两个视角的转化关系$`t`$（没有尺度）和$`R`$是已知的，那么可以用三角测量求解该点在两个视角下的深度。
### 单目相机运动估计的尺度不确定性
通过分解H或者F矩阵，我们可以得到初始帧到第二帧的运动参数$`t`$（没有尺度）和$`R`$。使用三角测量法，根据两帧图像中的关键点，以及t和R，可以计算出第一帧图像中关键点的深度（没有尺度，是一种相对深度）。这里有两种方式选择尺度因子：
- 把第一帧的相机运动参数中的$`t`$的模长作为尺度因子，后续的相机运动参数$`t`$以及所有的深度值都要除以这个尺度因子。也就是说把第一帧的$`t`$作为单位1.
- 计算第一帧关键点的平均深度，把这个平均深度作为尺度因子，所有特征点的深度值以及相机的平移向量$`t`$都要除以这个尺度因子。
这一步称为单目SLAM的初始化，初始化成功之后，第一帧和第二帧的深度信息就具备了，第二帧到第三帧的运动估计就可以用3D-2D（第二帧的深度已知，第三帧深度未知）的匹配点对使用PNP等算法来求解。所以，初始化之后的单目SLAM与双目SLAM很类似。
### 随机采样一致性算法(RANSAC)
上面提到的根据匹配点对求基础矩阵F或者单应性矩阵H，都要求固定的匹配点对数量，但是大多数时候匹配点对数量都多于8或者4，这时当然可以用最小二乘法的方法来计算最优的F或者H，但时间复杂度太高，不值得。大多是时候会采用RANSAC算法来求一个相对较好的F或者H，RANSAC的算法流程为：
1. 数据集选择： 随机从匹配点对中选择一个小的子集，选择的子集大小是模型拟合所需的数据点数（这里是8或者4）。
2. 模型拟合： 基于随机选择的数据子集，估计F或者H。
3. 评估一致性： 使用所有数据点来评估F或者H。对于每个数据点，计算其与模型的误差。具体来说，使用对极几何约束计算误差：$`e = x_2^TE{x_1} = p_2^TF{p_1}`$。如果误差小于设定的阈值，则认为该点是“内点”（inlier），否则是“外点”（outlier）。
4. 迭代过程： 重复步骤 1 到 3，直到找到一个模型，它拥有足够多的内点（超过预设的阈值）或达到最大迭代次数。
5. 返回最优模型： 从所有迭代中选择内点最多的模型作为最终估计的模型。
## 3D-2D的相机运动估计（PnP）
首先明确这里与前面的2D-2D运动估计有何区别：
- 这里的第一帧图像的深度是已知的，而单目视觉中深度是未知的，需要做初始化确定尺度因子。
- 这里是已知了很多特征点在世界坐标中的位置$`{P_w} = (X,Y,Z,1)`$和它们的像素坐标$`P = (u,v,1)`$，求解相机的投影矩阵$`{\left[ {R|t} \right]_{3 \times 4}} = KT`$，也就是说，这里是利用一帧图像来求解相机位姿，不涉及帧与帧之间的匹配。注意相机的投影矩阵是内参和外参的乘积。

**1.22日修正**:实际上PnP的3D-2D点并不一定非的是世界坐标系中的3D点，也可以是**相机坐标系中的3D点**（需要注意，2D点必须是像素坐标或者归一化平面内的坐标），PnP本质上求的是2D点所在平面（像素平面或者归一化平面）对应的相机坐标系与3D点所在坐标系的相对位姿。列举几种情况方便理解PnP:
1. 已知一帧图像中特征点在世界坐标系中的3D坐标，以及这些特征点的2D像素坐标，相机内参矩阵K未知：可以用PnP求解**像素坐标系与世界坐标系的相对位姿**$`K T_{cw}`$，理论上这种情况不需要相机内参K。但是！OpenCV中的solvePnP函数要求必须输入相机内参K，所以如果要这么做必须自己写程序（因为大多数情况下都会先把相机标定好，不做标定影响PnP精度，所以OpenCV没有考虑不标定的情况）。
2. 已知一帧图像中特征点在世界坐标系中的3D坐标，以及这些特征点的2D像素坐标，相机内参矩阵K已知：OpenCV中的solvePnP会先把特征点像素坐标左乘$`K^{-1}`$，得到特征点在归一化平面中的坐标，然后使用PnP求解出**相机外参**$`R_{cw}`$和$`t_{cw}`$.
3. 已知一帧图像中特征点在**相机坐标系**中的3D坐标，以及这些特征点在**下一帧图像中的2D像素坐标**，相机内参矩阵K已知：同样，OpenCV中的solvePnP会先把第二帧图像中的特征点2D像素坐标左乘$`K^{-1}`$，得到特征点在归一化平面中的坐标，然后使用PnP求解出**第一帧时相机与第二帧时相机的相对位姿**$`R_{21}`$和$`t_{21}`$，注意与上面那种情况的区别。这种情况下，把第一帧的相机坐标系当成世界坐标系就与上面那种情况一致了。
### 直接线性变换DLT
投影矩阵共有12个未知数，每一对3D-2D匹配点对可以提供两个方程（具体推导见14讲第161页），所以直接变换法最少需要6对匹配点（多于6对则使用最小二乘法）即可求解出投影矩阵，然后根据投影矩阵反求相机内外参。  
在我看来，DLT其实做的是相机标定的事（已知棋盘格角点的世界坐标和像素坐标，求相机内参），跟前面相机的运动估计不太一样，因为运动估计不知道特征点的世界坐标。  
值得注意的是，这里忽略了投影矩阵各参数之间的联系，最后解出来的投影矩阵的左上角的旋转矩阵可能不是正交的，需要做QR分解把它投影到SE(3)中去。
## Bundle Adjustment
在一帧图像中，如果有多对在世界坐标系中的3D点和它们对应在像素坐标系下的2D点，则可以用非线性优化的思想，求取一个最优的相机位姿。理论上，这些3D-2D点对满足：
```math
\left[ {\begin{array}{*{20}{c}}
{{u_i}}\\
{{v_i}}\\
1
\end{array}} \right] = \frac{1}{{{s_i}}}K\exp ({\xi ^ \wedge })\left[ {\begin{array}{*{20}{c}}
{{X_i}}\\
{{Y_i}}\\
{{Z_i}}\\
1
\end{array}} \right]
```
其中$`\xi `$即是相机外参对应的李代数，是一个6维的列向量(前三维描述平移，后三维描述旋转)。但是，我们是不知道$`\xi `$的，优化的思想就是先给$`\xi `$一个初值，现在上述等式是不成立的，等式两边存在误差：
```math
{e_i} = \left[ {\begin{array}{*{20}{c}}
{{u_i}}\\
{{v_i}}\\
1
\end{array}} \right] - \frac{1}{{{s_i}}}K\exp ({\xi ^ \wedge })\left[ {\begin{array}{*{20}{c}}
{{X_i}}\\
{{Y_i}}\\
{{Z_i}}\\
1
\end{array}} \right] = P_{uv}^i - \frac{1}{{{s_i}}}K\exp ({\xi ^ \wedge })P_w^i
```
- 注意这里的$`{s_i} \ne {Z_i}`$，因为此处的$`Z_i`$是定义在世界坐标系下的，而$`s_i`$描述的是相机坐标系下的某点与光心平面的深度。
- $`\xi ^ {\wedge }`$是一个$`4\times 4`$的矩阵。注意，这里的$`\xi ^ {\wedge }`$不是$`\xi`$的反对称矩阵，`^`符号表示把一个6维的列向量转换为$`4\times 4`$的矩阵，转换的规则为:
```math
\mathfrak{s}\mathfrak{e}(3) = \left\{ {{\mathbf{\xi }} = \left[ {\begin{array}{*{20}{c}}
  {\mathbf{\rho }} \\ 
  \phi  
\end{array}} \right] \in {\mathbb{R}^6},{\mathbf{\rho }} \in {\mathbb{R}^3},\phi  \in \mathfrak{s}\mathfrak{o}(3),{{\mathbf{\xi }}^ \wedge } = \left[ {\begin{array}{*{20}{c}}
  {{\phi ^ \wedge }}&{\mathbf{\rho }} \\ 
  {{{\mathbf{0}}^T}}&0 
\end{array}} \right] \in {\mathbb{R}^{4 \times 4}}} \right\}
```
考虑多对点，构造总的误差项：
```math
e = \sum\limits_{i = 1}^n {\frac{1}{2}\left\| {{e_i}} \right\|_2^2}
```
- 要优化相机的外参$`\xi `$，则把其余变量视为常量(其实除了$`P_w^i`$其他本身也是常量)，将$`e_i`$在$`\xi `$处**一阶泰勒展开**(这样做的好处参考十四讲P116)，用它的一阶泰勒展开式来近似$`e_i`$在$`\xi `$处的值：
```math
e_i \left( {{{(\xi  + \Delta \xi )}^ \wedge }} \right) \approx e_i ({\xi ^ \wedge }) + J({\xi ^ \wedge })\Delta {\xi ^ \wedge }
```
求$`e_i`$对$`\xi `$的雅克比矩阵比求$`e`$对$`\xi `$的雅克比矩阵简单的多，不妨把$`e_i`$写成:
```math
{e_i} = P_{uv}^i - \frac{1}{{{s_i}}}KP_c^i = P_{uv}^i - {P_i ^ \prime}
```
则根据链式求导法则：
```math
J({\xi ^ \wedge }) = \frac{{\partial {e_i}}}{{\partial \xi }} = \frac{{\partial {e_i}}}{{\partial P_c^i}} \cdot \frac{{\partial P_c^i}}{{\partial \xi}} = {J_1}{J_2}
```
需要分别求两个雅克比矩阵。
- 要优化空间点的位置$`P_w^i`$，需要求误差量对$`P_w^i`$的导数(雅可比矩阵)：$`\frac{{\partial {\mathbf{e}}}}{{\partial {\mathbf{P}}_w^i}}`$，根据关系式$`{e_i} = P_{uv}^i - \frac{1}{{{s_i}}}KP_c^i`$，利用链式求导法则：
```math
\frac{{\partial {{\mathbf{e}}_i}}}{{\partial {\mathbf{P}}_w^i}} = \frac{{\partial {{\mathbf{e}}_i}}}{{\partial {\mathbf{P}}_c^i}}\frac{{\partial {\mathbf{P}}_c^i}}{{\partial {\mathbf{P}}_w^i}} = {{\mathbf{J}}_3}{{\mathbf{J}}_4}
```
### 求雅可比矩阵$`J_1`$
因为$`P_{uv}^i`$是个常数，所以$`\frac{{\partial {e_i}}}{{\partial P_c^i}}`$可以写成：
```math
\frac{{\partial {e_i}}}{{\partial P_c^i}} = \frac{{ - \partial P_i^\prime}}{{\partial P_c^i}}
```
写出$`-P_i^\prime`$和$`P_c^i`$的关系式：
```math
\left[ {\begin{array}{*{20}{c}}
{{u^\prime}}\\
{{v^\prime}}\\
1
\end{array}} \right] = \frac{1}{{Z_c^\prime}}K\left[ {\begin{array}{*{20}{c}}
{X_c^\prime}\\
{Y_c^\prime}\\
{Z_c^\prime}
\end{array}} \right] = \left[ {\begin{array}{*{20}{c}}
{{f_x}}&0&{{c_x}}\\
0&{{f_y}}&{{c_y}}\\
0&0&1
\end{array}} \right]\left[ {\begin{array}{*{20}{c}}
{\frac{{X_c^\prime}}{{Z_c^\prime}}}\\
{\frac{{Y_c^\prime}}{{Z_c^\prime}}}\\
1
\end{array}} \right]
```
即:
```math
\left\{ {\begin{array}{*{20}{c}}
{{u^\prime} = {f_x}\frac{{X_c^\prime}}{{Z_c^\prime}} + {c_x}}\\
{{v^\prime} = {f_y}\frac{{Y_c^\prime}}{{Z_c^\prime}} + {c_y}}
\end{array}}\right.
```
根据雅可比矩阵的计算规则:
```math
{{J}_{1}}=\left[ \begin{array}{*{35}{l}}
   \frac{\partial {{u}^{'}}}{\partial X_{c}^{'}} & \frac{\partial {{u}^{'}}}{\partial Y_{c}^{'}} & \frac{\partial {{u}^{'}}}{\partial Z_{c}^{'}}  \\
   \frac{\partial {{v}^{'}}}{\partial X_{c}^{'}} & \frac{\partial {{v}^{'}}}{\partial Y_{c}^{'}} & \frac{\partial {{v}^{'}}}{\partial Z_{c}^{'}}  \\
\end{array} \right]=\left[ \begin{array}{*{35}{l}}
   \frac{{{f}_{x}}}{Z_{c}^{'}} & 0 & -\frac{{{f}_{x}}X_{c}^{'}}{Z{{_{c}^{'}}^{2}}}  \\
   0 & \frac{{{f}_{y}}}{Z_{c}^{'}} & -\frac{{{f}_{y}}Y_{c}^{'}}{Z{{_{c}^{'}}^{2}}}  \\
\end{array} \right]
```
### 求雅可比矩阵$`J_2`$
推导过程参考:https://zhuanlan.zhihu.com/p/460985235. 简单来说就是利用扰动模型求导，直接给出结果：
```math
\frac{{\partial P_c^i}}{{\partial \xi }} = {J_2} = {\left[ {\begin{array}{*{20}{c}}
  {\mathbf{I}}&{ - {{({{\mathbf{R}}_{cw}}{{\mathbf{p}}_w} + {\mathbf{t}})}^ \wedge }} \\ 
  {{{\mathbf{0}}^ \top }}&{{{\mathbf{0}}^ \top }} 
\end{array}} \right]_{\left[ {1:3} \right]}} = \left[ {\begin{array}{*{20}{c}}
  {\mathbf{I}}&{ - {\mathbf{P}}_c^{\wedge}} 
\end{array}} \right]
```
将两个雅可比矩阵相乘即可得到所求的雅可比矩阵。
### 求雅可比矩阵$`J_3`$和$`J_4`$
$`J_3`$与$`J_1`$完全相同，下面推导$`J_4`$：
```math
{{\mathbf{J}}_4} = \frac{{\partial {\mathbf{P}}_c^i}}{{\partial {\mathbf{P}}_w^i}} = \frac{{\partial \left( {{{\mathbf{R}}_{cw}}{\mathbf{P}}_w^i + t} \right)}}{{\partial {\mathbf{P}}_w^i}} = {{\mathbf{R}}_{cw}}
```
# 非线性优化
## 引言
首先理解为什么需要非线性优化，而不是直接求导，再令导数为0以求得全局最优的参数？答案就在谜面上：令导数等于0，再来解这个关于参数的方程是很困难的。所以要转变思路，不去求全局最优的参数，而是从某个初始点出发（在视觉SLAM里可以由PnP给出一个较好的初始值），用迭代的方法，每一次求得一个增量$`\Delta x`$，使得$`\left\| {f(x + \Delta x)} \right\|_2^2`$达到最小，如此往复直到增量$`\Delta x`$变得非常小（说明达到了局部或者全局最优），则停止优化。  
总结一下，非线性优化的流程就是：
1. 给定某个初始值$`x_0`$.
2. 对于第k次迭代，寻找一个增量$`\Delta {x_k}`$，使得$`\left\| {f(x + \Delta x)} \right\|_2^2`$达到极小值.
3. 如果增量$`\Delta x`$足够小，则停止迭代.
4. 否则令$`{x_{k + 1}} = {x_k} + \Delta {x_k}`$，回到步骤2.

这个流程里最重要也是最难的问题就是，如何求这个增量$`\Delta {x_k}`$？下面介绍几种方法来求解增量。
## 直接法（一阶和二阶梯度法）
想求一个最优的增量$`\Delta x`$让目标函数$`\left\| {f(x + \Delta x)} \right\|_2^2`$达到最小，可以先在$`x`$处将目标函数进行二阶泰勒展开（再高阶就不好求了）：
```math
\left\| {f({\mathbf{x}} + \Delta {\mathbf{x}})} \right\|_2^2 \approx f({\mathbf{x}})_2^2 + {\mathbf{J}}\left( {\mathbf{x}} \right)\Delta {\mathbf{x}} + \frac{1}{2}\Delta {{\mathbf{x}}^T}{\mathbf{H}}\Delta {\mathbf{x}}
```
注意这里边的雅可比矩阵$`J(x)`$和黑森矩阵$`H(x)`$，分别是函数$`\left\| {f(x)} \right\|_2^2`$关于$`x`$的一阶和二阶偏导数矩阵。注意看等式右边，经过泰勒展开变成了一个关于增量$`\Delta {x}`$的表达式，现在问题就简化了。  
如果只考虑一阶项，忽略二阶项，等式右边就是关于增量$`\Delta {x}`$的线性方程，此时目标函数对增量$`\Delta {x}`$求导即可得到梯度方向：
```math
\frac{{d\left\| {f({\mathbf{x}} + \Delta {\mathbf{x}})} \right\|_2^2}}{{d\Delta {\mathbf{x}}}} = {\mathbf{J}}{\left( {\mathbf{x}} \right)^T}
```
所以增量的方向就是梯度的反方向，再给梯度乘个系数$`\lambda`$，把这个系数取名叫学习率，得到最终的增量：$`\Delta {{\mathbf{x}}^*} =  - \lambda {\mathbf{J}}{\left( {\mathbf{x}} \right)^T}`$。这种方式又被叫做梯度下降法或者最速下降法，因为它的增量方向是梯度的负方向。这种方法的问题是过于贪心，需要控制好学习率，否则很有可能反而增加了迭代次数。  
如果同时考虑一阶项和二阶项，目标函数对增量$`\Delta {x}`$求导得到：
```math
\frac{{d\left\| {f({\mathbf{x}} + \Delta {\mathbf{x}})} \right\|_2^2}}{{d\Delta {\mathbf{x}}}} = {\mathbf{J}}{\left( {\mathbf{x}} \right)^T} + {\mathbf{H}}({\mathbf{x}})\Delta {\mathbf{x}}
```
令导数为0，即可得到增量：$`\Delta {\mathbf{x}} =  - {\mathbf{H}}{({\mathbf{x}})^{ - 1}}{\mathbf{J}}{\left( {\mathbf{x}} \right)^T}`$。这种方法有个很大的问题就是黑森矩阵$`{\mathbf{H}}({\mathbf{x}})`$不好求。  
## 高斯-牛顿迭代法
上面的直接法，无论是一阶还是二阶的方法，都避免不了要求函数$`\left\| {f(x)} \right\|_2^2`$关于$`x`$的一阶和二阶偏导数矩阵，这是非常麻烦的。一种最简单的思想就是，直接把$`f({\mathbf{x}})`$在$`x`$处进行一阶泰勒展开：
```math
f({\mathbf{x}} + \Delta {\mathbf{x}}) \approx f({\mathbf{x}}) + {\mathbf{J}}\left( {\mathbf{x}} \right)\Delta {\mathbf{x}}
```
问题依然是如何求得增量$`\Delta {x}`$使得目标函数$`\left\| {f(x + \Delta x)} \right\|_2^2`$最小，现在把上面展开后的式子代入目标函数，得到：
```math
\left\| {f({\mathbf{x}} + \Delta {\mathbf{x}})} \right\|_2^2 \approx \left\| {f({\mathbf{x}}) + {\mathbf{J}}\left( {\mathbf{x}} \right)\Delta {\mathbf{x}}} \right\|_2^2 = {\left( {f({\mathbf{x}}) + {\mathbf{J}}\left( {\mathbf{x}} \right)\Delta {\mathbf{x}}} \right)^T}\left( {f({\mathbf{x}}) + {\mathbf{J}}\left( {\mathbf{x}} \right)\Delta {\mathbf{x}}} \right)
```
把上面的式子全部展开可以得到：
```math
\left\| {f({\mathbf{x}} + \Delta {\mathbf{x}})} \right\|_2^2 \approx \left\| {f({\mathbf{x}})} \right\|_2^2 + 2f{({\mathbf{x}})^T}{\mathbf{J}}\left( {\mathbf{x}} \right)\Delta {\mathbf{x}} + \Delta {{\mathbf{x}}^T}{\mathbf{J}}{\left( {\mathbf{x}} \right)^T}{\mathbf{J}}\left( {\mathbf{x}} \right)\Delta {\mathbf{x}}
```
同样，求上面的式子关于增量$`\Delta {x}`$的导数：
```math
\frac{{d\left\| {f({\mathbf{x}} + \Delta {\mathbf{x}})} \right\|_2^2}}{{d\Delta {\mathbf{x}}}} \approx 2{\mathbf{J}}{\left( {\mathbf{x}} \right)^T}f({\mathbf{x}}) + 2{\mathbf{J}}{\left( {\mathbf{x}} \right)^T}{\mathbf{J}}\left( {\mathbf{x}} \right)\Delta {\mathbf{x}}
```
令导函数等于0，得到：
```math
{\mathbf{J}}{\left( {\mathbf{x}} \right)^T}{\mathbf{J}}\left( {\mathbf{x}} \right)\Delta {\mathbf{x}} =  - {\mathbf{J}}{\left( {\mathbf{x}} \right)^T}f({\mathbf{x}})
```
令等式左边系数矩阵为$`H`$，等式右边项为$`g`$，即：
```math
\begin{gathered}
  {\mathbf{H}}\Delta {\mathbf{x}} = g \hfill \\
  \Delta {\mathbf{x}} = {{\mathbf{H}}^{ - 1}}g \hfill \\ 
\end{gathered}
```
高斯牛顿迭代法避免了求$`\left\| {f(x)} \right\|_2^2`$关于$`x`$的雅可比矩阵和黑森矩阵，只需要求一次$`f({\mathbf{x}})`$关于$`x`$的雅可比矩阵就能求出增量$`\Delta {x}`$，是一种非常简单有效的方法。可以认为高斯牛顿迭代法是利用$`{\mathbf{J}}{\left( {\mathbf{x}} \right)^T}{\mathbf{J}}\left( {\mathbf{x}} \right)`$来近似黑森矩阵。  
这里边有个问题是，原则上$`{\mathbf{J}}{\left( {\mathbf{x}} \right)^T}{\mathbf{J}}\left( {\mathbf{x}} \right)`$必须是可逆的，但是实际应用中这个条件却不一定成立，后面的L-M方法会在这个方面做一些改进。
# 更为具体的SLAM优化过程
在视觉SLAM中，如果是只优化相机位姿或者特征点的3D世界坐标，以只优化相机位姿为例，采用高斯牛顿迭代法，其伪代码为：
```
for iter:max_iterations:
    error = 0;
    H_total=0;
    g_total=0;
    for matched_point:matched_points:
        计算当前匹配点对的重投影误差; // 这里的重投影误差就是前面的f(x)
        计算雅可比矩阵;
        根据雅可比矩阵和重投影误差计算H矩阵和g向量;
        将当前的H和g累加到H_total和g_total; // 因为误差项是最小二乘累加而来的,总的H矩阵和g向量自然是每一项误差累加
    end
    根据H_total和g_total解出相机位姿增量;
    判断一下增量是否合理,以及增量是否足够小从而达到终止条件;
    把增量的李代数表示转换为李群表示;
    用当前相机位姿左乘增量进行位姿更新;
end
```
但是，如果要同时做相机位姿和特征点位置更新，就不能这样单独求每一项的H和g再累加了，因为相机位姿和特征点坐标是会相互影响的(试想一下：更新完相机位姿之后，在优化特征点坐标计算重投影误差的时候就应该用这个新的相机位姿了)。所以办法就是把重投影误差函数写成一个列向量，具体来说：假如有N对匹配点，那么重投影误差向量的维度就是$`2N \times 1`$(每两行就是一对匹配点的重投影误差向量($`2 \times 1`$))：
```math
e = {\left[ {\begin{array}{*{20}{c}}
  {e_{uv}^1} \\ 
  {e_{uv}^2} \\ 
   \vdots  \\ 
  {e_{uv}^n} 
\end{array}} \right]_{_{2N \times 1}}}
```
待优化的总状态向量由相机位姿的李代数和每个3D点的坐标组成:
```math
x = {\left[ {\begin{array}{*{20}{c}}
  {{\xi _{6 \times 1}}} \\ 
  {P_w^1} \\ 
  {P_w^2} \\ 
   \vdots  \\ 
  {P_w^n} 
\end{array}} \right]_{\left( {6 + 3N} \right) \times 1}}
```
雅可比矩阵的维度就是$`2N \times \left( {6 + 3N} \right)`$：
```math
J = {\left[ {\begin{array}{*{20}{c}}
  {{{\left( {\frac{{\partial e_{uv}^1}}{{\partial {\xi _{6 \times 1}}}}} \right)}_{2 \times 6}}}&{{{\left( {\frac{{\partial e_{uv}^1}}{{\partial P_w^1}}} \right)}_{2 \times 3}}}&0& \cdots &0 \\ 
  {{{\left( {\frac{{\partial e_{uv}^2}}{{\partial {\xi _{6 \times 1}}}}} \right)}_{2 \times 6}}}&0&{{{\left( {\frac{{\partial e_{uv}^2}}{{\partial P_w^2}}} \right)}_{2 \times 3}}}& \cdots &0 \\ 
   \vdots & \vdots & \ddots & \vdots & \vdots  \\ 
  {{{\left( {\frac{{\partial e_{uv}^n}}{{\partial {\xi _{6 \times 1}}}}} \right)}_{2 \times 6}}}&0&0& \cdots &{{{\left( {\frac{{\partial e_{uv}^n}}{{\partial P_w^n}}} \right)}_{2 \times 3}}} 
\end{array}} \right]_{2N \times \left( {6 + 3N} \right)}}
```
可以看出来这个雅可比矩阵维度会随着待优化的状态变量的维度增大而增大，这也是为什么以前的SLAM系统不爱用非线性优化而爱用滤波器。这个雅可比矩阵是一个稀疏矩阵，其中很大一部分都是0矩阵，所以$`H = {J^T}J`$也是一个稀疏矩阵，回忆一下那个增量方程：
```math
H\Delta x = g
```
现在已知H是一个稀疏矩阵，就有很多方法来解这个线性方程组了，比如Cholesky分解、稀疏QR分解等等手段，总之利用稀疏性就能快速求出这个维度很高的增量$`\Delta x`$。

# C++知识点记录
## 类中的static关键字
如果用static修饰成员变量或者成员方法，则表示该成员方法(或变量)属于这个类，而**不属于类的实例化对象**。这将给静态成员方法带来如下性质：
- 静态成员方法不能调用非静态成员变量或方法，**只能调用静态成员变量或方法**。
- 调用静态成员方法时不需要类的实例对象，可以直接**通过类名调用**。但通过类的实例对象调用也是合法的(本质上还是通过类名调用)。
- 构造函数是一个特例，它可以被静态成员方法访问，因为构造函数不依赖于实例对象。

例如，下面的例子是合法的:
```cpp
class Config {
private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;

    Config() {} // 默认构造函数

public:
    ~Config() {} // 默认析构函数

    // 设置配置文件路径
    static void setParameterFile(const std::string& filename);

    // 获取配置文件中的数据
    template<typename T>
    static T getParam(const std::string& key) {
        return T(Config::config_->file_[key]);
    }
};
```
上述例子对应于程序设计模式中的**单例设计模式**。所谓单例设计模式，就是说整个程序中，这个类只允许存在一个实例，上述例子就是单例设计模式的实现方法之一。可以看到该类的构造函数是私密的，不能通过类名实例化，只能通过调用静态方法`setParameterFile`进行实例化，并且该静态方法还会检查`config_`指针是否为空，如果不为空也不会创建对象，因此可以确保整个程序中只有一个Config实例。
# 前端里程计的编写过程(RGBD)
## 程序设计思路
一个里程计需要干以下几件事情：
- 里程计需要记录参考帧和当前帧。所以我们需要构建一个“帧”的类。
  - 这个“帧”类需要存储一个RGB图，一个RGB图对应的深度图。这个帧还应该有一些基本的属性，包括帧的id(表示第几帧)、帧的时间戳、当前帧的位姿所对应的变换矩阵$`{T_{cw}}`$(缩写表示current、world)、**相机模型**。除此之外，它还应该定义一些方便的方法，例如：查询某个`cv::KeyPoint`在深度图中的深度，检查某个3D点(Eigen::Vector3d)是否在当前帧中等等。
  - 说到相机模型，那么最好定义一个相机类，这个类需要记录相机内参，并且提供各个坐标系的转换方法，例如：相机坐标系到像素坐标系、世界坐标系到像素坐标系等等。
- 里程计需要维护一个地图，把新来的帧加入到地图中。所以我们需要构建一个“地图类”。
  - 目前地图类要存储两个很关键的东西：一是前面定义的帧(frame)；二是路标点(landmark)。所以他也需要定义两种方法：一是插入关键帧；二是插入路标点。注意，在**机器人定位**中，为了节省时间，一般只把路标点(即特征点)加入到地图中，建立**稀疏地图**；在**三维重建或AR**中，可以把考虑把所有点都加入到地图中，建立**稠密地图**。SLAM的高效稠密建图也是研究方向之一。
- 里程计需要定义特征检测、描述子计算、特征匹配、当前帧与参考帧的相对位姿求解等成员方法。
- 里程计还有其他一些成员变量，例如：参考帧的位姿、参考帧和当前帧的特征描述子等等，此处不再赘述。
总结一下，我们大致需要定义一些基本类，包括：帧(Frame)、相机(Camera)、路标点(LandMark)、地图(Map)。然后用上述基本类定义一个里程计(Odometry)类。除此之外，为了方便读取一些随时需要调整的参数，包括相机内参、特征点数量等参数，可以定义一个单例类Config，每次程序运行时通过读取配置文件获取这些参数。
## 一些注意事项
### 关键帧的概念
关键帧是指对估计相机运动和构建地图有帮助的帧。试想假如相机一直没有运动，那么这时候不断把当前帧加入地图是没有意义的，这些帧就是“无关紧要”的，不是我们需要的关键帧。所以一种筛选关键帧的方法就是：检查估计的相机运动尺度是否大于我们设定的阈值，若大于则将当前帧作为关键帧。(也许可以作为一个可以创新的方向)
### 帧-地图匹配与帧-帧匹配
帧-帧匹配：把参考帧(即上一帧)的ORB特征点与当前帧的特征点进行匹配，然后用PnP解出相机运动的初值，接着利用BA进一步优化相机位姿和3D点坐标，最后把当前帧作为下一次求解的参考帧。它的特点是：
- 严重依赖参考帧的质量，如果参考帧图像质量太差，极大概率导致算法跑歪。
- 每一次都把上一次求解的位姿视为真值，会有较大的累计误差。
- 求解相机运动只考虑了上一帧的特征点，类似于传统的滤波器算法，没有用到历史信息。

帧-局部地图匹配：让当前帧与一个局部地图进行匹配，每一次计算需要更新局部地图，并且不断优化局部地图中的路标点。优点是：
- 用到了历史帧的特征信息(把历史帧的特征点放在一个局部地图中)。
- 会同时更新历史路标，让地图和定位更加精准。
## g2o使用指南
图优化能够直观的表达优化变量、误差函数的结构与关联。在图优化中，**顶点**表示**待优化的变量**，顶点连接的**边**表示**误差函数**。例如，在3D-2D运动估计问题中，如果我要优化相机位姿和3D点坐标，那么顶点就是6维的相机位姿向量和3维的特征点坐标，边就是重投影误差。  
用g2o来做图优化大致分为六个步骤：
1. 定义求解增量方程的求解器LinearSolver。这个求解器用来解增量方程$`H\Delta x = g`$，从而得到增量$`\Delta x`$。
2. 定义单次迭代的求解器BlockSolver。BlockSolver主要负责：
   - 管理相机位姿和特征点的3D坐标;
   - 求海森矩阵H以及梯度向量g;
   - 调用步骤1定义的LinearSolver求解增量方程，得到优化变量的增量;
3. 定义求解器Solver。Solver会干这几件事：
   - 调用步骤2定义的BlockSolver，得到优化变量的增量;
   - 判断是否接受该增量，以及算法是否收敛;
   - 若算法未收敛，则更新优化变量.
4. 定义优化问题的顶点和边。
5. 定义最核心的稀疏优化器g2o::SparseOptimizer，把顶点和边加入到SparseOptimizer中。这个优化器负责管理所有的优化变量，以及前面定义的所有求解器，管理整个优化过程。
6. 设置好参数之后开始优化。
