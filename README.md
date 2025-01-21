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
关键点的数据结构是```cv::KeyPoint```。一般会创建一个vector来存储关键点：```std::vector\<cv::KeyPoint> keypoints```。一个KeyPoint包括：
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
CV_PROP_RW int trainIdx; // 关键点在第一个描述子向量中的索引
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
## 超定方程
方程组个数大于未知量个数，方程不一定有精确解。处理方法是：
- 最小二乘法求最优的系数。
- RANSAC求一个较好的系数。
# 视觉里程计中的相机运动估计
## 单目视觉：根据2D-2D匹配点求解相机运动
### 对极几何约束
利用对极几何约束估计相机运动：
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
e\left( {{{(\xi  + \Delta \xi )}^ \wedge }} \right) \approx e({\xi ^ \wedge }) + J({\xi ^ \wedge })\Delta {\xi ^ \wedge }
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
  {\mathbf{I}}&{ - {\mathbf{P}}_c^\^} 
\end{array}} \right]
```
将两个雅可比矩阵相乘即可得到所求的雅可比矩阵。
### 求雅可比矩阵$`J_3`$和$`J_4`$
$`J_3`$与$`J_1`$完全相同，下面推导$`J_4`$：
```math
{{\mathbf{J}}_4} = \frac{{\partial {\mathbf{P}}_c^i}}{{\partial {\mathbf{P}}_w^i}} = \frac{{\partial \left( {{{\mathbf{R}}_{cw}}{\mathbf{P}}_w^i + t} \right)}}{{\partial {\mathbf{P}}_w^i}} = {{\mathbf{R}}_{cw}}
```
