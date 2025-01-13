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
