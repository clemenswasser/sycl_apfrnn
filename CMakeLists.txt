cmake_minimum_required(VERSION 4.2)

if (CMAKE_HOST_WIN32)
    find_package(oneDPLWindowsIntelLLVM CONFIG REQUIRED)
endif()

project(sycl_apfrnn LANGUAGES CXX)

find_package(IntelSYCL CONFIG REQUIRED)
find_package(oneDPL CONFIG REQUIRED)

add_executable(sycl_apfrnn main.cpp)
add_sycl_to_target(TARGET sycl_apfrnn SOURCES main.cpp)
target_link_libraries(sycl_apfrnn PRIVATE oneDPL)
if(WIN32)
    target_compile_options(sycl_apfrnn PRIVATE /clang:-fdebug-info-for-profiling)
    target_link_options(sycl_apfrnn PRIVATE -Xsycl-target-backend=spir64 "-g")
else()
    target_compile_options(sycl_apfrnn PRIVATE -gline-tables-only -fdebug-info-for-profiling)
endif()
