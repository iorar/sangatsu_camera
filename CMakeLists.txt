cmake_minimum_required(VERSION 3.10)
project(sangatsu_camera)

# Create Main project
set(SOURCES
    action_detector.cpp
)

# set(HEADERS
#     # main.h
# )

add_executable(sangatsu_camera
    ${SOURCES}
#    ${HEADERS}
)


# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    target_include_directories(sangatsu_camera PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(sangatsu_camera ${OpenCV_LIBS})
endif()


