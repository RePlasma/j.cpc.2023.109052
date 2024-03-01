cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(inferenceTime_libtorch)

set(TARGET_NAME inferenceTime_libtorch) # New line to define the variable

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp) # Using the variable
target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}") # Using the variable
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17) # Using the variable

# Windows specific post-build event
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${TARGET_NAME} # Using the variable
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${TARGET_NAME}>) # Using the variable
endif (MSVC)
