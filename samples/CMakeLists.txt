add_executable(host_test host_test.cpp)
target_link_libraries(host_test
    PRIVATE pep-cuda-math
)

add_executable(device_test device_test.cu)
target_link_libraries(device_test
    PRIVATE pep-cuda-math
)