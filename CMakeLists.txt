cmake_minimum_required(VERSION 3.18)

option(PCMATH_BUILD_SAMPLES "pep-cuda-math whether build samples" ON)
option(PCMATH_NAMESPACE_ALIAS "pep-cuda-math whether define namespace alias (pcm = pep::cmath)" ON)

if(PCMATH_BUILD_SAMPLES)
    project(pep-cuda-math LANGUAGES C CXX CUDA)
else()
    project(pep-cuda-math)
endif()

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL "In-source builds not allowed")
endif()

add_library(${PROJECT_NAME} INTERFACE)

target_sources(${PROJECT_NAME}
    INTERFACE
        include/pcmath/cuda_macro_utils.hpp
        include/pcmath/mat.hpp
        include/pcmath/pcmath.hpp
        include/pcmath/utils.hpp
        include/pcmath/vec.hpp
)

target_include_directories(${PROJECT_NAME}
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(${PROJECT_NAME}
    INTERFACE cxx_std_20
)

if(PCMATH_NAMESPACE_ALIAS)
    target_compile_definitions(${PROJECT_NAME}
        INTERFACE PCMATH_NAMESPACE_ALIAS
    )
endif()

if(PCMATH_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()