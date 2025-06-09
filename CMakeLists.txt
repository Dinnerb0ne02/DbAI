cmake_minimum_required(VERSION 3.10)

project(
    DbAI VERSION 0.0.1
    DESCRIPTION "ANN framework using transformer agent"
    LANGUAGES C
)

set(AUTHOR "Dinnerb0ne Ma <tomma_2022@outlook.com>")
set(COPYRIGHT "Copyright (C) 2025 ${AUTHOR}")
set(LICENSE "Apache-2.0")

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)


set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_INCLUDE_DIR ${PROJECT_ROOT_DIR}/include)
set(PROJECT_SRC_DIR ${PROJECT_ROOT_DIR}/src)
set(PROJECT_TESTS_DIR ${PROJECT_ROOT_DIR}/tests)

include_directories(${PROJECT_INCLUDE_DIR})

# include_directories(${PROJECT_INCLUDE_DIR}/network)
# include_directories(${PROJECT_INCLUDE_DIR}/activation)

file(GLOB_RECURSE PROJECT_SOURCES "${PROJECT_SRC_DIR}/*.c")
file(GLOB_RECURSE PROJECT_TEST_SOURCES "${PROJECT_TESTS_DIR}/*.c")

file(GLOB_RECURSE ALL_SRC "${PROJECT_SRC_DIR}/*/*.c")


# file(GLOB_RECURSE ACTIVATION_SOURCES ${SRC_DIR}/activation/*.c)
# file(GLOB_RECURSE DATA_SOURCES ${SRC_DIR}/data/*.c)
# file(GLOB_RECURSE LOSS_SOURCES ${SRC_DIR}/loss/*.c)
# file(GLOB_RECURSE LOSS_SOURCES ${SRC_DIR}/loss/*.c)

# 编译选项
add_compile_options(-static -std=${CMAKE_C_STANDARD} -Wall -O0)

add_executable(DbAI
    ${PROJECT_SOURCES}
    ${ALL_SRC}
    # 添加其他网络相关的源文件
    #${ACTIVATION_SOURCES}  # 添加激活函数的源文件
    # ${SRC_DIR}/network/layer.c
    # ${SRC_DIR}/network/neural_network.c
    # 其他源文件...
)



# Project metadata
set_target_properties(DbAI PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    DESCRIPTION "${PROJECT_DESCRIPTION}"
)
set_target_properties(DbAI PROPERTIES
    AUTHOR "${AUTHOR}"
    LICENSE "${LICENSE}"
)
message(STATUS "Project: ${PROJECT_NAME} version ${PROJECT_VERSION}")
message(STATUS "Description: ${PROJECT_DESCRIPTION}")
message(STATUS "Author: ${AUTHOR}")
message(STATUS "Copyright: ${COPYRIGHT}")


enable_testing()

foreach(TEST_SOURCE ${PROJECT_TEST_SOURCES})
    string(REPLACE "${PROJECT_TESTS_DIR}/" "" TEST_NAME ${TEST_SOURCE})
    string(REPLACE ".c" "" TEST_NAME ${TEST_NAME})
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
                   ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif()

install(TARGETS DbAI RUNTIME DESTINATION bin)
