cmake_minimum_required(VERSION 3.16)
project(strider LANGUAGES CXX)

# =============================================================================
# Platform guard: Linux x86_64 only
# =============================================================================
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Linux only. Detected: ${CMAKE_SYSTEM_NAME}")
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" STRIDER_PROC)
if(NOT STRIDER_PROC MATCHES "^(x86_64|amd64)$")
  message(FATAL_ERROR "x86_64 only. Detected: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
unset(STRIDER_PROC)

# =============================================================================
# C++ standard
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Options
# =============================================================================
option(STRIDER_BUILD_APP       "Build strider executable" ON)
option(STRIDER_ENABLE_OPTI     "Enable Opti/libmotioncapture integration" ON)
option(STRIDER_ENABLE_MPC_PY   "Enable embedded Python MPC (pybind11::embed)" ON)
option(STRIDER_ENABLE_DXL      "Enable Dynamixel SDK integration" ON)

# =============================================================================
# Paths
# =============================================================================
set(STRIDER_PY_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH "Python root dir (contains acados/)")
set(STRIDER_ACADOS_GEN_DIR "${CMAKE_SOURCE_DIR}/acados" CACHE PATH "acados dir (runtime lib path)")

# =============================================================================
# Dependencies (base)
# =============================================================================
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(REALSENSE2 REQUIRED IMPORTED_TARGET realsense2)

# =============================================================================
# Python / pybind11
# =============================================================================
if(STRIDER_ENABLE_MPC_PY)
  # Prefer Conda Python if available.
  if(DEFINED ENV{CONDA_PREFIX})
    set(Python3_ROOT_DIR "$ENV{CONDA_PREFIX}" CACHE PATH "Conda env root" FORCE)
    set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python3" CACHE FILEPATH "Conda python" FORCE)
  endif()

  find_package(Python3 COMPONENTS Interpreter Development.Embed REQUIRED)

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT PYBIND11_CMAKEDIR)
    message(FATAL_ERROR "pybind11 not found in this Python env (python -m pybind11 --cmakedir failed)")
  endif()

  find_package(pybind11 CONFIG REQUIRED
    PATHS "${PYBIND11_CMAKEDIR}"
    NO_DEFAULT_PATH
  )
endif()

# =============================================================================
# Dynamixel SDK
# =============================================================================
if(STRIDER_ENABLE_DXL)
  set(DEPS_DXL_ROOT "$ENV{HOME}/Pre_build/dynamixel_sdk" CACHE PATH "dynamixel_sdk prebuild root")

  set(DXL_INSTALL_DIR "${DEPS_DXL_ROOT}/install/dynamixel_sdk")
  set(DXL_INC_DIR     "${DXL_INSTALL_DIR}/include")
  set(DXL_LIB_DIR     "${DXL_INSTALL_DIR}/lib")
  set(DXL_SO          "${DXL_LIB_DIR}/libdynamixel_sdk.so")

  if(NOT EXISTS "${DXL_INC_DIR}/dynamixel_sdk/dynamixel_sdk.h")
    message(FATAL_ERROR "dynamixel_sdk.h not found: ${DXL_INC_DIR}/dynamixel_sdk/dynamixel_sdk.h")
  endif()

  if(NOT EXISTS "${DXL_SO}")
    message(FATAL_ERROR "libdynamixel_sdk.so not found: ${DXL_SO}")
  endif()

  add_library(dxl_deps INTERFACE)

  target_include_directories(dxl_deps INTERFACE
    "${DXL_INC_DIR}"
  )

  # Link with absolute .so (no ament/colcon env required)
  target_link_libraries(dxl_deps INTERFACE
    "${DXL_SO}"
  )

  # Ensure runtime can find libdynamixel_sdk.so without exporting LD_LIBRARY_PATH.
  target_link_options(dxl_deps INTERFACE
    "-Wl,-rpath,${DXL_LIB_DIR}"
  )
endif()

# =============================================================================
# OptiTrack / libmotioncapture
# =============================================================================
if(STRIDER_ENABLE_OPTI)
  set(DEPS_OPTI_ROOT "$ENV{HOME}/Pre_build/deps_opti" CACHE PATH "deps_opti root")

  set(LIBMOCAP_INC   "${DEPS_OPTI_ROOT}/libmotioncapture/include")
  set(LIBMOCAP_BUILD "${DEPS_OPTI_ROOT}/build/libmotioncapture")

  set(LIBMOCAP_A "${LIBMOCAP_BUILD}/liblibmotioncapture.a")
  set(VICON_A    "${LIBMOCAP_BUILD}/deps/vicon-datastream-sdk/libViconDataStreamSDK_CPP.a")
  set(QUALISYS_A "${LIBMOCAP_BUILD}/deps/qualisys_cpp_sdk/libqualisys_cpp_sdk.a")

  set(NATNET_LIBDIR "${DEPS_OPTI_ROOT}/libmotioncapture/deps/NatNetSDKCrossplatform/lib/ubuntu")

  set(BOOST_SYSTEM_SO "/usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0")
  set(BOOST_THREAD_SO "/usr/lib/x86_64-linux-gnu/libboost_thread.so.1.74.0")
  set(BOOST_ATOMIC_SO "/usr/lib/x86_64-linux-gnu/libboost_atomic.so.1.74.0")

  if(NOT EXISTS "${LIBMOCAP_INC}/libmotioncapture/motioncapture.h")
    message(FATAL_ERROR "motioncapture.h not found: ${LIBMOCAP_INC}/libmotioncapture/motioncapture.h")
  endif()

  foreach(p IN ITEMS "${LIBMOCAP_A}" "${VICON_A}" "${QUALISYS_A}")
    if(NOT EXISTS "${p}")
      message(FATAL_ERROR "Required Opti library not found: ${p}")
    endif()
  endforeach()

  foreach(p IN ITEMS "${BOOST_SYSTEM_SO}" "${BOOST_THREAD_SO}" "${BOOST_ATOMIC_SO}")
    if(NOT EXISTS "${p}")
      message(FATAL_ERROR "Boost .so not found: ${p}")
    endif()
  endforeach()

  if(NOT IS_DIRECTORY "${NATNET_LIBDIR}")
    message(FATAL_ERROR "NatNet lib dir not found: ${NATNET_LIBDIR}")
  endif()

  add_library(opti_deps INTERFACE)

  target_include_directories(opti_deps INTERFACE
    "${LIBMOCAP_INC}"
  )

  target_link_directories(opti_deps INTERFACE
    "${NATNET_LIBDIR}"
  )

  # Ensure runtime can find NatNet shared libs without exporting LD_LIBRARY_PATH.
  target_link_options(opti_deps INTERFACE
    "-Wl,-rpath,${NATNET_LIBDIR}"
  )

  target_link_libraries(opti_deps INTERFACE
    "${LIBMOCAP_A}"
    "${VICON_A}"
    "${QUALISYS_A}"

    "${BOOST_SYSTEM_SO}"
    "${BOOST_THREAD_SO}"
    "${BOOST_ATOMIC_SO}"

    NatNet
    dl
  )
endif()

# =============================================================================
# geometry_ctrl
# =============================================================================
add_library(geometry_ctrl STATIC
  src/geometry_ctrl/src/fdcl_control.cpp
  src/geometry_ctrl/src/fdcl_matrix_utils.cpp
)

target_include_directories(geometry_ctrl PUBLIC
  "${CMAKE_SOURCE_DIR}/src/geometry_ctrl/include"
  "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(geometry_ctrl PUBLIC
  Eigen3::Eigen
)

# =============================================================================
# strider_core
# =============================================================================
add_library(strider_core STATIC)

target_sources(strider_core PRIVATE
  src/dynamixel.cpp
  src/mmap_logger.cpp
  src/mpc_wrapper.cpp
  src/opti.cpp
  src/sbus.cpp
  src/t265.cpp
  src/teensy.cpp
)

if(STRIDER_ENABLE_DXL)
  target_sources(strider_core PRIVATE
    src/dynamixel.cpp
  )
endif()

if(STRIDER_ENABLE_MPC_PY)
  target_sources(strider_core PRIVATE
    src/mpc_wrapper.cpp
  )
endif()

if(STRIDER_ENABLE_OPTI)
  target_sources(strider_core PRIVATE
    src/opti.cpp
  )
endif()

target_include_directories(strider_core PUBLIC
  "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(strider_core PUBLIC
  Threads::Threads
  Eigen3::Eigen
  dl
  PkgConfig::REALSENSE2
)

if(STRIDER_ENABLE_DXL)
  target_link_libraries(strider_core PUBLIC
    dxl_deps
  )
endif()

if(STRIDER_ENABLE_MPC_PY)
  target_link_libraries(strider_core PUBLIC
    pybind11::embed
    Python3::Python
  )
endif()

if(STRIDER_ENABLE_OPTI)
  target_link_libraries(strider_core PUBLIC
    opti_deps
  )
endif()

# =============================================================================
# App
# =============================================================================
if(STRIDER_BUILD_APP)
  add_executable(strider apps/main.cpp)

  target_link_libraries(strider PRIVATE
    strider_core
    geometry_ctrl
  )

  # -------------------------
  # RPATH: keep binary self-contained as much as possible
  # -------------------------
  set(STRIDER_RPATH_LIST "$ORIGIN" "${STRIDER_ACADOS_GEN_DIR}")

  if(STRIDER_ENABLE_MPC_PY AND DEFINED ENV{CONDA_PREFIX})
    list(APPEND STRIDER_RPATH_LIST "$ENV{CONDA_PREFIX}/lib")
  endif()

  if(STRIDER_ENABLE_OPTI)
    list(APPEND STRIDER_RPATH_LIST "${NATNET_LIBDIR}")
  endif()

  if(STRIDER_ENABLE_DXL)
    list(APPEND STRIDER_RPATH_LIST "${DXL_LIB_DIR}")
  endif()

  list(JOIN STRIDER_RPATH_LIST ";" STRIDER_RPATH)

  set_target_properties(strider PROPERTIES
    BUILD_RPATH   "${STRIDER_RPATH}"
    INSTALL_RPATH "${STRIDER_RPATH}"
  )

  # -------------------------
  # Convenience run target
  # -------------------------
  set(STRIDER_LD_PATH_LIST "${STRIDER_ACADOS_GEN_DIR}")

  if(STRIDER_ENABLE_MPC_PY AND DEFINED ENV{CONDA_PREFIX})
    list(APPEND STRIDER_LD_PATH_LIST "$ENV{CONDA_PREFIX}/lib")
  endif()

  if(STRIDER_ENABLE_OPTI)
    list(APPEND STRIDER_LD_PATH_LIST "${NATNET_LIBDIR}")
  endif()

  if(STRIDER_ENABLE_DXL)
    list(APPEND STRIDER_LD_PATH_LIST "${DXL_LIB_DIR}")
  endif()

  list(JOIN STRIDER_LD_PATH_LIST ":" STRIDER_LD_PATH_PREFIX)

  set(STRIDER_RUN_ENV
    "LD_LIBRARY_PATH=${STRIDER_LD_PATH_PREFIX}:$ENV{LD_LIBRARY_PATH}"
  )

  if(STRIDER_ENABLE_MPC_PY)
    list(APPEND STRIDER_RUN_ENV
      "PYTHONPATH=${STRIDER_PY_DIR}"
    )
  endif()

  add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E env
            ${STRIDER_RUN_ENV}
            "$<TARGET_FILE:strider>"
    DEPENDS strider
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
endif()