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
set(STRIDER_SDK_DIR "${CMAKE_SOURCE_DIR}/SDK" CACHE PATH "Local SDK root")

# =============================================================================
# Dependencies (base)
# =============================================================================
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# =============================================================================
# RealSense2 (LOCAL SDK, absolute link to avoid -lrealsense2 issues)
# =============================================================================
set(REALSENSE2_PREFIX "${STRIDER_SDK_DIR}/realsense2" CACHE PATH "Local RealSense2 prefix")
set(REALSENSE2_INC_DIR "${REALSENSE2_PREFIX}/include")
set(REALSENSE2_LIB_DIR "${REALSENSE2_PREFIX}/lib")
if(NOT IS_DIRECTORY "${REALSENSE2_LIB_DIR}")
  set(REALSENSE2_LIB_DIR "${REALSENSE2_PREFIX}/lib64")
endif()

set(REALSENSE2_SO "${REALSENSE2_LIB_DIR}/librealsense2.so")
if(NOT EXISTS "${REALSENSE2_INC_DIR}/librealsense2/rs.hpp")
  message(FATAL_ERROR "RealSense headers not found: ${REALSENSE2_INC_DIR}/librealsense2/rs.hpp")
endif()
if(NOT EXISTS "${REALSENSE2_SO}")
  message(FATAL_ERROR "RealSense shared lib not found: ${REALSENSE2_SO}")
endif()

# =============================================================================
# Python / pybind11
# =============================================================================
if(STRIDER_ENABLE_MPC_PY)
  set(Python3_FIND_STRATEGY LOCATION CACHE STRING "Prefer explicit interpreter path" FORCE)

  # Candidate list: (1) active conda python, (2) system python3.11, (3) fallback python3.
  set(_STRIDER_PY_CANDIDATES "")
  if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/python3")
    list(APPEND _STRIDER_PY_CANDIDATES "$ENV{CONDA_PREFIX}/bin/python3")
  endif()

  find_program(_STRIDER_PY311 NAMES python3.11 HINTS /usr/bin /usr/local/bin)
  if(_STRIDER_PY311)
    list(APPEND _STRIDER_PY_CANDIDATES "${_STRIDER_PY311}")
  endif()
  find_program(_STRIDER_PY3 NAMES python3 HINTS /usr/bin /usr/local/bin)
  if(_STRIDER_PY3)
    list(APPEND _STRIDER_PY_CANDIDATES "${_STRIDER_PY3}")
  endif()

  # Pick the first interpreter that can import acados_template.
  set(_STRIDER_PY_OK "")
  foreach(_py IN LISTS _STRIDER_PY_CANDIDATES)
    execute_process(
      COMMAND "${_py}" -c "import acados_template"
      RESULT_VARIABLE _rc
      OUTPUT_QUIET ERROR_QUIET
    )
    if(_rc EQUAL 0)
      set(_STRIDER_PY_OK "${_py}")
      break()
    endif()
  endforeach()

  if(_STRIDER_PY_OK)
    set(Python3_EXECUTABLE "${_STRIDER_PY_OK}" CACHE FILEPATH "Python used for embedded MPC" FORCE)
  endif()

  # Require 3.11+ (your working configuration); adjust if you later support other versions.
  find_package(Python3 3.11 COMPONENTS Interpreter Development.Embed REQUIRED)

  # Hard fail early with a clear message if acados_template is still not importable.
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import acados_template"
    RESULT_VARIABLE _acados_template_rc
    OUTPUT_QUIET ERROR_QUIET
  )
  if(NOT _acados_template_rc EQUAL 0)
    message(FATAL_ERROR
      "Selected Python (${Python3_EXECUTABLE}) cannot import 'acados_template'.\n"
      "Fix: install/make it importable in this interpreter, or configure with -DPython3_EXECUTABLE=... "
      "pointing to a Python where 'import acados_template' works."
    )
  endif()

  unset(_STRIDER_PY_CANDIDATES)
  unset(_STRIDER_PY311)
  unset(_STRIDER_PY3)
  unset(_STRIDER_PY_OK)
  unset(_acados_template_rc)

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
# Dynamixel SDK (source-tree paths + built .so)
# =============================================================================
if(STRIDER_ENABLE_DXL)
  set(DXL_SRC_ROOT "${STRIDER_SDK_DIR}/DynamixelSDK_src/c++" CACHE PATH "DynamixelSDK C++ source root")
  set(DXL_INC_DIR  "${DXL_SRC_ROOT}/include")
  set(DXL_LIB_DIR  "${DXL_SRC_ROOT}/build/linux64")
  set(DXL_HEADER   "${DXL_INC_DIR}/dynamixel_sdk/dynamixel_sdk.h")
  set(DXL_SO       "${DXL_LIB_DIR}/libdxl_x64_cpp.so")

  if(NOT EXISTS "${DXL_HEADER}")
    message(FATAL_ERROR "dynamixel_sdk.h not found: ${DXL_HEADER}")
  endif()
  if(NOT EXISTS "${DXL_SO}")
    message(FATAL_ERROR "libdxl_x64_cpp.so not found: ${DXL_SO} (build it: make -C ${DXL_LIB_DIR})")
  endif()

  add_library(dxl_deps INTERFACE)
  target_include_directories(dxl_deps INTERFACE "${DXL_INC_DIR}")
  target_link_libraries(dxl_deps INTERFACE "${DXL_SO}")
  target_link_options(dxl_deps INTERFACE "-Wl,-rpath,${DXL_LIB_DIR}")
endif()

# =============================================================================
# OptiTrack / libmotioncapture (motion_capture_tracking source-tree)
#   NOTE: build libmotioncapture with closed_source/vrpn/vicon/qualisys OFF
# =============================================================================
if(STRIDER_ENABLE_OPTI)
  set(STRIDER_SDK_DIR "${CMAKE_SOURCE_DIR}/SDK" CACHE PATH "Local SDK root")
  set(OPTI_SDK_DIR "${STRIDER_SDK_DIR}/optiSDK" CACHE PATH "Opti SDK root")

  # motion_capture_tracking root (you cloned here)
  set(MCT_ROOT "${OPTI_SDK_DIR}/motion_capture_tracking")

  # libmotioncapture source root inside the repo
  set(LIBMOCAP_ROOT  "${MCT_ROOT}/deps/libmotioncapture")
  set(LIBMOCAP_INC   "${LIBMOCAP_ROOT}/include")
  set(LIBMOCAP_BUILD "${LIBMOCAP_ROOT}/build")

  # Header check
  if(NOT EXISTS "${LIBMOCAP_INC}/libmotioncapture/motioncapture.h")
    message(FATAL_ERROR "motioncapture.h not found: ${LIBMOCAP_INC}/libmotioncapture/motioncapture.h")
  endif()

  # Find built library (your output is typically liblibmotioncapture.a)
  find_library(LIBMOCAP_LIB
    NAMES motioncapture libmotioncapture liblibmotioncapture
    PATHS "${LIBMOCAP_BUILD}"
    NO_DEFAULT_PATH
  )
  if(NOT LIBMOCAP_LIB)
    message(FATAL_ERROR
      "libmotioncapture not found under: ${LIBMOCAP_BUILD}\n"
      "Expected liblibmotioncapture.a (or .so)."
    )
  endif()

  add_library(opti_deps INTERFACE)

  target_include_directories(opti_deps INTERFACE
    "${LIBMOCAP_INC}"
  )

  # Minimal link: assumes you built libmotioncapture with
  # OPTITRACK=ON, OPTITRACK_CLOSED_SOURCE=OFF, VRPN/VICON/QUALISYS=OFF
  target_link_libraries(opti_deps INTERFACE
    "${LIBMOCAP_LIB}"
    dl
  )

  # rpath is harmless for static, helpful if you ever build shared
  target_link_options(opti_deps INTERFACE
    "-Wl,-rpath,${LIBMOCAP_BUILD}"
  )

  unset(LIBMOCAP_LIB)
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

# core sources (gate by options)
target_sources(strider_core PRIVATE
  src/mmap_logger.cpp
  src/sbus.cpp
  src/t265.cpp
  src/teensy.cpp
)

if(STRIDER_ENABLE_DXL)
  target_sources(strider_core PRIVATE src/dynamixel.cpp)
endif()

if(STRIDER_ENABLE_MPC_PY)
  target_sources(strider_core PRIVATE src/mpc_wrapper.cpp)
endif()

if(STRIDER_ENABLE_OPTI)
  target_sources(strider_core PRIVATE src/opti.cpp)
endif()

target_include_directories(strider_core PUBLIC
  "${CMAKE_SOURCE_DIR}/include"
  "${REALSENSE2_INC_DIR}"
)

target_link_libraries(strider_core PUBLIC
  Threads::Threads
  Eigen3::Eigen
  dl
  "${REALSENSE2_SO}"
)

# Runtime rpath for RealSense2
target_link_options(strider_core PUBLIC
  "-Wl,-rpath,${REALSENSE2_LIB_DIR}"
)

if(STRIDER_ENABLE_DXL)
  target_link_libraries(strider_core PUBLIC dxl_deps)
endif()

if(STRIDER_ENABLE_MPC_PY)
  target_link_libraries(strider_core PUBLIC
    pybind11::embed
    Python3::Python
  )
endif()

if(STRIDER_ENABLE_OPTI)
  target_link_libraries(strider_core PUBLIC opti_deps)
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
  set(STRIDER_RPATH_LIST "$ORIGIN" "${STRIDER_ACADOS_GEN_DIR}" "${REALSENSE2_LIB_DIR}")

  if(STRIDER_ENABLE_MPC_PY AND DEFINED ENV{CONDA_PREFIX})
    list(APPEND STRIDER_RPATH_LIST "$ENV{CONDA_PREFIX}/lib")
  endif()

  if(STRIDER_ENABLE_OPTI)
    # only needed if you end up using shared libs there (harmless otherwise)
    list(APPEND STRIDER_RPATH_LIST "${LIBMOCAP_BUILD}")
    if(IS_DIRECTORY "${NATNET_LIBDIR}")
      list(APPEND STRIDER_RPATH_LIST "${NATNET_LIBDIR}")
    endif()
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
  set(STRIDER_LD_PATH_LIST "${STRIDER_ACADOS_GEN_DIR}" "${REALSENSE2_LIB_DIR}")

  if(STRIDER_ENABLE_MPC_PY AND DEFINED ENV{CONDA_PREFIX})
    list(APPEND STRIDER_LD_PATH_LIST "$ENV{CONDA_PREFIX}/lib")
  endif()

  if(STRIDER_ENABLE_OPTI)
    list(APPEND STRIDER_LD_PATH_LIST "${LIBMOCAP_BUILD}")
    if(IS_DIRECTORY "${NATNET_LIBDIR}")
      list(APPEND STRIDER_LD_PATH_LIST "${NATNET_LIBDIR}")
    endif()
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