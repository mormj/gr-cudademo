find_package(PkgConfig)

PKG_CHECK_MODULES(PC_CUDADEMO cudademo)

FIND_PATH(
    CUDADEMO_INCLUDE_DIRS
    NAMES cudademo/api.h
    HINTS $ENV{CUDADEMO_DIR}/include
        ${PC_CUDADEMO_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    CUDADEMO_LIBRARIES
    NAMES gnuradio-cudademo
    HINTS $ENV{CUDADEMO_DIR}/lib
        ${PC_CUDADEMO_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/cudademoTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CUDADEMO DEFAULT_MSG CUDADEMO_LIBRARIES CUDADEMO_INCLUDE_DIRS)
MARK_AS_ADVANCED(CUDADEMO_LIBRARIES CUDADEMO_INCLUDE_DIRS)
