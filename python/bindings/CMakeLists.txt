# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT cudademo_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT cudademo_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
    )

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND cudademo_python_files
    multiply_const_python.cc python_bindings.cc)

GR_PYBIND_MAKE_OOT(cudademo 
   ../..
   gr::cudademo
   "${cudademo_python_files}")

install(TARGETS cudademo_python DESTINATION ${GR_PYTHON_DIR}/cudademo COMPONENT pythonapi)
