cmake_minimum_required(VERSION 3.16.3)
project(khisto)
include(FetchContent)

# Extract the khiops version from pyproject.toml
# The version is specified in the "tool.khiops" section as "khiops-version = "x.y.z""
# It must correspond to a tag in the khiops repository
file(READ pyproject.toml PYPROJECT_FILE)
string(REGEX MATCH "khiops-version = \"([^\"]*)\"" _ ${PYPROJECT_FILE})
set(KHIOPS_VERSION ${CMAKE_MATCH_1})
if(NOT KHIOPS_VERSION)
  message(FATAL_ERROR "Could not extract khiops version from pyproject.toml")
endif()
message(STATUS "Khiops version: ${KHIOPS_VERSION}")

# Download and build khiops from the specified tag
fetchcontent_Declare(
  khiops
  GIT_REPOSITORY https://github.com/KhiopsML/khiops.git
    GIT_TAG ${KHIOPS_VERSION})
FetchContent_MakeAvailable(khiops)