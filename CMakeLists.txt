cmake_minimum_required(VERSION 3.5)

project(
  external_forces_estimator
  LANGUAGES CXX
  VERSION 1.0.0)

include(CTest)
enable_testing()

# Find mc_rtc if we are not building inside mc_rtc itself
find_package(mc_rtc REQUIRED)
if(NOT TARGET mc_rtc::mc_control)
  message(FATAL_ERROR "mc_rtc::mc_control is required to build this project")
endif() 

# Configure and install the plugin configuration
configure_file(etc/ExternalForcesEstimator.in.yaml
               "${CMAKE_CURRENT_BINARY_DIR}/etc/ExternalForcesEstimator.yaml")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/etc/ExternalForcesEstimator.yaml"
        DESTINATION "${MC_PLUGINS_RUNTIME_INSTALL_PREFIX}/etc")

set(plugin_SRC src/ExternalForcesEstimator.cpp)
set(plugin_HDR src/ExternalForcesEstimator.h)

# The add_plugin macro is provided by mc_rtc
add_plugin(ExternalForcesEstimator ${plugin_SRC} ${plugin_HDR})
