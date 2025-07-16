/*
 * Copyright 2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_control/GlobalPlugin.h>
#include <mc_rtc/log/FlatLog.h>

#include <RBDyn/Coriolis.h>
#include <RBDyn/FA.h>
#include <RBDyn/FK.h>
#include <RBDyn/FV.h>
#include <RBDyn/MultiBody.h>
#include <RBDyn/MultiBodyConfig.h>
#include <Eigen/src/Core/Matrix.h>

#include <mc_rbdyn/VirtualTorqueSensor.h>

#include <mc_tvm/Robot.h>

enum TorqueSourceType
{
  CommandedTorque,
  CurrentMeasurement,
  MotorTorqueMeasurement,
  JointTorqueMeasurement,
};

namespace mc_plugin
{

struct ExternalForcesEstimator : public mc_control::GlobalPlugin
{
  void init(mc_control::MCGlobalController & controller, const mc_rtc::Configuration & config) override;

  void reset(mc_control::MCGlobalController & controller) override;

  void before(mc_control::MCGlobalController &) override;

  void after(mc_control::MCGlobalController & controller) override;

  mc_control::GlobalPlugin::GlobalPluginConfiguration configuration() override;

  ~ExternalForcesEstimator() override;

  void computeForFixedBase(mc_control::MCGlobalController & controller);
  void computeForFloatingBase(mc_control::MCGlobalController & controller);
  void computeForwardDynamic(mc_control::MCGlobalController & controller);
  void computeCHatPc0Hat(mc_control::MCGlobalController & controller);
  void addGui(mc_control::MCGlobalController & controller);
  void addLog(mc_control::MCGlobalController & controller);
  void removeLog(mc_control::MCGlobalController & controller);

private:
  bool robotIsFloatingBase;
  int dofNumber;
  int counter;
  double dt;
  bool verbose;
  bool isActive;

  double residualGains;
  std::string referenceFrame;

  rbd::Jacobian jac;
  rbd::Coriolis * coriolis;
  rbd::ForwardDynamics forwardDynamics;

  Eigen::VectorXd pzero;

  Eigen::VectorXd integralTermIntern;
  Eigen::VectorXd internResidual;
  Eigen::VectorXd integralTermExtern;
  Eigen::VectorXd externResidual;
  Eigen::VectorXd residualWithRotorInertia;
  Eigen::VectorXd integralTermWithRotorInertia;

  Eigen::VectorXd FTSensorTorques;
  Eigen::VectorXd prevFTSensorTorques;
  Eigen::VectorXd filteredFTSensorTorques;
  Eigen::VectorXd newExternalTorques;
  Eigen::VectorXd externalTorques;
  Eigen::VectorXd filteredExternalTorques;
  sva::ForceVecd externalForces;
  sva::ForceVecd externalForcesResidual;
  sva::ForceVecd newExternalForces;
  sva::ForceVecd filteredFTSensorForces;
  Eigen::Vector6d externalForcesFT;
  mc_rbdyn::VirtualTorqueSensor * extTorqueSensor;

  // Used for collision avoidance observer, not for the control
  Eigen::VectorXd residualSpeed;
  Eigen::VectorXd integralTermSpeed;
  double residualSpeedGain;

  // Force sensor
  bool use_force_sensor_;
  TorqueSourceType tau_mes_src_;

  std::string ft_sensor_name_;

  // Floating base residual computation
  Eigen::VectorXd internalResidual;
  Eigen::Vector6d externalResidual;
  Eigen::MatrixXd prevH;
  Eigen::MatrixXd prevF;
  Eigen::MatrixXd prevI_c_0;
  Eigen::MatrixXd mimicExclusion;

  std::vector<sva::ForceVecd> EstimationAtFTSensors;

  // Custom forward dynamic calculation
  Eigen::MatrixXd H;
  Eigen::MatrixXd F;
  Eigen::MatrixXd Ic0;
  Eigen::MatrixXd Hd;
  Eigen::MatrixXd Fd;
  Eigen::MatrixXd Ic0d;

  Eigen::VectorXd c_hat;

  Eigen::IOFormat format;

  // Logging
  Eigen::VectorXd alphas;
};

} // namespace mc_plugin
