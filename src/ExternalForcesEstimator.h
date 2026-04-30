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
#include <memory>

#include <mc_tvm/Robot.h>

enum TorqueSourceType
{
  CommandedTorque,
  CurrentMeasurement,
  MotorTorqueMeasurement,
  JointTorqueMeasurement,
};

enum FloatingBaseMode
{
  FullGeneralized,
  Decoupled,
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
  void computeForFloatingBaseFullGeneralized(mc_control::MCGlobalController & controller,
                                             const mc_rbdyn::Robot & robot,
                                             const mc_rbdyn::Robot & realRobot,
                                             const rbd::MultiBodyConfig & mbc,
                                             const Eigen::VectorXd & qdot,
                                             const Eigen::VectorXd & tau,
                                             const Eigen::MatrixXd & coriolisMatrix);
  void computeForFloatingBaseDecoupled(mc_control::MCGlobalController & controller,
                                       const mc_rbdyn::Robot & robot,
                                       const mc_rbdyn::Robot & realRobot,
                                       const rbd::MultiBodyConfig & mbc,
                                       const Eigen::VectorXd & qdot,
                                       const Eigen::VectorXd & tau,
                                       const Eigen::MatrixXd & coriolisMatrix);
  void computeForwardDynamic(mc_control::MCGlobalController & controller);
  void computeCHatPc0Hat(mc_control::MCGlobalController & controller, const rbd::MultiBodyConfig & mbc);
  void addGui(mc_control::MCGlobalController & controller);
  void addLog(mc_control::MCGlobalController & controller);
  void removeLog(mc_control::MCGlobalController & controller);

private:
  void initializeActiveJoints(const mc_rbdyn::Robot & robot);
  void loadConfiguration(const mc_rtc::Configuration & config);
  void initializeEstimatorState(const mc_rbdyn::Robot & robot, const Eigen::VectorXd & qdot);
  rbd::MultiBodyConfig prepareRuntimeInputs(const mc_rbdyn::Robot & robot,
                                            const mc_rbdyn::Robot & realRobot,
                                            int preservedPrefix,
                                            Eigen::VectorXd & qdot,
                                            Eigen::VectorXd & tau);
  Eigen::VectorXd readMeasuredTorque(const mc_rbdyn::Robot & robot,
                                     const mc_rbdyn::Robot & realRobot,
                                     int preservedPrefix) const;
  bool updatePluginActivation(mc_control::MCGlobalController & controller) const;
  void updateSpeedResidualDatastore(mc_control::MCGlobalController & controller);
  void publishExternalTorqueState(mc_control::MCGlobalController & controller,
                                  const mc_rbdyn::Robot & robot,
                                  const mc_rbdyn::Robot & realRobot,
                                  const Eigen::VectorXd & torques,
                                  const Eigen::VectorXd & accelerations);
  void clearExternalTorqueState(mc_control::MCGlobalController & controller,
                                const mc_rbdyn::Robot & realRobot) const;
  void finalizeExternalTorqueComputation(mc_control::MCGlobalController & controller,
                                         const mc_rbdyn::Robot & robot,
                                         const mc_rbdyn::Robot & realRobot,
                                         Eigen::VectorXd torques,
                                         Eigen::VectorXd accelerations,
                                         int preservedPrefix,
                                         bool warnWhenInactive,
                                         bool logPluginState);
  void resetResidualGain(double gain);

  std::vector<int> activeJointIndices; // A vector of the same size as the number of joints, with 1 for
                                       // estimated joints and 0 for non-estimated joints
  int actuatedDofNumber = 0;

  bool robotIsFloatingBase = false;
  int dofNumber = 0;
  int counter = 0;
  double dt = 0.0;
  bool verbose = false;
  bool isActive = true;

  double residualGains = 0.0;
  std::string referenceFrame;

  rbd::Jacobian jac;
  std::unique_ptr<rbd::Coriolis> coriolis;
  rbd::ForwardDynamics forwardDynamics;

  Eigen::VectorXd pzero;

  Eigen::VectorXd integralTermNormal;
  Eigen::VectorXd residualNormal;

  Eigen::VectorXd integralTermIntern;
  Eigen::VectorXd internResidual;
  Eigen::VectorXd integralTermExtern;
  Eigen::VectorXd externResidual;
  Eigen::VectorXd residualWithRotorInertia;
  Eigen::VectorXd integralTermWithRotorInertia;

  Eigen::VectorXd FTSensorTorques;
  Eigen::VectorXd filteredFTSensorTorques;
  Eigen::VectorXd newExternalTorques;
  Eigen::VectorXd externalTorques;
  Eigen::VectorXd filteredExternalTorques;
  sva::ForceVecd externalForces;
  sva::ForceVecd externalForcesResidual;
  sva::ForceVecd newExternalForces;
  sva::ForceVecd filteredFTSensorForces;
  Eigen::Vector6d externalForcesFT;

  // Used for collision avoidance observer, not for the control
  Eigen::VectorXd residualSpeed;
  Eigen::VectorXd integralTermSpeed;
  double residualSpeedGain;

  // Force sensor
  bool use_force_sensor_ = false;
  TorqueSourceType tau_mes_src_ = TorqueSourceType::JointTorqueMeasurement;
  FloatingBaseMode floating_base_mode_ = FloatingBaseMode::Decoupled;

  std::string ft_sensor_name_;

  // Floating base residual computation
  std::vector<sva::ForceVecd> EstimationAtFTSensors;

  // Custom forward dynamic calculation
  Eigen::MatrixXd H;
  Eigen::MatrixXd F;
  Eigen::MatrixXd Ic0;
  Eigen::MatrixXd Hd;
  Eigen::MatrixXd Fd;
  Eigen::MatrixXd Ic0d;

  Eigen::VectorXd c_hat;

  // Logging
  Eigen::VectorXd alphas;
  Eigen::VectorXd gravity;
  Eigen::VectorXd inputTorque;
  Eigen::VectorXd commandedAcceleration;
};

} // namespace mc_plugin
