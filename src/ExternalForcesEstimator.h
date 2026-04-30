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

enum class TorqueSourceType
{
  CommandedTorque,
  CurrentMeasurement,
  MotorTorqueMeasurement,
  JointTorqueMeasurement,
};

enum class FloatingBaseMode
{
  FullGeneralized,
  Decoupled,
};

namespace mc_plugin
{

struct EstimatorBackend;

struct ExternalForcesEstimator : public mc_control::GlobalPlugin
{
  void init(mc_control::MCGlobalController & controller, const mc_rtc::Configuration & config) override;

  void reset(mc_control::MCGlobalController & controller) override;

  void before(mc_control::MCGlobalController &) override;

  void after(mc_control::MCGlobalController & controller) override;

  mc_control::GlobalPlugin::GlobalPluginConfiguration configuration() override;

  ~ExternalForcesEstimator() override;

  void computeForwardDynamic(mc_control::MCGlobalController & controller);
  void computeCHatPc0Hat(mc_control::MCGlobalController & controller, const rbd::MultiBodyConfig & mbc);
  void addGui(mc_control::MCGlobalController & controller);
  void addLog(mc_control::MCGlobalController & controller);
  void removeLog(mc_control::MCGlobalController & controller);

  struct ForceFusionState
  {
    Eigen::VectorXd sensorTorques;
    Eigen::VectorXd filteredSensorTorques;
    Eigen::VectorXd fusedTorques;
    Eigen::VectorXd publishedTorques;
    Eigen::VectorXd filteredPublishedTorques;
    sva::ForceVecd fusedWrench = sva::ForceVecd::Zero();
    sva::ForceVecd residualWrench = sva::ForceVecd::Zero();
    sva::ForceVecd unfilteredWrench = sva::ForceVecd::Zero();
    sva::ForceVecd filteredSensorWrench = sva::ForceVecd::Zero();
    Eigen::Vector6d sensorWrench = Eigen::Vector6d::Zero();
  };

  struct SpeedObserverState
  {
    Eigen::VectorXd residual;
    Eigen::VectorXd integral;
  };

  struct RuntimeDiagnostics
  {
    Eigen::VectorXd alphas;
    Eigen::VectorXd gravity;
    Eigen::VectorXd inputTorque;
    Eigen::VectorXd commandedAcceleration;
  };

  struct EstimatorInputs
  {
    mc_control::MCGlobalController * controller = nullptr;
    const mc_rbdyn::Robot * robot = nullptr;
    const mc_rbdyn::Robot * realRobot = nullptr;
    rbd::MultiBodyConfig mbc;
    Eigen::VectorXd qdot;
    Eigen::VectorXd tau;
    Eigen::VectorXd commandedAcceleration;
    Eigen::VectorXd gravity;
    Eigen::MatrixXd coriolisMatrix;
    int preservedPrefix = 0;
    bool warnWhenInactive = false;
    bool logPluginState = false;
  };

  struct EstimatorResult
  {
    Eigen::VectorXd torques;
    Eigen::VectorXd accelerations;
    ForceFusionState forceFusion;
    std::vector<sva::ForceVecd> sensorEstimations;
    int preservedPrefix = 0;
    bool warnWhenInactive = false;
    bool logPluginState = false;
  };

  EstimatorResult computeForFixedBase(mc_control::MCGlobalController & controller);
  EstimatorResult computeForFloatingBase(mc_control::MCGlobalController & controller);
  EstimatorResult computeForFloatingBaseFullGeneralized(mc_control::MCGlobalController & controller);
  EstimatorResult computeForFloatingBaseDecoupled(mc_control::MCGlobalController & controller);

private:
  struct ResidualObserverState
  {
    Eigen::VectorXd integralFull;
    Eigen::VectorXd residualFull;
    Eigen::VectorXd integralJoint;
    Eigen::VectorXd jointResidual;
    Eigen::VectorXd integralBase;
    Eigen::VectorXd baseResidual;
    Eigen::VectorXd rotorInertiaResidual;
    Eigen::VectorXd rotorInertiaIntegral;
  };

  void initializeActiveJoints(const mc_rbdyn::Robot & robot);
  void loadConfiguration(const mc_rtc::Configuration & config);
  void initializeEstimatorState(const mc_rbdyn::Robot & robot, const Eigen::VectorXd & qdot);
  EstimatorInputs buildEstimatorInputs(mc_control::MCGlobalController & controller,
                                       int preservedPrefix,
                                       bool warnWhenInactive,
                                       bool logPluginState);
  rbd::MultiBodyConfig prepareRuntimeInputs(const mc_rbdyn::Robot & robot,
                                            const mc_rbdyn::Robot & realRobot,
                                            int preservedPrefix,
                                            Eigen::VectorXd & qdot,
                                            Eigen::VectorXd & tau);
  void updateDiagnostics(const EstimatorInputs & inputs);
  Eigen::VectorXd readMeasuredTorque(const mc_rbdyn::Robot & robot,
                                     const mc_rbdyn::Robot & realRobot,
                                     int preservedPrefix) const;
  bool updatePluginActivation(mc_control::MCGlobalController & controller) const;
  void updateSpeedResidualDatastore(mc_control::MCGlobalController & controller);
  void applyEstimatorResult(const EstimatorResult & result);
  ForceFusionState computeFixedBaseForceFusion(const mc_rbdyn::Robot & robot,
                                               const mc_rbdyn::Robot & realRobot,
                                               const rbd::MultiBodyConfig & mbc,
                                               const Eigen::VectorXd & jointResidual);
  ForceFusionState computeFullGeneralizedForceFusion(const mc_rbdyn::Robot & robot,
                                                     const rbd::MultiBodyConfig & mbc,
                                                     const Eigen::VectorXd & activeResidual);
  std::vector<sva::ForceVecd> estimateFloatingBaseSensorWrenches(const mc_rbdyn::Robot & robot,
                                                                 const mc_rbdyn::Robot & realRobot,
                                                                 const rbd::MultiBodyConfig & mbc,
                                                                 const Eigen::MatrixXd & FT,
                                                                 const Eigen::MatrixXd & I_c_0_inv,
                                                                 const Eigen::VectorXd & residualFB) const;
  /** Write the estimated external torques and equivalent accelerations to both control and real robots. */
  void updateRobotExternalForces(mc_control::MCGlobalController & controller,
                                 const mc_rbdyn::Robot & robot,
                                 const mc_rbdyn::Robot & realRobot,
                                 const Eigen::VectorXd & torques,
                                 const Eigen::VectorXd & accelerations);
  /** Clear the external torques and equivalent accelerations written by this estimator. */
  void clearRobotExternalForces(mc_control::MCGlobalController & controller,
                                const mc_rbdyn::Robot & realRobot) const;
  /** Resolve plugin ownership and either update the robot with the estimate or clear the estimator contribution. */
  void resolveAndUpdateRobot(mc_control::MCGlobalController & controller,
                             const mc_rbdyn::Robot & robot,
                             const mc_rbdyn::Robot & realRobot,
                             Eigen::VectorXd torques,
                             Eigen::VectorXd accelerations,
                             int preservedPrefix,
                             bool warnWhenInactive,
                             bool logPluginState);
  void resetResidualGain(double gain);

  std::unique_ptr<EstimatorBackend> backend_;

  std::vector<int> activeJointIndices; // A vector of the same size as the number of joints, with 1 for
                                       // estimated joints and 0 for non-estimated joints
  int actuatedDofNumber = 0;

  bool robotIsFloatingBase = false;
  int dofNumber = 0;
  int counter = 0;
  double dt = 0.0;
  bool verbose = false;
  bool isActive = true;

  double residualGain = 0.0;
  std::string referenceFrame;

  rbd::Jacobian jac;
  std::unique_ptr<rbd::Coriolis> coriolis;
  rbd::ForwardDynamics forwardDynamics;

  Eigen::VectorXd pzero;
  ResidualObserverState residualObserver_;
  ForceFusionState forceFusion_;

  // Used for collision avoidance observer, not for the control
  SpeedObserverState speedObserver_;
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
  RuntimeDiagnostics diagnostics_;
};

} // namespace mc_plugin
