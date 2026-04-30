#include "ExternalForcesEstimator.h"
#include "EstimatorMathUtils.h"
#include <mc_control/GlobalPluginMacros.h>
#include <mc_control/mc_global_controller.h>
#include <mc_rtc/logging.h>
#include <mc_rtc/unique_ptr.h>
#include <SpaceVecAlg/EigenTypedef.h>
#include <SpaceVecAlg/EigenUtility.h>
#include <SpaceVecAlg/SpaceVecAlg>
#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace mc_plugin
{

using detail::mapFullDofByJointName;
using detail::sanitizeTorqueInput;
using detail::scatterEntries;
using detail::selectCols;
using detail::selectEntries;
using detail::selectRows;
using detail::selectSubmatrix;
using detail::zeroInactiveEntries;

namespace
{

struct EstimatorConfig
{
  double residualGain = 0.0;
  std::string referenceFrame;
  bool verbose = false;
  std::string ftSensorName;
  bool useForceSensor = false;
  TorqueSourceType torqueSource = TorqueSourceType::JointTorqueMeasurement;
  FloatingBaseMode floatingBaseMode = FloatingBaseMode::Decoupled;
  double residualSpeedGain = 100.0;
};

constexpr const char * kPluginName = "ResidualEstimator";
constexpr const char * kPluginRegistryKey = "extTorquePlugin";
constexpr const char * kResidualOnlyKey = "EF_Estimator::getResidualOnly";
constexpr const char * kSpeedResidualKey = "speed_residual";

TorqueSourceType parseTorqueSourceType(const std::string & sourceType)
{
  if(sourceType == "CommandedTorque")
  {
    mc_rtc::log::info("Using CommandedTorque input");
    return TorqueSourceType::CommandedTorque;
  }
  if(sourceType == "CurrentMeasurement")
  {
    mc_rtc::log::info("Using CurrentMeasurement input");
    return TorqueSourceType::CurrentMeasurement;
  }
  if(sourceType == "MotorTorqueMeasurement")
  {
    mc_rtc::log::info("Using MotorTorqueMeasurement input");
    return TorqueSourceType::MotorTorqueMeasurement;
  }
  if(sourceType == "JointTorqueMeasurement")
  {
    mc_rtc::log::info("Using JointTorqueMeasurement input");
    return TorqueSourceType::JointTorqueMeasurement;
  }
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[ExternalForceEstimator] error in configuration with entry\"torque_source_type\".\n\tPossible values are: "
      "CommandedTorque, CurrentMeasurement, MotorTorqueMeasurement, JointTorqueMeasurement");
}

FloatingBaseMode parseFloatingBaseMode(const std::string & mode)
{
  if(mode.empty() || mode == "Decoupled")
  {
    return FloatingBaseMode::Decoupled;
  }
  if(mode == "FullGeneralized")
  {
    return FloatingBaseMode::FullGeneralized;
  }
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[ExternalForceEstimator] error in configuration with entry\"floating_base_mode\".\n\tPossible values are: "
      "Decoupled, FullGeneralized");
}

} // namespace

ExternalForcesEstimator::~ExternalForcesEstimator() = default;

void ExternalForcesEstimator::initializeActiveJoints(const mc_rbdyn::Robot & robot)
{
  activeJointIndices.clear();
  std::vector<std::string> activeJointNames;
  std::vector<std::string> activeGripperJoints;
  for(const auto & g : robot.grippers())
  {
    for(const auto & n : g.get().activeJoints())
    {
      activeGripperJoints.push_back(n);
    }
  }

  auto isActiveGripperJoint = [&](const std::string & jointName)
  { return std::find(activeGripperJoints.begin(), activeGripperJoints.end(), jointName) != activeGripperJoints.end(); };

  for(const auto & j : robot.mb().joints())
  {
    if(j.dof() != 1 || j.isMimic() || isActiveGripperJoint(j.name()))
    {
      continue;
    }
    mc_rtc::log::info("[ExternalForcesEstimator][Init] Estimated joint -> {}", j.name());
    activeJointNames.push_back(j.name());
  }

  int pos = robotIsFloatingBase ? 6 : 0;
  for(int jI = robotIsFloatingBase ? 1 : 0; jI < robot.mb().nrJoints(); ++jI)
  {
    const auto & j = robot.mb().joint(jI);
    if(j.dof() == 1)
    {
      if(std::find(activeJointNames.begin(), activeJointNames.end(), j.name()) != activeJointNames.end())
      {
        mc_rtc::log::info("[ExternalForcesEstimator][Init] Joint pos {} name {}", pos, j.name());
        activeJointIndices.push_back(pos);
      }
      pos++;
    }
  }
  actuatedDofNumber = static_cast<int>(activeJointIndices.size());
  mc_rtc::log::info("[ExternalForcesEstimator][Init] actuatedDofNumber = {}", actuatedDofNumber);
}

void ExternalForcesEstimator::loadConfiguration(const mc_rtc::Configuration & config)
{
  EstimatorConfig parsed;
  parsed.residualGain = config("residual_gain", 0.0);
  parsed.referenceFrame = config("reference_frame", std::string{});
  parsed.verbose = config("verbose", false);
  parsed.ftSensorName = config("ft_sensor_name", std::string{});
  parsed.useForceSensor = config("use_force_sensor", false);
  parsed.torqueSource = parseTorqueSourceType(config("torque_source_type", std::string{}));
  parsed.floatingBaseMode = parseFloatingBaseMode(config("floating_base_mode", std::string{"Decoupled"}));
  parsed.residualSpeedGain = config("residual_speed_gain", 100.0);

  residualGain = parsed.residualGain;
  referenceFrame = std::move(parsed.referenceFrame);
  verbose = parsed.verbose;
  ft_sensor_name_ = std::move(parsed.ftSensorName);
  use_force_sensor_ = parsed.useForceSensor;
  tau_mes_src_ = parsed.torqueSource;
  floating_base_mode_ = parsed.floatingBaseMode;
  residualSpeedGain = parsed.residualSpeedGain;
}

void ExternalForcesEstimator::initializeEstimatorState(const mc_rbdyn::Robot & robot, const Eigen::VectorXd & qdot)
{
  auto mbc = robot.mbc();
  mbc.alpha = rbd::vectorToDof(robot.mb(), qdot);
  rbd::forwardVelocity(robot.mb(), mbc);
  forwardDynamics.computeC(robot.mb(), mbc);
  forwardDynamics.computeH(robot.mb(), mbc);
  auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  pzero = selectEntries(inertiaMatrix * qdot, activeJointIndices);

  residualObserver_.integralBase = Eigen::VectorXd::Zero(6);
  residualObserver_.integralJoint = Eigen::VectorXd::Zero(actuatedDofNumber);
  residualObserver_.jointResidual = Eigen::VectorXd::Zero(actuatedDofNumber);
  residualObserver_.rotorInertiaIntegral = Eigen::VectorXd::Zero(actuatedDofNumber);
  residualObserver_.baseResidual = Eigen::VectorXd::Zero(6);
  residualObserver_.rotorInertiaResidual = Eigen::VectorXd::Zero(actuatedDofNumber);
  forceFusion_.sensorTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  forceFusion_.filteredSensorTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  forceFusion_.fusedTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  forceFusion_.filteredPublishedTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  forceFusion_.fusedWrench = sva::ForceVecd::Zero();
  forceFusion_.residualWrench = sva::ForceVecd::Zero();
  forceFusion_.sensorWrench = Eigen::Vector6d::Zero();
  residualObserver_.integralFull = Eigen::VectorXd::Zero(dofNumber);
  residualObserver_.residualFull = Eigen::VectorXd::Zero(dofNumber);
  speedObserver_.integral = Eigen::VectorXd::Zero(actuatedDofNumber);
  speedObserver_.residual = Eigen::VectorXd::Zero(actuatedDofNumber);
  EstimationAtFTSensors.assign(static_cast<size_t>(robot.forceSensors().size()), sva::ForceVecd::Zero());
  counter = 0;
}

void ExternalForcesEstimator::resetResidualGain(double gain)
{
  residualObserver_.integralJoint.setZero();
  residualObserver_.jointResidual.setZero();
  forceFusion_.filteredSensorTorques.setZero();
  residualGain = gain;
}

ExternalForcesEstimator::EstimatorInputs
ExternalForcesEstimator::buildEstimatorInputs(mc_control::MCGlobalController & controller,
                                              int preservedPrefix,
                                              bool warnWhenInactive,
                                              bool logPluginState)
{
  auto & robot = controller.controller().robot();
  auto & realRobot = controller.controller().realRobot(controller.controller().robots()[0].name());

  EstimatorInputs inputs;
  inputs.controller = &controller;
  inputs.robot = &robot;
  inputs.realRobot = &realRobot;
  inputs.preservedPrefix = preservedPrefix;
  inputs.warnWhenInactive = warnWhenInactive;
  inputs.logPluginState = logPluginState;
  inputs.mbc = prepareRuntimeInputs(robot, realRobot, preservedPrefix, inputs.qdot, inputs.tau);
  inputs.commandedAcceleration = robotIsFloatingBase ? rbd::dofToVector(robot.mb(), robot.alphaD())
                                                     : Eigen::VectorXd::Zero(dofNumber);
  zeroInactiveEntries(inputs.commandedAcceleration, activeJointIndices, preservedPrefix);

  forwardDynamics.computeC(robot.mb(), inputs.mbc);
  forwardDynamics.computeH(robot.mb(), inputs.mbc);
  inputs.coriolisMatrix = coriolis->coriolis(robot.mb(), inputs.mbc);
  inputs.gravity = forwardDynamics.C() - inputs.coriolisMatrix * inputs.qdot;
  return inputs;
}

void ExternalForcesEstimator::updateDiagnostics(const EstimatorInputs & inputs)
{
  diagnostics_.alphas = inputs.qdot;
  diagnostics_.inputTorque = inputs.tau;
  diagnostics_.commandedAcceleration = inputs.commandedAcceleration;
  diagnostics_.gravity = inputs.gravity;
}

rbd::MultiBodyConfig ExternalForcesEstimator::prepareRuntimeInputs(const mc_rbdyn::Robot & robot,
                                                                   const mc_rbdyn::Robot & realRobot,
                                                                   int preservedPrefix,
                                                                   Eigen::VectorXd & qdot,
                                                                   Eigen::VectorXd & tau)
{
  auto mbc = robot.mbc();
  qdot = rbd::dofToVector(robot.mb(), mbc.alpha);
  zeroInactiveEntries(qdot, activeJointIndices, preservedPrefix);
  mbc.alpha = rbd::vectorToDof(robot.mb(), qdot);
  rbd::forwardVelocity(robot.mb(), mbc);
  tau = readMeasuredTorque(robot, realRobot, preservedPrefix);
  return mbc;
}

Eigen::VectorXd ExternalForcesEstimator::readMeasuredTorque(const mc_rbdyn::Robot & robot,
                                                            const mc_rbdyn::Robot & realRobot,
                                                            int preservedPrefix) const
{
  switch(tau_mes_src_)
  {
    case TorqueSourceType::CommandedTorque:
    {
      if(preservedPrefix == 0)
      {
        mc_rtc::log::error_and_throw<std::runtime_error>("Not implemented yet");
      }
      auto tau = rbd::dofToVector(robot.mb(), robot.jointTorque());
      zeroInactiveEntries(tau, activeJointIndices, preservedPrefix);
      return tau;
    }
    case TorqueSourceType::CurrentMeasurement:
      mc_rtc::log::error_and_throw<std::runtime_error>("Not implemented yet");
    case TorqueSourceType::MotorTorqueMeasurement:
      if(preservedPrefix == 0)
      {
        mc_rtc::log::error_and_throw<std::runtime_error>("Not implemented yet");
      }
      return sanitizeTorqueInput(realRobot, robot,
                                 Eigen::Map<const Eigen::VectorXd>(realRobot.jointTorques().data(),
                                                                   realRobot.jointTorques().size())
                                     * robot.mb().joint(robot.mb().nrJoints() - 1).gearRatio(),
                                 activeJointIndices, dofNumber, preservedPrefix);
    case TorqueSourceType::JointTorqueMeasurement:
      return sanitizeTorqueInput(realRobot, robot,
                                 Eigen::Map<const Eigen::VectorXd>(realRobot.jointTorques().data(),
                                                                   realRobot.jointTorques().size()),
                                 activeJointIndices, dofNumber, preservedPrefix);
  }

  mc_rtc::log::error_and_throw<std::runtime_error>("[ExternalForcesEstimator] Unsupported torque source type");
}

bool ExternalForcesEstimator::updatePluginActivation(mc_control::MCGlobalController & controller) const
{
  auto & extTorquePlugin = controller.controller().datastore().get<std::vector<std::string>>(kPluginRegistryKey);
  if(isActive)
  {
    if(std::find(extTorquePlugin.begin(), extTorquePlugin.end(), kPluginName) == extTorquePlugin.end())
    {
      extTorquePlugin.push_back(kPluginName);
    }
  }
  else
  {
    extTorquePlugin.erase(std::remove(extTorquePlugin.begin(), extTorquePlugin.end(), kPluginName), extTorquePlugin.end());
  }

  bool onePluginIsActive = !extTorquePlugin.empty();
  if(onePluginIsActive)
  {
    for(const auto & pluginName : extTorquePlugin)
    {
      if(pluginName != kPluginName)
      {
        if(verbose)
        {
          mc_rtc::log::info(
              "[ExternalForcesEstimator] Another plugin is active: {}, the last plugin sets the external torques.",
              pluginName);
        }
        break;
      }
    }
  }
  return onePluginIsActive;
}

void ExternalForcesEstimator::updateSpeedResidualDatastore(mc_control::MCGlobalController & controller)
{
  if(!controller.controller().datastore().has(kSpeedResidualKey))
  {
    controller.controller().datastore().make<Eigen::VectorXd>(kSpeedResidualKey, speedObserver_.residual);
  }
  else
  {
    controller.controller().datastore().assign(kSpeedResidualKey, speedObserver_.residual);
  }
}

ExternalForcesEstimator::ForceFusionState
ExternalForcesEstimator::computeFixedBaseForceFusion(const mc_rbdyn::Robot & robot,
                                                     const mc_rbdyn::Robot & realRobot,
                                                     const rbd::MultiBodyConfig & mbc,
                                                     const Eigen::VectorXd & jointResidual)
{
  ForceFusionState fusion;
  fusion.sensorTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  fusion.filteredSensorTorques = forceFusion_.filteredSensorTorques;
  fusion.fusedTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  fusion.filteredPublishedTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  const auto R = robot.bodyPosW(referenceFrame).rotation();

  auto jTranspose = jac.jacobian(robot.mb(), mbc);
  jTranspose.transposeInPlace();
  auto jTransposeActive = selectRows(jTranspose, activeJointIndices);
  Eigen::VectorXd residualWrench = jTransposeActive.completeOrthogonalDecomposition().solve(jointResidual);
  fusion.residualWrench = sva::ForceVecd(residualWrench);
  fusion.residualWrench.force() = R * fusion.residualWrench.force();
  fusion.residualWrench.couple() = R * fusion.residualWrench.couple();

  if(use_force_sensor_ && ft_sensor_name_ != "none")
  {
    auto sva_EF_FT = realRobot.forceSensor(ft_sensor_name_).wrenchWithoutGravity(realRobot);
    fusion.sensorWrench = sva_EF_FT.vector();
    fusion.fusedWrench.force() = R.transpose() * sva_EF_FT.force();
    fusion.fusedWrench.couple() = R.transpose() * sva_EF_FT.couple();
    fusion.sensorTorques =
        selectEntries(jac.jacobian(robot.mb(), mbc).transpose() * fusion.fusedWrench.vector(), activeJointIndices);
    double alpha = 1 - exp(-dt * residualGain);
    fusion.filteredSensorTorques += alpha * (fusion.sensorTorques - fusion.filteredSensorTorques);
    fusion.fusedTorques = jointResidual + (fusion.sensorTorques - fusion.filteredSensorTorques);
    fusion.filteredPublishedTorques = fusion.fusedTorques;
    fusion.filteredSensorWrench =
        sva::ForceVecd(jTransposeActive.completeOrthogonalDecomposition().solve(fusion.filteredSensorTorques));
    fusion.filteredSensorWrench.force() = R * fusion.filteredSensorWrench.force();
    fusion.filteredSensorWrench.couple() = R * fusion.filteredSensorWrench.couple();

    fusion.unfilteredWrench = sva::ForceVecd(jTransposeActive.completeOrthogonalDecomposition().solve(fusion.fusedTorques));
    fusion.unfilteredWrench.force() = R * fusion.unfilteredWrench.force();
    fusion.unfilteredWrench.couple() = R * fusion.unfilteredWrench.couple();
    fusion.publishedTorques = scatterEntries(fusion.filteredPublishedTorques, activeJointIndices, dofNumber);
  }
  else
  {
    fusion.publishedTorques = scatterEntries(jointResidual, activeJointIndices, dofNumber);
  }

  auto activeExternalTorques = selectEntries(fusion.publishedTorques, activeJointIndices);
  fusion.fusedWrench = sva::ForceVecd(jTransposeActive.completeOrthogonalDecomposition().solve(activeExternalTorques));
  fusion.fusedWrench.force() = R * fusion.fusedWrench.force();
  fusion.fusedWrench.couple() = R * fusion.fusedWrench.couple();
  return fusion;
}

ExternalForcesEstimator::ForceFusionState
ExternalForcesEstimator::computeFullGeneralizedForceFusion(const mc_rbdyn::Robot & robot,
                                                           const rbd::MultiBodyConfig & mbc,
                                                           const Eigen::VectorXd & activeResidual)
{
  ForceFusionState fusion;
  fusion.sensorTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  fusion.filteredSensorTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  const auto R = robot.bodyPosW(robot.frame(referenceFrame).body()).rotation();

  auto jTranspose = jac.jacobian(robot.mb(), mbc);
  jTranspose.transposeInPlace();
  auto jTransposeActive = selectRows(jTranspose, activeJointIndices);
  fusion.residualWrench = sva::ForceVecd(jTransposeActive.completeOrthogonalDecomposition().solve(activeResidual));
  fusion.residualWrench.force() = R * fusion.residualWrench.force();
  fusion.residualWrench.couple() = R * fusion.residualWrench.couple();
  fusion.fusedWrench = fusion.residualWrench;
  fusion.filteredPublishedTorques = activeResidual;
  fusion.fusedTorques = activeResidual;
  fusion.unfilteredWrench = fusion.residualWrench;
  return fusion;
}

std::vector<sva::ForceVecd>
ExternalForcesEstimator::estimateFloatingBaseSensorWrenches(const mc_rbdyn::Robot & robot,
                                                            const mc_rbdyn::Robot & realRobot,
                                                            const rbd::MultiBodyConfig & mbc,
                                                            const Eigen::MatrixXd & FT,
                                                            const Eigen::MatrixXd & I_c_0_inv,
                                                            const Eigen::VectorXd & residualFB) const
{
  std::vector<sva::ForceVecd> estimations(static_cast<size_t>(realRobot.forceSensors().size()), sva::ForceVecd::Zero());
  Eigen::MatrixXd augmented_Jfb_forces(6 + actuatedDofNumber, 6);

  size_t fsi = 0;
  for(auto & sensor : realRobot.forceSensors())
  {
    rbd::Jacobian jacobian_forces = rbd::Jacobian(robot.mb(), sensor.parentBody());
    Eigen::MatrixXd Jac_forces = jacobian_forces.jacobian(robot.mb(), mbc, realRobot.posW());
    Eigen::MatrixXd fullJac_forces(6, dofNumber);
    jacobian_forces.fullJacobian(robot.mb(), Jac_forces, fullJac_forces);
    Eigen::MatrixXd Jfb_forces_T = selectCols(fullJac_forces, activeJointIndices).transpose() - FT * I_c_0_inv;
    augmented_Jfb_forces.block(0, 0, 6, 6).setIdentity();
    augmented_Jfb_forces.block(6, 0, actuatedDofNumber, 6) = Jfb_forces_T;
    Eigen::VectorXd estimated_wrench_fb = augmented_Jfb_forces.completeOrthogonalDecomposition().solve(residualFB);
    Eigen::VectorXd estimated_wrench = realRobot.posW().matrix().transpose() * estimated_wrench_fb;
    estimations[fsi] = sva::ForceVecd(estimated_wrench);
    fsi++;
  }

  return estimations;
}

void ExternalForcesEstimator::updateRobotExternalForces(mc_control::MCGlobalController & controller,
                                                        const mc_rbdyn::Robot & robot,
                                                        const mc_rbdyn::Robot & realRobot,
                                                        const Eigen::VectorXd & torques,
                                                        const Eigen::VectorXd & accelerations)
{
  auto realExternalTorques = mapFullDofByJointName(robot, torques, realRobot, realRobot.mb().nrDof());
  auto realExternalAccelerations = mapFullDofByJointName(robot, accelerations, realRobot, realRobot.mb().nrDof());
  controller.controller().robot().setExternalTorques(torques);
  controller.controller().robot().setExternalTorquesAcc(accelerations);
  controller.controller().realRobot().setExternalTorques(realExternalTorques);
  controller.controller().realRobot().setExternalTorquesAcc(realExternalAccelerations);
  counter = 0;
}

void ExternalForcesEstimator::clearRobotExternalForces(mc_control::MCGlobalController & controller,
                                                       const mc_rbdyn::Robot & realRobot) const
{
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(dofNumber);
  controller.controller().robot().setExternalTorques(zero);
  controller.controller().robot().setExternalTorquesAcc(zero);
  controller.controller().realRobot().setExternalTorques(Eigen::VectorXd::Zero(realRobot.mb().nrDof()));
  controller.controller().realRobot().setExternalTorquesAcc(Eigen::VectorXd::Zero(realRobot.mb().nrDof()));
}

void ExternalForcesEstimator::resolveAndUpdateRobot(mc_control::MCGlobalController & controller,
                                                    const mc_rbdyn::Robot & robot,
                                                    const mc_rbdyn::Robot & realRobot,
                                                    Eigen::VectorXd torques,
                                                    Eigen::VectorXd accelerations,
                                                    int preservedPrefix,
                                                    bool warnWhenInactive,
                                                    bool logPluginState)
{
  zeroInactiveEntries(torques, activeJointIndices, preservedPrefix);
  zeroInactiveEntries(accelerations, activeJointIndices, preservedPrefix);

  if(warnWhenInactive)
  {
    counter++;
  }

  const bool onePluginIsActive = updatePluginActivation(controller);
  if(isActive)
  {
    updateRobotExternalForces(controller, robot, realRobot, torques, accelerations);
  }
  else if(!onePluginIsActive)
  {
    clearRobotExternalForces(controller, realRobot);
    if(warnWhenInactive && counter == 1)
    {
      mc_rtc::log::warning("External force feedback inactive");
    }
  }
  else if(logPluginState)
  {
    const auto & extTorquePlugin = controller.controller().datastore().get<std::vector<std::string>>(kPluginRegistryKey);
    mc_rtc::log::info("[mc_residual] isActive = {}, onePluginIsActive = {}, extTorquePlugin = {}", isActive,
                      onePluginIsActive, fmt::join(extTorquePlugin, ","));
  }
}

void ExternalForcesEstimator::init(mc_control::MCGlobalController & controller, const mc_rtc::Configuration & config)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  auto & robot = ctl.controller().robot(ctl.controller().robots()[0].name());
  dt = ctl.timestep();
  dofNumber = robot.mb().nrDof();
  robotIsFloatingBase = (robot.mb().nrJoints() > 0 && robot.mb().joint(0).type() == rbd::Joint::Free);
  mc_rtc::log::info("[ExternalForcesEstimator][Init] dofNumber = {}", dofNumber);

  initializeActiveJoints(robot);

  if(!ctl.controller().datastore().has(kPluginRegistryKey))
  {
    ctl.controller().datastore().make_initializer<std::vector<std::string>>(kPluginRegistryKey);
  }

  Eigen::VectorXd qdot(dofNumber);
  qdot = robot.tvmRobot().alpha()->value();
  zeroInactiveEntries(qdot, activeJointIndices, robot.mb().joint(0).type() == rbd::Joint::Free ? 6 : 0);
  loadConfiguration(config);

  jac = rbd::Jacobian(robot.mb(), referenceFrame);
  coriolis = std::make_unique<rbd::Coriolis>(robot.mb());
  forwardDynamics = rbd::ForwardDynamics(robot.mb());
  if(robotIsFloatingBase)
  {
    if(floating_base_mode_ == FloatingBaseMode::FullGeneralized)
    {
      backend_ = makeFloatingBaseFullGeneralizedBackend();
    }
    else
    {
      backend_ = makeFloatingBaseDecoupledBackend();
    }
  }
  else
  {
    backend_ = makeFixedBaseEstimatorBackend();
  }
  initializeEstimatorState(robot, qdot);

  // Create datastore's entries to change modify parameters from code
  ctl.controller().datastore().make<Eigen::VectorXd>(kResidualOnlyKey, residualObserver_.jointResidual);

  ctl.controller().datastore().make_call("EF_Estimator::isActive", [this]() { return this->isActive; });
  ctl.controller().datastore().make_call("EF_Estimator::toggleActive", [this]() { this->isActive = !this->isActive; });
  ctl.controller().datastore().make_call("EF_Estimator::useForceSensor", [this]() { return this->use_force_sensor_; });
  ctl.controller().datastore().make_call("EF_Estimator::toggleForceSensor",
                                         [this]() { this->use_force_sensor_ = !this->use_force_sensor_; });
  ctl.controller().datastore().make_call("EF_Estimator::setGain", [this](double gain) { this->resetResidualGain(gain); });

  addGui(controller);
  addLog(controller);

  mc_rtc::log::info("[ExternalForcesEstimator][Init] selected backend = {}", backend_->name());
  mc_rtc::log::info("[ExternalForcesEstimator][Init] called with configuration:\n{}", config.dump(true, true));
}

void ExternalForcesEstimator::reset(mc_control::MCGlobalController & controller)
{
  removeLog(controller);
  mc_rtc::log::info("[ExternalForcesEstimator][Reset] called");
}

void ExternalForcesEstimator::before(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  if(ctl.controller().robot().encoderVelocities().empty())
  {
    return;
  }

  auto result = backend_->run(*this, controller);
  auto & robot = ctl.controller().robot();
  auto & realRobot = ctl.controller().realRobot(ctl.controller().robots()[0].name());
  resolveAndUpdateRobot(ctl, robot, realRobot, result.torques, result.accelerations, result.preservedPrefix,
                        result.warnWhenInactive, result.logPluginState);

  // mc_rtc::log::info("[mc_residual] realRobot & = {}, realRobot.externalTorques = {}",
  //                   fmt::ptr(&controller.controller().realRobot()),
  //                   controller.controller().realRobot().externalTorques());
}

void ExternalForcesEstimator::after(mc_control::MCGlobalController & controller)
{
  // mc_rtc::log::info("ExternalForcesEstimator::after");
}

mc_control::GlobalPlugin::GlobalPluginConfiguration ExternalForcesEstimator::configuration()
{
  mc_control::GlobalPlugin::GlobalPluginConfiguration out;
  out.should_run_before = true;
  out.should_run_after = false;
  out.should_always_run = false;
  return out;
}

void ExternalForcesEstimator::computeForwardDynamic(mc_control::MCGlobalController & controller)
{
  auto mb = controller.realRobot(controller.robots()[0].name()).mb();
  auto mbc = controller.realRobot(controller.robots()[0].name()).mbc();
  std::vector<sva::RBInertiad> I_st_(static_cast<size_t>(mb.nrBodies()));
  std::vector<Eigen::Matrix6d> Id_st_(static_cast<size_t>(mb.nrBodies()));
  std::vector<Eigen::Matrix6d> Xd_p_vec(static_cast<size_t>(mb.nrBodies()));
  std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>> F_(static_cast<size_t>(mb.nrJoints()));
  std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>> Fd_(static_cast<size_t>(mb.nrJoints()));
  std::vector<int> dofPos_(static_cast<size_t>(mb.nrJoints()));

  int dofP = 0;
  for(int i = 0; i < mb.nrJoints(); ++i)
  {
    const auto ui = static_cast<size_t>(i);
    F_[ui].resize(6, mb.joint(i).dof());
    Fd_[ui].resize(6, mb.joint(i).dof());
    dofPos_[ui] = dofP;
    dofP += mb.joint(i).dof();
  }

  const std::vector<rbd::Body> & bodies = mb.bodies();
  const std::vector<rbd::Joint> & joints = mb.joints();
  const std::vector<int> & pred = mb.predecessors();

  H.setZero(mb.nrDof(), mb.nrDof());
  Hd.setZero(mb.nrDof(), mb.nrDof());
  for(std::size_t i = 0; i < bodies.size(); ++i)
  {
    const auto ui = static_cast<size_t>(i);
    const sva::PTransformd & X_p_i = mbc.parentToSon[ui];
    Xd_p_vec[ui] = -sva::vector6ToCrossMatrix(mbc.jointVelocity[ui].vector()) * X_p_i.matrix();
    I_st_[i] = bodies[i].inertia();
    Id_st_[i].setZero();
  }

  for(int i = static_cast<int>(bodies.size()) - 1; i >= 0; --i)
  {
    const auto ui = static_cast<size_t>(i);
    if(pred[ui] != -1)
    {
      const sva::PTransformd & X_p_i = mbc.parentToSon[ui];
      Eigen::Matrix6d Xd_p_i = Xd_p_vec[ui];
      I_st_[static_cast<size_t>(pred[ui])] += X_p_i.transMul(I_st_[ui]);
      Id_st_[static_cast<size_t>(pred[ui])] += X_p_i.matrix().transpose() * Id_st_[ui] * X_p_i.matrix()
                                               + Xd_p_i.transpose() * I_st_[ui].matrix() * X_p_i.matrix()
                                               + X_p_i.matrix().transpose() * I_st_[ui].matrix() * Xd_p_i;
    }

    for(int dof = 0; dof < joints[ui].dof(); ++dof)
    {
      F_[ui].col(dof).noalias() = (I_st_[ui] * sva::MotionVecd(mbc.motionSubspace[ui].col(dof))).vector();
      Fd_[ui].col(dof).noalias() = Id_st_[ui] * mbc.motionSubspace[ui].col(dof);
    }

    H.block(dofPos_[ui], dofPos_[ui], joints[ui].dof(), joints[ui].dof()).noalias() =
        mbc.motionSubspace[ui].transpose() * F_[ui];
    Hd.block(dofPos_[ui], dofPos_[ui], joints[ui].dof(), joints[ui].dof()).noalias() =
        mbc.motionSubspace[ui].transpose() * Fd_[ui];

    size_t j = ui;
    while(pred[j] != -1)
    {
      const sva::PTransformd & X_p_j = mbc.parentToSon[j];
      const Eigen::Matrix6d & Xd_p_j = Xd_p_vec[j];
      for(int dof = 0; dof < joints[ui].dof(); ++dof)
      {
        F_[ui].col(dof) = X_p_j.transMul(sva::ForceVecd(F_[ui].col(dof))).vector();
        Fd_[ui].col(dof) =
            X_p_j.transMul(sva::ForceVecd(Fd_[ui].col(dof))).vector() + Xd_p_j.transpose() * F_[ui].col(dof);
      }
      j = static_cast<size_t>(pred[j]);

      if(joints[j].dof() != 0)
      {
        H.block(dofPos_[ui], dofPos_[j], joints[ui].dof(), joints[j].dof()).noalias() =
            F_[ui].transpose() * mbc.motionSubspace[j];
        Hd.block(dofPos_[ui], dofPos_[j], joints[ui].dof(), joints[j].dof()).noalias() =
            Fd_[ui].transpose() * mbc.motionSubspace[j];

        H.block(dofPos_[j], dofPos_[ui], joints[j].dof(), joints[ui].dof()).noalias() =
            H.block(dofPos_[ui], dofPos_[j], joints[ui].dof(), joints[j].dof()).transpose();
        Hd.block(dofPos_[j], dofPos_[ui], joints[j].dof(), joints[ui].dof()).noalias() =
            Hd.block(dofPos_[ui], dofPos_[j], joints[ui].dof(), joints[j].dof()).transpose();
      }
    }
  }

  H.noalias() = H;
}

void ExternalForcesEstimator::computeCHatPc0Hat(mc_control::MCGlobalController & controller,
                                                const rbd::MultiBodyConfig & mbc)
{
  auto mb = controller.realRobot(controller.robots()[0].name()).mb();
  c_hat = Eigen::VectorXd::Zero(mb.nrDof());
  std::vector<sva::MotionVecd> acc_(static_cast<size_t>(mb.nrBodies()));
  std::vector<sva::ForceVecd> f_(static_cast<size_t>(mb.nrBodies()));
  std::vector<int> dofPos_(static_cast<size_t>(mb.nrJoints()));

  int dofP = 0;
  for(int i = 0; i < mb.nrJoints(); ++i)
  {
    const auto ui = static_cast<size_t>(i);
    dofPos_[ui] = dofP;
    dofP += mb.joint(i).dof();
  }

  const std::vector<rbd::Body> & bodies = mb.bodies();
  const std::vector<rbd::Joint> & joints = mb.joints();
  const std::vector<int> & pred = mb.predecessors();

  sva::MotionVecd a_0(Eigen::Vector3d::Zero(), mbc.gravity);

  for(std::size_t i = 0; i < bodies.size(); ++i)
  {
    const sva::PTransformd & X_p_i = mbc.parentToSon[i];

    const sva::MotionVecd & vj_i = mbc.jointVelocity[i];

    const sva::MotionVecd & vb_i = mbc.bodyVelB[i];

    if(pred[i] != -1)
      acc_[i] = X_p_i * acc_[static_cast<size_t>(pred[i])] + vb_i.cross(vj_i);
    else
      acc_[i] = X_p_i * a_0 + vb_i.cross(vj_i);

    f_[i] = bodies[i].inertia() * acc_[i] + vb_i.crossDual(bodies[i].inertia() * vb_i);
  }

  for(int i = static_cast<int>(bodies.size()) - 1; i >= 0; --i)
  {
    const auto ui = static_cast<size_t>(i);
    c_hat.segment(dofPos_[ui], joints[ui].dof()).noalias() = mbc.motionSubspace[ui].transpose() * f_[ui].vector();

    if(pred[ui] != -1)
    {
      const sva::PTransformd & X_p_i = mbc.parentToSon[ui];
      f_[static_cast<size_t>(pred[ui])] += X_p_i.transMul(f_[ui]);
    }
  }

  // mc_rtc::log::info("C hat = {}", c_hat.transpose());
}

void ExternalForcesEstimator::addGui(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  ctl.controller().gui()->addElement({"Plugins", "External forces estimator"},
                                     mc_rtc::gui::Checkbox("Is estimation feedback active", isActive),
                                     mc_rtc::gui::Checkbox("Use force sensor", use_force_sensor_),
                                     mc_rtc::gui::NumberInput("Gain", [this]() { return this->residualGain; },
                                                              [this](double gain) { resetResidualGain(gain); }),
                                     mc_rtc::gui::NumberInput(
                                         "Residual speed gain", [this]() { return this->residualSpeedGain; },
                                         [this](double gainSpeed)
                                         {
                                           if(gainSpeed != residualSpeedGain)
                                           {
                                             speedObserver_.integral.setZero();
                                             speedObserver_.residual.setZero();
                                           }
                                           residualSpeedGain = gainSpeed;
                                          }),
                                     mc_rtc::gui::Label("nrDof", [this]() { return this->dofNumber; }));

  if(backend_)
  {
    backend_->addToGui(*this, controller);
  }
}

void ExternalForcesEstimator::addLog(mc_control::MCGlobalController & controller)
{
  auto & logger = controller.controller().logger();
  logger.addLogEntry("ExternalForceEstimator_alpha", this, [this]() { return diagnostics_.alphas; });
  logger.addLogEntry("ExternalForceEstimator_inputTorque", this, [this]() { return diagnostics_.inputTorque; });
  logger.addLogEntry("gravity", this, [this]() { return diagnostics_.gravity; });
  logger.addLogEntry("commanded_acceleration", this, [this]() { return diagnostics_.commandedAcceleration; });
  logger.addLogEntry("ExternalForceEstimator_gain", this, [this]() { return this->residualGain; });
  logger.addLogEntry("ExternalForceEstimator_isActive", this, [this]() { return this->isActive; });

  if(backend_)
  {
    backend_->addToLogger(*this, controller);
  }
}

void ExternalForcesEstimator::removeLog(mc_control::MCGlobalController & controller)
{
  controller.controller().logger().removeLogEntries(this);
}

} // namespace mc_plugin

EXPORT_MC_RTC_PLUGIN("ExternalForcesEstimator", mc_plugin::ExternalForcesEstimator)
