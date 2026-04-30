#include "ExternalForcesEstimator.h"
#include <mc_control/GlobalPluginMacros.h>
#include <mc_control/mc_global_controller.h>
#include <mc_rtc/logging.h>
#include <mc_rtc/unique_ptr.h>
#include <SpaceVecAlg/EigenTypedef.h>
#include <SpaceVecAlg/EigenUtility.h>
#include <SpaceVecAlg/SpaceVecAlg>
#include "EstimatorMathUtils.h"
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
  ForwardDynamicsMode forwardDynamicsMode = ForwardDynamicsMode::Classical;
  BiasTermMode biasTermMode = BiasTermMode::Classical;
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
  if(mode == "Default")
  {
    return FloatingBaseMode::FullGeneralized;
  }
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[ExternalForceEstimator] error in configuration with entry\"floating_base_mode\".\n\tPossible values are: "
      "Decoupled, Default");
}

ForwardDynamicsMode parseForwardDynamicsMode(const std::string & mode)
{
  if(mode.empty() || mode == "Default")
  {
    return ForwardDynamicsMode::Classical;
  }
  if(mode == "Flacco")
  {
    return ForwardDynamicsMode::Flacco;
  }
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[ExternalForceEstimator] error in configuration with entry\"forward_dynamics_mode\".\n\tPossible values "
      "are: Default, Flacco");
}

BiasTermMode parseBiasTermMode(const std::string & mode)
{
  if(mode.empty() || mode == "Default")
  {
    return BiasTermMode::Classical;
  }
  if(mode == "Flacco")
  {
    return BiasTermMode::Flacco;
  }
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[ExternalForceEstimator] error in configuration with entry\"bias_term_mode\".\n\tPossible values are: "
      "Default, Flacco");
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
  parsed.forwardDynamicsMode = parseForwardDynamicsMode(config("forward_dynamics_mode", std::string{"Default"}));
  parsed.biasTermMode = parseBiasTermMode(config("bias_term_mode", std::string{"Default"}));
  parsed.residualSpeedGain = config("residual_speed_gain", 100.0);

  residualGain = parsed.residualGain;
  referenceFrame = std::move(parsed.referenceFrame);
  verbose = parsed.verbose;
  ft_sensor_name_ = std::move(parsed.ftSensorName);
  use_force_sensor_ = parsed.useForceSensor;
  tau_mes_src_ = parsed.torqueSource;
  floating_base_mode_ = parsed.floatingBaseMode;
  forward_dynamics_mode_ = parsed.forwardDynamicsMode;
  bias_term_mode_ = parsed.biasTermMode;
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

  residuals_.integralBase = Eigen::VectorXd::Zero(6);
  residuals_.integralJoint = Eigen::VectorXd::Zero(actuatedDofNumber);
  residuals_.jointResidual = Eigen::VectorXd::Zero(actuatedDofNumber);
  residuals_.rotorInertiaIntegral = Eigen::VectorXd::Zero(actuatedDofNumber);
  residuals_.baseResidual = Eigen::VectorXd::Zero(6);
  residuals_.rotorInertiaResidual = Eigen::VectorXd::Zero(actuatedDofNumber);
  residuals_.integralFull = Eigen::VectorXd::Zero(dofNumber);
  residuals_.residualFull = Eigen::VectorXd::Zero(dofNumber);
  forceEffects_.sensorTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  forceEffects_.filteredSensorTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  forceEffects_.fusedTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  forceEffects_.filteredPublishedTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  forceEffects_.fusedWrench = sva::ForceVecd::Zero();
  forceEffects_.residualWrench = sva::ForceVecd::Zero();
  forceEffects_.sensorWrench = Eigen::Vector6d::Zero();
  forceEffects_.sensorForceEstimations.assign(static_cast<size_t>(robot.forceSensors().size()), sva::ForceVecd::Zero());
  speedResidual_.integral = Eigen::VectorXd::Zero(actuatedDofNumber);
  speedResidual_.residual = Eigen::VectorXd::Zero(actuatedDofNumber);
  counter = 0;
}

void ExternalForcesEstimator::resetResidualGain(double gain)
{
  residuals_.integralJoint.setZero();
  residuals_.jointResidual.setZero();
  forceEffects_.filteredSensorTorques.setZero();
  residualGain = gain;
}

ExternalForcesEstimator::EstimatorInputs ExternalForcesEstimator::buildEstimatorInputs(
    mc_control::MCGlobalController & controller,
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
  inputs.commandedAcceleration =
      robotIsFloatingBase ? rbd::dofToVector(robot.mb(), robot.alphaD()) : Eigen::VectorXd::Zero(dofNumber);
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
      return sanitizeTorqueInput(
          realRobot, robot,
          Eigen::Map<const Eigen::VectorXd>(realRobot.jointTorques().data(), realRobot.jointTorques().size())
              * robot.mb().joint(robot.mb().nrJoints() - 1).gearRatio(),
          activeJointIndices, dofNumber, preservedPrefix);
    case TorqueSourceType::JointTorqueMeasurement:
      return sanitizeTorqueInput(
          realRobot, robot,
          Eigen::Map<const Eigen::VectorXd>(realRobot.jointTorques().data(), realRobot.jointTorques().size()),
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
    extTorquePlugin.erase(std::remove(extTorquePlugin.begin(), extTorquePlugin.end(), kPluginName),
                          extTorquePlugin.end());
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
    controller.controller().datastore().make<Eigen::VectorXd>(kSpeedResidualKey, speedResidual_.residual);
  }
  else
  {
    controller.controller().datastore().assign(kSpeedResidualKey, speedResidual_.residual);
  }
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
    const auto & extTorquePlugin =
        controller.controller().datastore().get<std::vector<std::string>>(kPluginRegistryKey);
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
  ctl.controller().datastore().make<Eigen::VectorXd>(kResidualOnlyKey, residuals_.jointResidual);

  ctl.controller().datastore().make_call("EF_Estimator::isActive", [this]() { return this->isActive; });
  ctl.controller().datastore().make_call("EF_Estimator::toggleActive", [this]() { this->isActive = !this->isActive; });
  ctl.controller().datastore().make_call("EF_Estimator::useForceSensor", [this]() { return this->use_force_sensor_; });
  ctl.controller().datastore().make_call("EF_Estimator::toggleForceSensor",
                                         [this]() { this->use_force_sensor_ = !this->use_force_sensor_; });
  ctl.controller().datastore().make_call("EF_Estimator::setGain",
                                         [this](double gain) { this->resetResidualGain(gain); });

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

  auto state = estimatorData();
  auto result = backend_->run(state, controller);
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

void ExternalForcesEstimator::addGui(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  ctl.controller().gui()->addElement(
      {"Plugins", "External forces estimator"}, mc_rtc::gui::Checkbox("Is estimation feedback active", isActive),
      mc_rtc::gui::Checkbox("Use force sensor", use_force_sensor_),
      mc_rtc::gui::NumberInput(
          "Gain", [this]() { return this->residualGain; }, [this](double gain) { resetResidualGain(gain); }),
      mc_rtc::gui::NumberInput(
          "Residual speed gain", [this]() { return this->residualSpeedGain; },
          [this](double gainSpeed)
          {
            if(gainSpeed != residualSpeedGain)
            {
              speedResidual_.integral.setZero();
              speedResidual_.residual.setZero();
            }
            residualSpeedGain = gainSpeed;
          }),
      mc_rtc::gui::Label("nrDof", [this]() { return this->dofNumber; }));

  if(backend_)
  {
    auto state = estimatorData();
    backend_->addToGui(state, controller);
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
    auto state = estimatorData();
    backend_->addToLogger(state, controller);
  }
}

void ExternalForcesEstimator::removeLog(mc_control::MCGlobalController & controller)
{
  controller.controller().logger().removeLogEntries(this);
}

} // namespace mc_plugin

EXPORT_MC_RTC_PLUGIN("ExternalForcesEstimator", mc_plugin::ExternalForcesEstimator)
