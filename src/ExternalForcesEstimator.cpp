#include "ExternalForcesEstimator.h"
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

namespace
{

Eigen::VectorXd selectEntries(const Eigen::VectorXd & vector, const std::vector<int> & indices)
{
  Eigen::VectorXd out(indices.size());
  for(size_t i = 0; i < indices.size(); ++i)
  {
    out(static_cast<Eigen::Index>(i)) = vector(indices[i]);
  }
  return out;
}

Eigen::VectorXd scatterEntries(const Eigen::VectorXd & vector, const std::vector<int> & indices, int fullSize)
{
  Eigen::VectorXd out = Eigen::VectorXd::Zero(fullSize);
  for(size_t i = 0; i < indices.size(); ++i)
  {
    out(indices[i]) = vector(static_cast<Eigen::Index>(i));
  }
  return out;
}

void zeroInactiveEntries(Eigen::VectorXd & vector, const std::vector<int> & activeIndices, int preservedPrefix)
{
  std::vector<bool> active(static_cast<size_t>(vector.size()), false);
  for(int i = 0; i < preservedPrefix && i < vector.size(); ++i)
  {
    active[static_cast<size_t>(i)] = true;
  }
  for(int idx : activeIndices)
  {
    if(0 <= idx && idx < vector.size())
    {
      active[static_cast<size_t>(idx)] = true;
    }
  }
  for(int i = preservedPrefix; i < vector.size(); ++i)
  {
    if(!active[static_cast<size_t>(i)])
    {
      vector(i) = 0.0;
    }
  }
}

Eigen::MatrixXd selectRows(const Eigen::MatrixXd & matrix, const std::vector<int> & indices)
{
  Eigen::MatrixXd out(indices.size(), matrix.cols());
  for(size_t i = 0; i < indices.size(); ++i)
  {
    out.row(static_cast<Eigen::Index>(i)) = matrix.row(indices[i]);
  }
  return out;
}

Eigen::MatrixXd selectCols(const Eigen::MatrixXd & matrix, const std::vector<int> & indices)
{
  Eigen::MatrixXd out(matrix.rows(), indices.size());
  for(size_t i = 0; i < indices.size(); ++i)
  {
    out.col(static_cast<Eigen::Index>(i)) = matrix.col(indices[i]);
  }
  return out;
}

Eigen::MatrixXd selectSubmatrix(const Eigen::MatrixXd & matrix, const std::vector<int> & indices)
{
  Eigen::MatrixXd out(indices.size(), indices.size());
  for(size_t row = 0; row < indices.size(); ++row)
  {
    for(size_t col = 0; col < indices.size(); ++col)
    {
      out(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) = matrix(indices[row], indices[col]);
    }
  }
  return out;
}

Eigen::VectorXd sanitizeTorqueInput(const Eigen::VectorXd & raw,
                                    const std::vector<int> & activeIndices,
                                    int fullSize,
                                    int preservedPrefix)
{
  if(raw.size() == fullSize)
  {
    Eigen::VectorXd out = raw;
    zeroInactiveEntries(out, activeIndices, preservedPrefix);
    return out;
  }
  if(raw.size() == static_cast<Eigen::Index>(activeIndices.size()))
  {
    return scatterEntries(raw, activeIndices, fullSize);
  }
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[ExternalForcesEstimator] Unexpected torque vector size {}, expected {} or {}", raw.size(), fullSize,
      activeIndices.size());
}

} // namespace

ExternalForcesEstimator::~ExternalForcesEstimator() = default;

void ExternalForcesEstimator::init(mc_control::MCGlobalController & controller, const mc_rtc::Configuration & config)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  auto & robot = ctl.robot(ctl.robots()[0].name());
  auto & tvmRobot = robot.tvmRobot();
  auto & realRobot = ctl.realRobot(ctl.robots()[0].name());
  dt = ctl.timestep();

  dofNumber = realRobot.mb().nrDof();
  std::vector<std::string> activeJointNames;
  mc_rtc::log::info("[ExternalForcesEstimator][Init] dofNumber = {}", dofNumber);

  std::vector<std::string> active_gripper_joints;
  for(const auto & g : robot.grippers())
  {
    for(const auto & n : g.get().activeJoints())
    {
      active_gripper_joints.push_back(n);
    }
  }
  auto isActiveGripperJoint = [&](const std::string & j)
  { return std::find(active_gripper_joints.begin(), active_gripper_joints.end(), j) != active_gripper_joints.end(); };
  for(const auto & j : robot.mb().joints())
  {
    if(j.dof() != 1 || j.isMimic() || isActiveGripperJoint(j.name()))
    {
      continue;
    }
    mc_rtc::log::info("[ExternalForcesEstimator][Init] Estimated joint -> {}", j.name());
    activeJointNames.push_back(j.name());
  }

  int pos = 0;
  if(robot.mb().nrJoints() > 0 && robot.mb().joint(0).type() == rbd::Joint::Free)
  {
    pos = 6; // Skip the floating base joints
  }
  for(int jI = robot.mb().joint(0).type() == rbd::Joint::Free ? 1 : 0; jI < robot.mb().nrJoints(); ++jI)
  {
    const auto & j = robot.mb().joint(jI);
    if(j.dof() == 1) // prismatic or revolute
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

  if(!ctl.controller().datastore().has("extTorquePlugin"))
  {
    ctl.controller().datastore().make_initializer<std::vector<std::string>>("extTorquePlugin");
  }

  Eigen::VectorXd qdot(dofNumber);
  qdot = tvmRobot.alpha()->value();
  zeroInactiveEntries(qdot, activeJointIndices, robot.mb().joint(0).type() == rbd::Joint::Free ? 6 : 0);

  // load config
  residualGains = config("residual_gain", 0.0);
  referenceFrame = config("reference_frame", (std::string) "");
  verbose = config("verbose", false);
  ft_sensor_name_ = config("ft_sensor_name", (std::string) "");
  use_force_sensor_ = config("use_force_sensor", false);

  std::string source_type = config("torque_source_type", (std::string) "");
  if(source_type.compare("CommandedTorque") == 0)
  {
    tau_mes_src_ = TorqueSourceType::CommandedTorque;
    mc_rtc::log::info("Using CommandedTorque input");
  }
  else if(source_type.compare("CurrentMeasurement") == 0)
  {
    tau_mes_src_ = TorqueSourceType::CurrentMeasurement;
    mc_rtc::log::info("Using CurrentMeasurement input");
  }
  else if(source_type.compare("MotorTorqueMeasurement") == 0)
  {
    tau_mes_src_ = TorqueSourceType::MotorTorqueMeasurement;
    mc_rtc::log::info("Using MotorTorqueMeasurement input");
  }
  else if(source_type.compare("JointTorqueMeasurement") == 0)
  {
    tau_mes_src_ = TorqueSourceType::JointTorqueMeasurement;
    mc_rtc::log::info("Using JointTorqueMeasurement input");
  }
  else
  {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[ExternalForceEstimator] error in configuration with entry\"torque_source_type\".\n\tPossible values are: "
        "CommandedTorque, CurrentMeasurement, MotorTorqueMeasurement, JointTorqueMeasurement");
  }
  residualSpeedGain = config("residual_speed_gain", 100.0);
  // config loaded

  robotIsFloatingBase = (robot.mb().nrJoints() > 0 && robot.mb().joint(0).type() == rbd::Joint::Free);

  jac = rbd::Jacobian(robot.mb(), referenceFrame);
  coriolis = new rbd::Coriolis(robot.mb());
  forwardDynamics = rbd::ForwardDynamics(robot.mb());
  auto mbc = robot.mbc();
  mbc.alpha = rbd::vectorToDof(robot.mb(), qdot);
  rbd::forwardVelocity(robot.mb(), mbc);
  forwardDynamics.computeC(robot.mb(), mbc);
  forwardDynamics.computeH(robot.mb(), mbc);
  auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  prevH = inertiaMatrix;
  format = Eigen::IOFormat(2, 0, " ", "\n", " ", " ", "[", "]");
  pzero = selectEntries(inertiaMatrix * qdot, activeJointIndices);

  integralTermExtern = Eigen::VectorXd::Zero(6);
  integralTermIntern = Eigen::VectorXd::Zero(actuatedDofNumber);
  internResidual = Eigen::VectorXd::Zero(actuatedDofNumber);
  integralTermWithRotorInertia = Eigen::VectorXd::Zero(actuatedDofNumber);
  externResidual = Eigen::VectorXd::Zero(6);
  residualWithRotorInertia = Eigen::VectorXd::Zero(actuatedDofNumber);
  FTSensorTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  filteredFTSensorTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  newExternalTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  filteredExternalTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  externalForces = sva::ForceVecd::Zero();
  externalForcesResidual = sva::ForceVecd::Zero();
  externalForcesFT = Eigen::Vector6d::Zero();
  integralTermNormal = Eigen::VectorXd::Zero(dofNumber);
  residualNormal = Eigen::VectorXd::Zero(dofNumber);

  integralTermSpeed = Eigen::VectorXd::Zero(actuatedDofNumber);
  residualSpeed = Eigen::VectorXd::Zero(actuatedDofNumber);

  for(int i = 0; i < robot.forceSensors().size(); i++)
  {
    EstimationAtFTSensors.push_back(sva::ForceVecd::Zero());
  }

  counter = 0;

  mimicExclusion.setIdentity(dofNumber, dofNumber);

  // Create datastore's entries to change modify parameters from code
  ctl.controller().datastore().make<Eigen::VectorXd>("EF_Estimator::getResidualOnly", internResidual);

  ctl.controller().datastore().make_call("EF_Estimator::isActive", [this]() { return this->isActive; });
  ctl.controller().datastore().make_call("EF_Estimator::toggleActive", [this]() { this->isActive = !this->isActive; });
  ctl.controller().datastore().make_call("EF_Estimator::useForceSensor", [this]() { return this->use_force_sensor_; });
  ctl.controller().datastore().make_call("EF_Estimator::toggleForceSensor",
                                         [this]() { this->use_force_sensor_ = !this->use_force_sensor_; });
  ctl.controller().datastore().make_call("EF_Estimator::setGain",
                                         [this](double gain)
                                         {
                                           this->integralTermIntern.setZero();
                                           this->internResidual.setZero();
                                           this->filteredFTSensorTorques.setZero();
                                           this->residualGains = gain;
                                         });

  addGui(controller);
  addLog(controller);

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

  if(robotIsFloatingBase)
  {
    // mc_rtc::log::info("ExternalForcesEstimator::before: Floating base detected, using floating base dynamics");
    computeForFloatingBase(controller);
  }
  else
  {
    // mc_rtc::log::info("ExternalForcesEstimator::before: Fixed base detected, using fixed base dynamics");
    computeForFixedBase(controller);
  }

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

void ExternalForcesEstimator::computeForFixedBase(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  auto & robot = ctl.robot();
  auto & realRobot = ctl.realRobot(ctl.robots()[0].name());

  Eigen::VectorXd qdot(dofNumber), tau(dofNumber);
  auto mbc = realRobot.mbc();
  qdot = rbd::dofToVector(realRobot.mb(), mbc.alpha);
  zeroInactiveEntries(qdot, activeJointIndices, 0);
  mbc.alpha = rbd::vectorToDof(realRobot.mb(), qdot);
  rbd::forwardVelocity(realRobot.mb(), mbc);
  switch(tau_mes_src_)
  {
    case TorqueSourceType::CommandedTorque:
      mc_rtc::log::error_and_throw<std::runtime_error>("Not implemented yet"); // Need friction model to finalize
      // rbd::paramToVector(robot.jointTorque(), tau);
      break;
    case TorqueSourceType::CurrentMeasurement:
      mc_rtc::log::error_and_throw<std::runtime_error>("Not implemented yet");
      break;
    case TorqueSourceType::MotorTorqueMeasurement:
      mc_rtc::log::error_and_throw<std::runtime_error>("Not implemented yet"); // Need friction model to finalize
      // tau = Eigen::VectorXd::Map(realRobot.jointTorques().data(), realRobot.jointTorques().size())
      //       * robot.mb().joint(robot.mb().nrJoints() - 1).gearRatio();
      break;
    case TorqueSourceType::JointTorqueMeasurement:
      tau = sanitizeTorqueInput(Eigen::Map<const Eigen::VectorXd>(realRobot.jointTorques().data(),
                                                                  realRobot.jointTorques().size()),
                                activeJointIndices, dofNumber, 0);
      break;
  }

  auto R = controller.robot().bodyPosW(referenceFrame).rotation();

  forwardDynamics.computeC(realRobot.mb(), mbc);
  forwardDynamics.computeH(realRobot.mb(), mbc);
  auto coriolisMatrix = coriolis->coriolis(realRobot.mb(), mbc);
  auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  auto inertiaMatrixActive = selectSubmatrix(inertiaMatrix, activeJointIndices);
  auto qdotActive = selectEntries(qdot, activeJointIndices);
  auto tauActive = selectEntries(tau, activeJointIndices);
  auto coriolisGravityTerm = selectEntries(forwardDynamics.C(), activeJointIndices);
  auto coriolisMatrixActive = selectSubmatrix(coriolisMatrix + coriolisMatrix.transpose(), activeJointIndices);

  integralTermIntern +=
      (tauActive + coriolisMatrixActive * qdotActive - coriolisGravityTerm + internResidual) * ctl.timestep();
  auto pt = inertiaMatrixActive * qdotActive;

  internResidual = residualGains * (pt - integralTermIntern + pzero);
  ctl.controller().datastore().assign<Eigen::VectorXd>("EF_Estimator::getResidualOnly", internResidual);

  auto inertiaMatrixWithRotorInertia = selectSubmatrix(forwardDynamics.H(), activeJointIndices);
  auto ptWithRotorInertia = inertiaMatrixWithRotorInertia * qdotActive;
  integralTermWithRotorInertia +=
      (tauActive + coriolisMatrixActive * qdotActive - coriolisGravityTerm + residualWithRotorInertia) * ctl.timestep();
  residualWithRotorInertia = residualGains * (ptWithRotorInertia - integralTermWithRotorInertia + pzero);

  // Residual speed observer
  integralTermSpeed +=
      (tauActive + coriolisMatrixActive * qdotActive - coriolisGravityTerm + residualSpeed) * ctl.timestep();
  residualSpeed = residualSpeedGain * (pt - integralTermSpeed + pzero);
  if(!ctl.controller().datastore().has("speed_residual"))
  {
    ctl.controller().datastore().make<Eigen::VectorXd>("speed_residual", residualSpeed);
  }
  else
  {
    ctl.controller().datastore().assign("speed_residual", residualSpeed);
  }

  auto jTranspose = jac.jacobian(realRobot.mb(), mbc);
  jTranspose.transposeInPlace();
  auto jTransposeActive = selectRows(jTranspose, activeJointIndices);
  Eigen::VectorXd FR = jTransposeActive.completeOrthogonalDecomposition().solve(internResidual);
  externalForcesResidual = sva::ForceVecd(FR);
  externalForcesResidual.force() = R * externalForcesResidual.force();
  externalForcesResidual.couple() = R * externalForcesResidual.couple();
  // mc_rtc::log::info("===== {}", jTranspose.completeOrthogonalDecomposition().pseudoInverse()*jTranspose);

  if(use_force_sensor_ && ft_sensor_name_ != "none")
  {
    auto sva_EF_FT = realRobot.forceSensor(ft_sensor_name_).wrenchWithoutGravity(realRobot);
    externalForcesFT = sva_EF_FT.vector();
    // Applying some rotation so it match the same world as the residual
    externalForces.force() = R.transpose() * sva_EF_FT.force();
    externalForces.couple() = R.transpose() * sva_EF_FT.couple();
    FTSensorTorques =
        selectEntries(jac.jacobian(realRobot.mb(), mbc).transpose() * externalForces.vector(), activeJointIndices);
    double alpha = 1 - exp(-dt * residualGains);
    filteredFTSensorTorques += alpha * (FTSensorTorques - filteredFTSensorTorques);
    newExternalTorques = internResidual + (FTSensorTorques - filteredFTSensorTorques);
    filteredExternalTorques = newExternalTorques;
    filteredFTSensorForces =
        sva::ForceVecd(jTransposeActive.completeOrthogonalDecomposition().solve(filteredFTSensorTorques));
    filteredFTSensorForces.force() = R * filteredFTSensorForces.force();
    filteredFTSensorForces.couple() = R * filteredFTSensorForces.couple();

    newExternalForces = sva::ForceVecd(jTransposeActive.completeOrthogonalDecomposition().solve(newExternalTorques));
    newExternalForces.force() = R * newExternalForces.force();
    newExternalForces.couple() = R * newExternalForces.couple();
    externalTorques = scatterEntries(filteredExternalTorques, activeJointIndices, dofNumber);
  }
  else
  {
    // If the force sensor is not used, we use the residual as external forces
    externalTorques = scatterEntries(internResidual, activeJointIndices, dofNumber);
  }

  auto activeExternalTorques = selectEntries(externalTorques, activeJointIndices);
  externalForces = sva::ForceVecd(jTransposeActive.completeOrthogonalDecomposition().solve(activeExternalTorques));
  externalForces.force() = R * externalForces.force();
  externalForces.couple() = R * externalForces.couple();

  Eigen::VectorXd externalAccelerations = Eigen::VectorXd::Zero(dofNumber);
  externalAccelerations = forwardDynamics.H().ldlt().solve(externalTorques);

  counter++;

  std::vector<std::string> & extTorquePlugin =
      ctl.controller().datastore().get<std::vector<std::string>>("extTorquePlugin");

  if(isActive)
  {
    extTorquePlugin.push_back("ResidualEstimator");
  }
  else
  {
    extTorquePlugin.erase(std::remove(extTorquePlugin.begin(), extTorquePlugin.end(), "ResidualEstimator"),
                          extTorquePlugin.end());
  }

  // bool anotherPluginIsActive = false;
  bool onePluginIsActive = false;
  if(extTorquePlugin.size() > 0)
  {
    onePluginIsActive = true;
    for(const auto & pluginName : extTorquePlugin)
    {
      if(pluginName != "ResidualEstimator")
      {
        if(verbose)
          mc_rtc::log::info(
              "[ExternalForcesEstimator] Another plugin is active: {}, the last plugin sets the external torques.",
              pluginName);
        break;
      }
    }
  }

  if(isActive)
  {
    for(int i = 0; i < externalTorques.size(); i++)
    {
      int idx = i;
      // If the joint is not estimated, set the external torque to zero
      if(std::find(activeJointIndices.begin(), activeJointIndices.end(), idx) == activeJointIndices.end())
      {
        externalTorques[idx] = 0.0;
      }
    }

    ctl.controller().realRobot().setExternalTorques(externalTorques);
    ctl.controller().realRobot().setExternalTorquesAcc(externalAccelerations);
    counter = 0;
  }
  else if(!onePluginIsActive)
  {
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(dofNumber);
    ctl.controller().realRobot().setExternalTorques(zero);
    ctl.controller().realRobot().setExternalTorquesAcc(zero);
    if(counter == 1) mc_rtc::log::warning("External force feedback inactive");
  }
  else
  {
    mc_rtc::log::info("[mc_residual] isActive = {}, onePluginIsActive = {}, extTorquePlugin = {}", isActive,
                      onePluginIsActive, extTorquePlugin);
  }
}

void ExternalForcesEstimator::computeForFloatingBase(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  auto & robot = ctl.robot();
  auto & realRobot = ctl.realRobot(ctl.robots()[0].name());

  Eigen::VectorXd qdot(dofNumber), tau(dofNumber), tau_joint(actuatedDofNumber);
  auto mbc = realRobot.mbc();
  qdot.setZero();
  qdot = rbd::dofToVector(realRobot.mb(), mbc.alpha);
  zeroInactiveEntries(qdot, activeJointIndices, 6);
  mbc.alpha = rbd::vectorToDof(realRobot.mb(), qdot);
  rbd::forwardVelocity(realRobot.mb(), mbc);
  // mc_rtc::log::info("alpha = {}", qdot.transpose());
  // qdot.setZero();
  // qdot.tail(dofNumber - 6) =
  //     Eigen::Map<const Eigen::VectorXd>(realRobot.encoderVelocities().data(), realRobot.encoderVelocities().size());
  // mc_rtc::log::info("encoderVelocities = {}", qdot.transpose());
  alphas = qdot;
  tau.setZero();
  tau_joint.setZero();

  switch(tau_mes_src_)
  {
    case TorqueSourceType::CommandedTorque:
      tau = rbd::dofToVector(realRobot.mb(), robot.jointTorque());
      zeroInactiveEntries(tau, activeJointIndices, 6);
      tau_joint = selectEntries(tau, activeJointIndices);
      break;
    case TorqueSourceType::CurrentMeasurement:
      mc_rtc::log::error_and_throw<std::runtime_error>("Not implemented yet");
      break;
    case TorqueSourceType::MotorTorqueMeasurement:
      tau_joint = selectEntries(sanitizeTorqueInput(Eigen::Map<const Eigen::VectorXd>(realRobot.jointTorques().data(),
                                                                                       realRobot.jointTorques().size())
                                                        * robot.mb().joint(robot.mb().nrJoints() - 1).gearRatio(),
                                                    activeJointIndices, dofNumber, 6),
                                activeJointIndices);
      break;
    case TorqueSourceType::JointTorqueMeasurement:
      tau_joint = selectEntries(sanitizeTorqueInput(Eigen::Map<const Eigen::VectorXd>(realRobot.jointTorques().data(),
                                                                                       realRobot.jointTorques().size()),
                                                    activeJointIndices, dofNumber, 6),
                                activeJointIndices);
      break;
  }
  tau = scatterEntries(tau_joint, activeJointIndices, dofNumber);
  inputTorque = tau;
  commandedAcceleration = rbd::dofToVector(robot.mb(), robot.alphaD());
  zeroInactiveEntries(commandedAcceleration, activeJointIndices, 6);

  // std::cout << "==============================" << std::endl;
  Eigen::VectorXd qdot_fb = qdot.head(6);
  Eigen::VectorXd qdot_joint = selectEntries(qdot, activeJointIndices);
  // mc_rtc::log::info("Size tau = {}", tau.size());
  // std::cout << "qdot_fb = \n" << qdot_fb.transpose() << std::endl;
  // std::cout << "qdot_joint = \n" << qdot_joint.transpose() << std::endl;
  // std::cout << "tau = \n" << tau.transpose() << std::endl;
  // std::cout << "tau_joint = \n" << tau_joint.transpose() << std::endl;

  auto R = controller.robot().bodyPosW(robot.frame(referenceFrame).body()).rotation();

  forwardDynamics.computeC(realRobot.mb(), mbc);
  forwardDynamics.computeH(realRobot.mb(), mbc);
  auto coriolisMatrix = coriolis->coriolis(realRobot.mb(), mbc);
  Eigen::VectorXd coriolisGravityTerm = Eigen::VectorXd::Zero(dofNumber);
  computeCHatPc0Hat(controller, mbc);
  coriolisGravityTerm = forwardDynamics.C();
  gravity = coriolisGravityTerm - coriolisMatrix * qdot;

  format = Eigen::IOFormat(2, 0, " ", "\n", " ", " ", "[", "]");

  // computeForwardDynamic(controller);

  H = forwardDynamics.H() - forwardDynamics.HIr();
  auto F = selectCols(H.topRows(6), activeJointIndices);
  auto FT = F.transpose();
  auto Ic0 = H.topLeftCorner(6, 6);
  auto Hsub = selectSubmatrix(H, activeJointIndices);
  auto I_c_0_inv = Ic0.inverse();

  auto Hd = coriolisMatrix + coriolisMatrix.transpose();
  // prevH = H;
  auto Fd = selectCols(Hd.topRows(6), activeJointIndices);
  auto FdT = Fd.transpose();
  auto I_c_0d = Hd.topLeftCorner(6, 6);
  auto Hdsub = selectSubmatrix(Hd, activeJointIndices);

  auto Hfb = Hsub - FT * I_c_0_inv * F;
  // mc_rtc::log::info("Cvec = {}", c_hat.transpose());
  Eigen::VectorXd Cfb = selectEntries(coriolisGravityTerm, activeJointIndices) - FT * I_c_0_inv * coriolisGravityTerm.head(6);
  // mc_rtc::log::info("Cvec = {}", c_hat.tail(dofNumber - 6).transpose().eval());
  // mc_rtc::log::info("Cvec = {}", (-FT * I_c_0_inv * c_hat.head(6)).eval());
  auto Hfbd = Hdsub - FdT * I_c_0_inv * F - FT * I_c_0_inv * Fd - FT * (-I_c_0_inv * I_c_0d * I_c_0_inv) * F;

  Eigen::VectorXd fsum = Eigen::VectorXd::Zero(6);
  for(size_t i = 0; i < realRobot.forceSensors().size(); i++)
  {
    auto jacobian = rbd::Jacobian(realRobot.mb(), realRobot.forceSensors()[i].parentBody());
    auto fsensor = realRobot.forceSensors()[i].worldWrenchWithoutGravity(realRobot);
    fsum += realRobot.posW().dualMul(fsensor).vector();
    // std::cout << "parentBody = \n" << realRobot.forceSensors()[i].parentBody() << std::endl;
    // std::cout << "fsensor = \n" << fsensor << std::endl;
    // std::cout << "posW*sensor = \n" << realRobot.posW().dualMul(fsensor).vector() << std::endl;
  }
  // std::cout << "fsum = \n" << fsum << std::endl;

  Eigen::VectorXd torque_sum = Eigen::VectorXd::Zero(actuatedDofNumber);
  for(size_t i = 0; i < realRobot.forceSensors().size(); i++)
  {
    auto jacobian = rbd::Jacobian(realRobot.mb(), realRobot.forceSensors()[i].parentBody(),
                                  realRobot.forceSensors()[i].X_fsactual_parent().translation());
    auto fsensor = realRobot.forceSensors()[i].worldWrenchWithoutGravity(realRobot);
    Eigen::MatrixXd Jac = jacobian.jacobian(realRobot.mb(), mbc, realRobot.posW());
    Eigen::MatrixXd fullJac(6, dofNumber);
    jacobian.fullJacobian(realRobot.mb(), Jac, fullJac);
    // mc_rtc::log::info("Jac, rows = {}, cols = {}", Jac.rows(), Jac.cols());
    // mc_rtc::log::info("fullJac, rows = {}, cols = {}", fullJac.rows(), fullJac.cols());
    Eigen::MatrixXd Jfb = selectCols(fullJac, activeJointIndices).transpose() - FT * I_c_0_inv;
    torque_sum += Jfb * realRobot.posW().dualMul(fsensor).vector();
    // std::cout << "parentBody = \n" << realRobot.forceSensors()[i].parentBody() << std::endl;
    // std::cout << "Jac = \n" << Jac.format(format) << std::endl;
    // std::cout << "fullJac = \n" << fullJac.format(format) << std::endl;
    // std::cout << "FT * I_c_0_inv = \n" << (FT * I_c_0_inv).format(format) << std::endl;
    // std::cout << "fsensor = \n" << fsensor << std::endl;
    // std::cout << "Jfb = \n" << Jfb.format(format) << std::endl;
    // std::cout << "J^fb*sensor = \n" << Jfb * realRobot.posW().dualMul(fsensor).vector() << std::endl;
  }
  // torque_sum.setZero();
  // std::cout << "torque_sum = \n" << torque_sum.transpose() << std::endl;
  // std::cout << "Hfbd*qdot_joint = \n" << Hfbd * qdot_joint << std::endl;
  // std::cout << "Cfb = \n" << Cfb << std::endl;
  // std::cout << "Hfb*qdot_joint = \n" << Hfb * qdot_joint << std::endl;
  // std::cout << "Ic0*qdot_fb = \n" << (Ic0 * qdot_fb).transpose() << std::endl;
  // std::cout << "F*qdot_joint = \n" << (F * qdot_joint).transpose() << std::endl;
  // std::cout << "I_c_0d*qdot_fb = \n" << (I_c_0d * qdot_fb).transpose() << std::endl;
  // std::cout << "Fd*qdot_joint = \n" << (Fd * qdot_joint).transpose() << std::endl;
  // std::cout << "pc0 = \n" << coriolisGravityTerm.head(6).transpose() << std::endl;

  fsum.setZero();
  torque_sum.setZero();

  // Eigen::VectorXd sensor_tau(dofNumber);
  // sensor_tau << fsum, torque_sum;

  integralTermNormal +=
      (tau + (coriolisMatrix + coriolisMatrix.transpose()) * qdot - forwardDynamics.C() + residualNormal)
      * ctl.timestep();
  residualNormal = residualGains * (H * qdot - integralTermNormal);
  // residualNormal.head(6).setZero();

  integralTermIntern += (tau_joint + torque_sum + Hfbd * qdot_joint - Cfb + internResidual) * ctl.timestep();
  internResidual = residualGains * (Hfb * qdot_joint - integralTermIntern);
  integralTermExtern +=
      (I_c_0d * qdot_fb + Fd * qdot_joint - coriolisGravityTerm.head(6) + fsum + externResidual) * ctl.timestep();
  externResidual = residualGains * (Ic0 * qdot_fb + F * qdot_joint - integralTermExtern);
  // std::cout << "integralTermIntern = \n" << integralTermIntern.transpose() << std::endl;
  // std::cout << "internResidual = \n" << internResidual.transpose() << std::endl;
  // std::cout << "integralTermExtern = \n" << integralTermExtern.transpose() << std::endl;
  // std::cout << "externResidual = \n" << externResidual.transpose() << std::endl;

  Eigen::VectorXd residual_fb(6 + actuatedDofNumber);
  residual_fb.head(6) = externResidual;
  residual_fb.tail(actuatedDofNumber) = internResidual;

  Eigen::VectorXd residual = Eigen::VectorXd::Zero(dofNumber);
  residual.head(6) = externResidual;
  residual += scatterEntries(internResidual + FT * I_c_0_inv * externResidual, activeJointIndices, dofNumber);

  // std::cout << "Shoulder PTransformd" << realRobot.bodyPosW("R_SHOULDER_P_S").matrix().format(format) << std::endl;

  Eigen::MatrixXd augmented_Jfb_forces(6 + actuatedDofNumber, 6);

  size_t fsi = 0;
  for(auto & sensor : realRobot.forceSensors())
  {
    // mc_rtc::log::info("Sensor {} parentBody {}", sensor.name(), sensor.parentBody());
    rbd::Jacobian jacobian_forces = rbd::Jacobian(realRobot.mb(), sensor.parentBody());
    Eigen::MatrixXd Jac_forces = jacobian_forces.jacobian(realRobot.mb(), mbc, realRobot.posW());
    Eigen::MatrixXd fullJac_forces(6, dofNumber);
    jacobian_forces.fullJacobian(realRobot.mb(), Jac_forces, fullJac_forces);
    Eigen::MatrixXd Jfb_forces_T = selectCols(fullJac_forces, activeJointIndices).transpose() - FT * I_c_0_inv;
    augmented_Jfb_forces.block(0, 0, 6, 6).setIdentity();
    augmented_Jfb_forces.block(6, 0, actuatedDofNumber, 6) = Jfb_forces_T;
    Eigen::VectorXd estimated_wrench_fb = augmented_Jfb_forces.completeOrthogonalDecomposition().solve(residual_fb);
    Eigen::VectorXd estimated_wrench = realRobot.posW().matrix().transpose() * estimated_wrench_fb;
    EstimationAtFTSensors[fsi] = sva::ForceVecd(estimated_wrench);
    fsi++;

    // for(auto estimation : EstimationAtFTSensors)
    // {
    //   mc_rtc::log::info("Sensor {} - {}", sensor.name(), estimation.vector().transpose());
    // }
  }

  externalTorques = residual;
  Eigen::VectorXd externalAccelerations = Eigen::VectorXd::Zero(dofNumber);
  // mc_rtc::log::info("dofNumber = {}", dofNumber);
  // mc_rtc::log::info("Hfb: rows = {}, cols = {}", Hfb.rows(), Hfb.cols());
  externalAccelerations = H.ldlt().solve(externalTorques);
  // std::cout << "Equivalent Acc = \n" << externalAccelerations.transpose() << std::endl;

  std::vector<std::string> & extTorquePlugin =
      ctl.controller().datastore().get<std::vector<std::string>>("extTorquePlugin");

  if(isActive)
  {
    extTorquePlugin.push_back("ResidualEstimator");
  }
  else
  {
    extTorquePlugin.erase(std::remove(extTorquePlugin.begin(), extTorquePlugin.end(), "ResidualEstimator"),
                          extTorquePlugin.end());
  }

  // bool anotherPluginIsActive = false;
  bool onePluginIsActive = false;
  if(extTorquePlugin.size() > 1)
  {
    onePluginIsActive = true;
    for(const auto & pluginName : extTorquePlugin)
    {
      if(pluginName != "ResidualEstimator")
      {
        // anotherPluginIsActive = true;
        if(verbose)
          mc_rtc::log::info(
              "[ExternalForcesEstimator] Another plugin is active: {}, the last plugin sets the external torques.",
              pluginName);
        break;
      }
    }
  }

  if(isActive)
  {
    ctl.controller().realRobot().setExternalTorques(externalTorques);
    ctl.controller().realRobot().setExternalTorquesAcc(externalAccelerations);
    counter = 0;
  }
  else if(!onePluginIsActive)
  {
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(dofNumber);
    ctl.controller().realRobot().setExternalTorques(zero);
    ctl.controller().realRobot().setExternalTorquesAcc(zero);
    if(counter == 1) mc_rtc::log::warning("External force feedback inactive");
  }
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
                                     mc_rtc::gui::NumberInput(
                                         "Gain", [this]() { return this->residualGains; },
                                         [this](double gain)
                                         {
                                           if(gain != residualGains)
                                           {
                                             integralTermIntern.setZero();
                                             internResidual.setZero();
                                             filteredFTSensorTorques.setZero();
                                           }
                                           residualGains = gain;
                                         }),
                                     mc_rtc::gui::NumberInput(
                                         "Residual speed gain", [this]() { return this->residualSpeedGain; },
                                         [this](double gainSpeed)
                                         {
                                           if(gainSpeed != residualSpeedGain)
                                           {
                                             integralTermSpeed.setZero();
                                             residualSpeed.setZero();
                                           }
                                           residualSpeedGain = gainSpeed;
                                         }),
                                     mc_rtc::gui::Label("nrDof", [this]() { return this->dofNumber; }));

  auto fConf = mc_rtc::gui::ForceConfig();
  // fConf.color = mc_rtc::gui::Color::Blue;
  fConf.force_scale = 0.01;

  ctl.controller().gui()->addElement({"Plugins", "External forces estimator"},
                                     mc_rtc::gui::Force(
                                         "EndEffector", fConf, [this]() { return this->externalForces; },
                                         [this, &controller]()
                                         {
                                           auto transform = controller.robot().bodyPosW(
                                               controller.robot().frame(referenceFrame).body());
                                           return transform;
                                         }));

  fConf.color = mc_rtc::gui::Color::Yellow;

  ctl.controller().gui()->addElement(
      {"Plugins", "External forces estimator"},
      mc_rtc::gui::Force(
          "EndEffector Residual", fConf, [this]() { return this->externalForcesResidual; },
          [this, &controller]()
          {
            auto transform = controller.robot().bodyPosW(controller.robot().frame(referenceFrame).body());
            return transform;
          }));

  fConf.color = mc_rtc::gui::Color::Red;

  ctl.controller().gui()->addElement(
      {"Plugins", "External forces estimator"},
      mc_rtc::gui::Force(
          "EndEffector F/T sensor", fConf, [this]()
          { return sva::ForceVecd(this->externalForcesFT.segment(0, 3), this->externalForcesFT.segment(3, 3)); },
          [this, &controller]()
          {
            auto transform = controller.robot().bodyPosW(controller.robot().frame(referenceFrame).body());
            return transform;
          }));

  fConf.color = mc_rtc::gui::Color::Blue;

  size_t fsi = 0;
  for(auto & sensor : ctl.robot().forceSensors())
  {
    ctl.controller().gui()->addElement({"Plugins", "External forces estimator"},
                                       mc_rtc::gui::Force(
                                           fmt::format("Estimation at {}", sensor.name()), fConf, [this, fsi]()
                                           { return this->EstimationAtFTSensors[fsi]; }, [this, sensor, &controller]()
                                           { return controller.realRobot().bodyPosW(sensor.parent()); }));
    fsi++;
  }
}

void ExternalForcesEstimator::addLog(mc_control::MCGlobalController & controller)
{
  controller.controller().logger().addLogEntry("ExternalForceEstimator_alpha", [&, this]() { return alphas; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_inputTorque",
                                               [&, this]() { return inputTorque; });
  controller.controller().logger().addLogEntry("gravity", [&, this]() { return gravity; });
  controller.controller().logger().addLogEntry("commanded_acceleration", [&, this]() { return commandedAcceleration; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_gain",
                                               [&, this]() { return this->residualGains; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_wrench",
                                               [&, this]() { return this->externalForces; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_non_filtered_wrench",
                                               [&, this]() { return this->newExternalForces; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_residual_joint_torque",
                                               [&, this]() { return this->internResidual; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_external_residual_joint_torque",
                                               [&, this]() -> Eigen::Vector6d { return this->externResidual; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_residual_wrench",
                                               [&, this]() { return this->externalForcesResidual; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_integralTerm",
                                               [&, this]() { return this->integralTermIntern; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_FTSensor_filtered_torque",
                                               [&, this]() { return this->filteredFTSensorTorques; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_FTSensor_filtered_wrench",
                                               [&, this]() { return this->filteredFTSensorForces; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_FTSensor_torque",
                                               [&, this]() { return this->FTSensorTorques; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_FTSensor_wrench",
                                               [&, this]() { return this->externalForcesFT; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_non_filtered_torque_value",
                                               [&, this]() { return this->newExternalTorques; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_torque_value",
                                               [&, this]() { return this->externalTorques; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_isActive",
                                               [&, this]() { return this->isActive; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_residualWithRotorInertia",
                                               [&, this]() { return this->residualWithRotorInertia; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_residualSpeed",
                                               [&, this]() { return this->residualSpeed; });
}

void ExternalForcesEstimator::removeLog(mc_control::MCGlobalController & controller)
{
  controller.controller().logger().removeLogEntry("ExternalForceEstimator_gain");
  controller.controller().logger().removeLogEntry("ExternalForceEstimator_wrench");
  controller.controller().logger().removeLogEntry("ExternalForceEstimator_residual_joint_torque");
  controller.controller().logger().removeLogEntry("ExternalForceEstimator_residual_wrench");
  controller.controller().logger().removeLogEntry("ExternalForceEstimator_integralTerm");
  controller.controller().logger().removeLogEntry("ExternalForceEstimator_FTSensor_filtered");
  controller.controller().logger().removeLogEntry("ExternalForceEstimator_FTSensor_torque");
  controller.controller().logger().removeLogEntry("ExternalForceEstimator_FTSensor_wrench");
  controller.controller().logger().removeLogEntry("ExternalForceEstimator_torque_value");
  controller.controller().logger().removeLogEntry("ExternalForceEstimator_isActive");
}

} // namespace mc_plugin

EXPORT_MC_RTC_PLUGIN("ExternalForcesEstimator", mc_plugin::ExternalForcesEstimator)
