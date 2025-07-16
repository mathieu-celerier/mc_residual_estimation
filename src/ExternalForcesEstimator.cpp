#include "ExternalForcesEstimator.h"
#include <mc_control/GlobalPluginMacros.h>
#include <mc_control/mc_global_controller.h>
#include <mc_rtc/logging.h>
#include <SpaceVecAlg/EigenTypedef.h>
#include <SpaceVecAlg/EigenUtility.h>
#include <SpaceVecAlg/SpaceVecAlg>
#include <Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <string>
#include <vector>

namespace mc_plugin
{

ExternalForcesEstimator::~ExternalForcesEstimator() = default;

void ExternalForcesEstimator::init(mc_control::MCGlobalController & controller, const mc_rtc::Configuration & config)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  mc_rtc::log::warning("Robot name = {}", ctl.robots()[0].name());
  auto & robot = ctl.robot(ctl.robots()[0].name());
  for(auto & j : robot.mb().joints())
  {
    mc_rtc::log::info("Plugin joint -> {}", j.name());
  }
  auto & tvmRobot = robot.tvmRobot();
  auto & realRobot = ctl.realRobot(ctl.robots()[0].name());
  auto & rjo = robot.refJointOrder();

  dt = ctl.timestep();

  dofNumber = realRobot.mb().nrDof();

  if(!ctl.controller().datastore().has("extTorquePlugin"))
  {
    ctl.controller().datastore().make_initializer<std::vector<std::string>>("extTorquePlugin", "");
  }

  if(!robot.hasDevice<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor"))
  {
    mc_rtc::log::error_and_throw<std::runtime_error>("[ExternalForcesEstimator][Init] No \"VirtualTorqueSensor\" with "
                                                     "the name \"ExtTorquesVirtSensor\" found in the "
                                                     "robot module, please add one to the robot's RobotModule.");
  }
  extTorqueSensor = &robot.device<mc_rbdyn::VirtualTorqueSensor>("ExtTorquesVirtSensor");

  Eigen::VectorXd qdot(dofNumber);
  qdot = tvmRobot.alpha()->value();

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
  }
  else if(source_type.compare("CurrentMeasurement") == 0)
  {
    tau_mes_src_ = TorqueSourceType::CurrentMeasurement;
  }
  else if(source_type.compare("MotorTorqueMeasurement") == 0)
  {
    tau_mes_src_ = TorqueSourceType::MotorTorqueMeasurement;
  }
  else if(source_type.compare("JointTorqueMeasurement") == 0)
  {
    tau_mes_src_ = TorqueSourceType::JointTorqueMeasurement;
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
  forwardDynamics.computeC(robot.mb(), robot.mbc());
  forwardDynamics.computeH(robot.mb(), robot.mbc());
  auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  prevH = inertiaMatrix;
  format = Eigen::IOFormat(2, 0, " ", "\n", " ", " ", "[", "]");
  pzero = inertiaMatrix * qdot;

  integralTermExtern = Eigen::VectorXd::Zero(6);
  if(robotIsFloatingBase)
  {
    // Remove the floating base part of the residuals
    integralTermIntern = Eigen::VectorXd::Zero(dofNumber - 6);
    internResidual = Eigen::VectorXd::Zero(dofNumber - 6);
  }
  else
  {
    integralTermIntern = Eigen::VectorXd::Zero(dofNumber);
    internResidual = Eigen::VectorXd::Zero(dofNumber);
  }
  integralTermWithRotorInertia = Eigen::VectorXd::Zero(dofNumber);
  externResidual = Eigen::VectorXd::Zero(6);
  residualWithRotorInertia = Eigen::VectorXd::Zero(dofNumber);
  FTSensorTorques = Eigen::VectorXd::Zero(dofNumber);
  filteredFTSensorTorques = Eigen::VectorXd::Zero(dofNumber);
  newExternalTorques = Eigen::VectorXd::Zero(dofNumber);
  filteredExternalTorques = Eigen::VectorXd::Zero(dofNumber);
  externalForces = sva::ForceVecd::Zero();
  externalForcesResidual = sva::ForceVecd::Zero();
  externalForcesFT = Eigen::Vector6d::Zero();

  integralTermSpeed = Eigen::VectorXd::Zero(dofNumber);
  residualSpeed = Eigen::VectorXd::Zero(dofNumber);

  for(int i = 0; i < robot.forceSensors().size(); i++)
  {
    EstimationAtFTSensors.push_back(sva::ForceVecd::Zero());
  }

  counter = 0;

  mimicExclusion.setIdentity(dofNumber, dofNumber);

  // Create datastore's entries to change modify parameters from code
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

  if(ctl.robot().encoderVelocities().empty())
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

  auto & rjo = realRobot.refJointOrder();

  Eigen::VectorXd qdot(dofNumber), tau(dofNumber);
  rbd::paramToVector(realRobot.alpha(), qdot);
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
      tau = Eigen::VectorXd::Map(realRobot.jointTorques().data(), realRobot.jointTorques().size());
      break;
  }

  auto R = controller.robot().bodyPosW(referenceFrame).rotation();

  forwardDynamics.computeC(realRobot.mb(), realRobot.mbc());
  forwardDynamics.computeH(realRobot.mb(), realRobot.mbc());
  auto coriolisMatrix = coriolis->coriolis(realRobot.mb(), realRobot.mbc());
  Eigen::VectorXd coriolisGravityTerm = forwardDynamics.C();

  integralTermIntern +=
      (tau + (coriolisMatrix + coriolisMatrix.transpose()) * qdot - coriolisGravityTerm + internResidual)
      * ctl.timestep();
  auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  auto pt = inertiaMatrix * qdot;

  internResidual = residualGains * (pt - integralTermIntern + pzero);

  auto inertiaMatrixWithRotorInertia = forwardDynamics.H();
  auto ptWithRotorInertia = inertiaMatrixWithRotorInertia * qdot;
  integralTermWithRotorInertia +=
      (tau + (coriolisMatrix + coriolisMatrix.transpose()) * qdot - coriolisGravityTerm + residualWithRotorInertia)
      * ctl.timestep();
  residualWithRotorInertia = residualGains * (ptWithRotorInertia - integralTermWithRotorInertia + pzero);

  // Residual speed observer
  integralTermSpeed +=
      (tau + (coriolisMatrix + coriolisMatrix.transpose()) * qdot - coriolisGravityTerm + residualSpeed)
      * ctl.timestep();
  residualSpeed = residualSpeedGain * (pt - integralTermSpeed + pzero);
  if(!ctl.controller().datastore().has("speed_residual"))
  {
    ctl.controller().datastore().make<Eigen::VectorXd>("speed_residual", residualSpeed);
  }
  else
  {
    ctl.controller().datastore().assign("speed_residual", residualSpeed);
  }

  auto jTranspose = jac.jacobian(realRobot.mb(), realRobot.mbc());
  jTranspose.transposeInPlace();
  Eigen::VectorXd FR = jTranspose.completeOrthogonalDecomposition().solve(internResidual);
  externalForcesResidual = sva::ForceVecd(FR);
  externalForcesResidual.force() = R * externalForcesResidual.force();
  externalForcesResidual.couple() = R * externalForcesResidual.couple();
  // mc_rtc::log::info("===== {}", jTranspose.completeOrthogonalDecomposition().pseudoInverse()*jTranspose);

  if(use_force_sensor_)
  {
    auto sva_EF_FT = realRobot.forceSensor(ft_sensor_name_).wrenchWithoutGravity(realRobot);
    externalForcesFT = sva_EF_FT.vector();
    // Applying some rotation so it match the same world as the residual
    externalForces.force() = R.transpose() * sva_EF_FT.force();
    externalForces.couple() = R.transpose() * sva_EF_FT.couple();
    FTSensorTorques = jac.jacobian(realRobot.mb(), realRobot.mbc()).transpose() * (externalForces.vector());
    double alpha = 1 - exp(-dt * residualGains);
    filteredFTSensorTorques += alpha * (FTSensorTorques - filteredFTSensorTorques);
    newExternalTorques = internResidual + (FTSensorTorques - filteredFTSensorTorques);
    filteredExternalTorques = newExternalTorques;
    filteredFTSensorForces = sva::ForceVecd(jac.jacobian(realRobot.mb(), realRobot.mbc())
                                                .transpose()
                                                .completeOrthogonalDecomposition()
                                                .solve(filteredFTSensorTorques));
    filteredFTSensorForces.force() = R * filteredFTSensorForces.force();
    filteredFTSensorForces.couple() = R * filteredFTSensorForces.couple();

    newExternalForces = sva::ForceVecd(jac.jacobian(realRobot.mb(), realRobot.mbc())
                                           .transpose()
                                           .completeOrthogonalDecomposition()
                                           .solve(newExternalTorques));
    newExternalForces.force() = R * newExternalForces.force();
    newExternalForces.couple() = R * newExternalForces.couple();
    externalTorques = filteredExternalTorques;
  }
  else
  {
    // If the force sensor is not used, we use the residual as external forces
    externalTorques = internResidual;
  }

  externalForces = sva::ForceVecd(jac.jacobian(realRobot.mb(), realRobot.mbc())
                                      .transpose()
                                      .completeOrthogonalDecomposition()
                                      .solve(externalTorques));
  externalForces.force() = R * externalForces.force();
  externalForces.couple() = R * externalForces.couple();

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
    extTorqueSensor->torques(externalTorques);
    counter = 0;
  }
  else if(!onePluginIsActive)
  {
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(dofNumber);
    extTorqueSensor->torques(zero);
    if(counter == 1) mc_rtc::log::warning("External force feedback inactive");
  }
}

void ExternalForcesEstimator::computeForFloatingBase(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  auto & robot = ctl.robot();
  auto & realRobot = ctl.realRobot(ctl.robots()[0].name());
  auto & realTvmRobot = realRobot.tvmRobot();

  auto & rjo = realRobot.refJointOrder();

  Eigen::VectorXd qdot(dofNumber), tau(dofNumber), tau_joint(dofNumber - 6);
  qdot = rbd::paramToVector(realRobot.mb(), realRobot.alpha());
  mc_rtc::log::info("Alphas size = {}", qdot.size());
  alphas = qdot;
  tau_joint.setZero();

  switch(tau_mes_src_)
  {
    case TorqueSourceType::CommandedTorque:
      tau = rbd::paramToVector(realRobot.mb(), robot.jointTorque());
      tau_joint = tau.tail(dofNumber - 6);
      break;
    case TorqueSourceType::CurrentMeasurement:
      mc_rtc::log::error_and_throw<std::runtime_error>("Not implemented yet");
      break;
    case TorqueSourceType::MotorTorqueMeasurement:
      tau_joint = Eigen::Map<const Eigen::VectorXd>(realRobot.jointTorques().data(), realRobot.jointTorques().size())
                  * robot.mb().joint(robot.mb().nrJoints() - 1).gearRatio();
      break;
    case TorqueSourceType::JointTorqueMeasurement:
      mc_rtc::log::info("JointTorques size = {}", realRobot.jointTorques().size());
      tau_joint = Eigen::Map<const Eigen::VectorXd>(realRobot.jointTorques().data(), realRobot.jointTorques().size());
      break;
  }

  // std::cout << "==============================" << std::endl;
  Eigen::VectorXd qdot_fb = qdot.head(6);
  Eigen::VectorXd qdot_joint = qdot.tail(dofNumber - 6);
  // mc_rtc::log::info("Size tau = {}", tau.size());
  // std::cout << "qdot_fb = \n" << qdot_fb.transpose() << std::endl;
  // std::cout << "qdot_joint = \n" << qdot_joint.transpose() << std::endl;
  // std::cout << "tau = \n" << tau.transpose() << std::endl;
  // std::cout << "tau_joint = \n" << tau_joint.transpose() << std::endl;

  auto R = controller.robot().bodyPosW(robot.frame(referenceFrame).body()).rotation();

  forwardDynamics.computeC(realRobot.mb(), realRobot.mbc());
  forwardDynamics.computeH(realRobot.mb(), realRobot.mbc());
  auto coriolisMatrix = coriolis->coriolis(realRobot.mb(), realRobot.mbc());
  Eigen::VectorXd coriolisGravityTerm = Eigen::VectorXd::Zero(dofNumber);
  computeCHatPc0Hat(controller);
  coriolisGravityTerm = c_hat;

  format = Eigen::IOFormat(2, 0, " ", "\n", " ", " ", "[", "]");

  // computeForwardDynamic(controller);

  H = forwardDynamics.H() - forwardDynamics.HIr();
  auto F = H.topRightCorner(6, dofNumber - 6);
  auto FT = F.transpose();
  auto Ic0 = H.topLeftCorner(6, 6);
  auto Hsub = H.bottomRightCorner(dofNumber - 6, dofNumber - 6);
  auto I_c_0_inv = Ic0.inverse();

  auto Hd = coriolisMatrix + coriolisMatrix.transpose();
  // prevH = H;
  auto Fd = Hd.topRightCorner(6, dofNumber - 6);
  auto FdT = Fd.transpose();
  auto I_c_0d = Hd.topLeftCorner(6, 6);
  auto Hdsub = Hd.bottomRightCorner(dofNumber - 6, dofNumber - 6);

  auto Hfb = Hsub - FT * I_c_0_inv * F;
  // mc_rtc::log::info("Cvec = {}", c_hat.transpose());
  Eigen::VectorXd Cfb = Eigen::VectorXd::Zero(dofNumber - 6);
  Cfb = coriolisGravityTerm.tail(dofNumber - 6) - FT * I_c_0_inv * coriolisGravityTerm.head(6);
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

  Eigen::VectorXd torque_sum = Eigen::VectorXd::Zero(dofNumber - 6);
  for(size_t i = 0; i < realRobot.forceSensors().size(); i++)
  {
    auto jacobian = rbd::Jacobian(realRobot.mb(), realRobot.forceSensors()[i].parentBody(),
                                  realRobot.forceSensors()[i].X_fsactual_parent().translation());
    auto fsensor = realRobot.forceSensors()[i].worldWrenchWithoutGravity(realRobot);
    Eigen::MatrixXd Jac = jacobian.jacobian(realRobot.mb(), realRobot.mbc(), realRobot.posW());
    Eigen::MatrixXd fullJac(6, dofNumber);
    jacobian.fullJacobian(realRobot.mb(), Jac, fullJac);
    // mc_rtc::log::info("Jac, rows = {}, cols = {}", Jac.rows(), Jac.cols());
    // mc_rtc::log::info("fullJac, rows = {}, cols = {}", fullJac.rows(), fullJac.cols());
    Eigen::MatrixXd Jfb = fullJac.block(0, 6, 6, dofNumber - 6).transpose() - FT * I_c_0_inv;
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

  integralTermIntern += (tau_joint + torque_sum + Hfbd * qdot_joint - Cfb + internResidual) * ctl.timestep();
  internResidual = residualGains * (Hfb * qdot_joint - integralTermIntern);
  integralTermExtern +=
      (I_c_0d * qdot_fb + Fd * qdot_joint - coriolisGravityTerm.head(6) + fsum + externResidual) * ctl.timestep();
  externResidual = 5 * residualGains * (Ic0 * qdot_fb + F * qdot_joint - integralTermExtern);
  // std::cout << "integralTermIntern = \n" << integralTermIntern.transpose() << std::endl;
  // std::cout << "internResidual = \n" << internResidual.transpose() << std::endl;
  // std::cout << "integralTermExtern = \n" << integralTermExtern.transpose() << std::endl;
  // std::cout << "externResidual = \n" << externResidual.transpose() << std::endl;

  Eigen::VectorXd residual_fb(dofNumber);
  residual_fb.head(6) = externResidual;
  residual_fb.tail(dofNumber - 6) = internResidual;

  Eigen::VectorXd residual(dofNumber);
  residual.head(6) = externResidual;
  residual.tail(dofNumber - 6) = internResidual + FT * I_c_0_inv * externResidual;

  // std::cout << "Shoulder PTransformd" << realRobot.bodyPosW("R_SHOULDER_P_S").matrix().format(format) << std::endl;

  Eigen::MatrixXd augmented_Jfb_forces(dofNumber, 6);

  size_t fsi = 0;
  for(auto & sensor : realRobot.forceSensors())
  {
    // mc_rtc::log::info("Sensor {} parentBody {}", sensor.name(), sensor.parentBody());
    rbd::Jacobian jacobian_forces = rbd::Jacobian(realRobot.mb(), sensor.parentBody());
    Eigen::MatrixXd Jac_forces = jacobian_forces.jacobian(realRobot.mb(), realRobot.mbc(), realRobot.posW());
    Eigen::MatrixXd fullJac_forces(6, dofNumber);
    jacobian_forces.fullJacobian(realRobot.mb(), Jac_forces, fullJac_forces);
    Eigen::MatrixXd Jfb_forces_T = fullJac_forces.block(0, 6, 6, dofNumber - 6).transpose() - FT * I_c_0_inv;
    augmented_Jfb_forces.block(0, 0, 6, 6).setIdentity();
    augmented_Jfb_forces.block(6, 0, dofNumber - 6, 6) = Jfb_forces_T;
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
  if(extTorquePlugin.size() > 0)
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
    extTorqueSensor->torques(externalTorques);
    extTorqueSensor->equivalentAcc(externalAccelerations);
    counter = 0;
  }
  else if(!onePluginIsActive)
  {
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(dofNumber);
    extTorqueSensor->torques(zero);
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

void ExternalForcesEstimator::computeCHatPc0Hat(mc_control::MCGlobalController & controller)
{
  auto mb = controller.realRobot(controller.robots()[0].name()).mb();
  auto mbc = controller.realRobot(controller.robots()[0].name()).mbc();
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
