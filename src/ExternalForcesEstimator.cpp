#include "ExternalForcesEstimator.h"
#include <mc_control/GlobalPluginMacros.h>
#include <string>
#include <vector>

namespace mc_plugin
{

ExternalForcesEstimator::~ExternalForcesEstimator() = default;

void ExternalForcesEstimator::init(mc_control::MCGlobalController & controller, const mc_rtc::Configuration & config)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  auto & robot = ctl.robot(ctl.robots()[0].name());
  auto & tvmRobot = robot.tvmRobot();
  auto & realRobot = ctl.realRobot(ctl.robots()[0].name());
  auto & rjo = robot.refJointOrder();

  dt = ctl.timestep();

  dofNumber = ctl.robot(ctl.robots()[0].name()).mb().nrDof();

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

  jac = rbd::Jacobian(robot.mb(), robot.frame(referenceFrame).body());
  coriolis = new rbd::Coriolis(robot.mb());
  forwardDynamics = rbd::ForwardDynamics(robot.mb());

  forwardDynamics.computeH(robot.mb(), robot.mbc());
  auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  format = Eigen::IOFormat(2, 0, " ", "\n", " ", " ", "[", "]");
  pzero = inertiaMatrix * qdot;

  integralTermExtern = Eigen::VectorXd::Zero(6);
  integralTermIntern = Eigen::VectorXd::Zero(dofNumber - 6);
  integralTermWithRotorInertia = Eigen::VectorXd::Zero(dofNumber);
  internResidual = Eigen::VectorXd::Zero(dofNumber - 6);
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

  counter = 0;

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

  auto & robot = ctl.robot();
  if(robot.mb().nrJoints() > 0 && robot.mb().joint(0).type() == rbd::Joint::Free)
  {
    computeForFloatingBase(controller);
  }
  else
  {
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
      rbd::paramToVector(robot.jointTorque(), tau);
      break;
    case TorqueSourceType::CurrentMeasurement:
      mc_rtc::log::error_and_throw<std::runtime_error>("Not implemented yet");
      break;
    case TorqueSourceType::MotorTorqueMeasurement:
      tau = Eigen::VectorXd::Map(realRobot.jointTorques().data(), realRobot.jointTorques().size())
            * robot.mb().joint(robot.mb().nrJoints() - 1).gearRatio();
      break;
    case TorqueSourceType::JointTorqueMeasurement:
      tau = Eigen::VectorXd::Map(realRobot.jointTorques().data(), realRobot.jointTorques().size());
      break;
  }

  auto R = controller.robot().bodyPosW(robot.frame(referenceFrame).body()).rotation();

  forwardDynamics.computeC(realRobot.mb(), realRobot.mbc());
  forwardDynamics.computeH(realRobot.mb(), realRobot.mbc());
  auto coriolisMatrix = coriolis->coriolis(realRobot.mb(), realRobot.mbc());
  auto coriolisGravityTerm = forwardDynamics.C();

  integralTermIntern += (tau + (coriolisMatrix + coriolisMatrix.transpose()) * qdot - coriolisGravityTerm
                         + extTorqueSensor->torques() + internResidual)
                        * ctl.timestep();
  auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  auto pt = inertiaMatrix * qdot;

  internResidual = residualGains * (pt - integralTermIntern + pzero);

  auto inertiaMatrixWithRotorInertia = forwardDynamics.H();
  auto ptWithRotorInertia = inertiaMatrixWithRotorInertia * qdot;
  integralTermWithRotorInertia += (tau + (coriolisMatrix + coriolisMatrix.transpose()) * qdot - coriolisGravityTerm
                                   + extTorqueSensor->torques() + residualWithRotorInertia)
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

  auto sva_EF_FT = realRobot.forceSensor(ft_sensor_name_).wrenchWithoutGravity(realRobot);
  // if(!use_force_sensor_) sva_EF_FT = sva_EF_FT.Zero();
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

  if(use_force_sensor_)
  {
    externalTorques = filteredExternalTorques;
  }
  else
  {
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

  Eigen::VectorXd qdot(dofNumber), tau(dofNumber);
  qdot = rbd::paramToVector(realRobot.mb(), realRobot.alpha());

  switch(tau_mes_src_)
  {
    case TorqueSourceType::CommandedTorque:
      rbd::paramToVector(robot.jointTorque(), tau);
      break;
    case TorqueSourceType::CurrentMeasurement:
      mc_rtc::log::error_and_throw<std::runtime_error>("Not implemented yet");
      break;
    case TorqueSourceType::MotorTorqueMeasurement:
      tau = Eigen::Map<const Eigen::VectorXd>(realRobot.jointTorques().data(), realRobot.jointTorques().size())
            * robot.mb().joint(robot.mb().nrJoints() - 1).gearRatio();
      break;
    case TorqueSourceType::JointTorqueMeasurement:
      tau = Eigen::Map<const Eigen::VectorXd>(realRobot.jointTorques().data(), realRobot.jointTorques().size());

      break;
  }

  std::cout << "==============================" << std::endl;
  Eigen::VectorXd qdot_fb = qdot.head<6>();
  Eigen::VectorXd qdot_joint = qdot.tail(dofNumber - 6);
  Eigen::VectorXd tau_joint = tau.tail(dofNumber - 6);
  std::cout << "qdot_fb = \n" << qdot_fb << std::endl;
  std::cout << "qdot_joint = \n" << qdot_joint << std::endl;
  std::cout << "tau_joint = \n" << tau_joint << std::endl;

  auto R = controller.robot().bodyPosW(robot.frame(referenceFrame).body()).rotation();

  forwardDynamics.computeC(realRobot.mb(), realRobot.mbc());
  forwardDynamics.computeH(realRobot.mb(), realRobot.mbc());
  auto coriolisMatrix = coriolis->coriolis(realRobot.mb(), realRobot.mbc());
  auto coriolisGravityTerm = forwardDynamics.C();

  format = Eigen::IOFormat(2, 0, " ", "\n", " ", " ", "[", "]");

  auto H = forwardDynamics.H() - forwardDynamics.HIr();
  auto F = H.topRightCorner(6, dofNumber - 6);
  auto FT = H.bottomLeftCorner(dofNumber - 6, 6);
  auto Ic0 = H.topLeftCorner(6, 6);
  auto Hsub = H.bottomRightCorner(dofNumber - 6, dofNumber - 6);
  auto I_c_0_inv = Ic0.inverse();

  auto Hd = coriolisMatrix + coriolisMatrix.transpose();
  auto Fd = Hd.topRightCorner(6, dofNumber - 6);
  auto FdT = Hd.bottomLeftCorner(dofNumber - 6, 6);
  auto I_c_0d = Hd.topLeftCorner(6, 6);
  auto Hdsub = Hd.bottomRightCorner(dofNumber - 6, dofNumber - 6);

  auto Hfb = Hsub - FT * I_c_0_inv * F;
  auto Cfb = coriolisGravityTerm.tail(dofNumber - 6);
  Cfb = coriolisGravityTerm.tail(dofNumber - 6) - FT * I_c_0_inv * coriolisGravityTerm.head<6>();
  auto Hfbd = Hdsub - FdT * I_c_0_inv * F - F.transpose() * I_c_0_inv * Fd - FT * (-I_c_0_inv * I_c_0d * I_c_0_inv) * F;

  Eigen::VectorXd fsum = Eigen::VectorXd::Zero(6);
  for(size_t i = 0; i < realRobot.forceSensors().size(); i++)
  {
    auto jacobian = rbd::Jacobian(realRobot.mb(), realRobot.forceSensors()[i].parentBody());
    auto fsensor = realRobot.forceSensors()[i].worldWrenchWithoutGravity(realRobot);
    fsum += jacobian.jacobian(realRobot.mb(), realRobot.mbc()) * realRobot.posW().dualMul(fsensor).vector();
  }

  Eigen::VectorXd torque_sum = Eigen::VectorXd::Zero(dofNumber - 6);
  for(size_t i = 0; i < realRobot.forceSensors().size(); i++)
  {
    auto jacobian = rbd::Jacobian(realRobot.mb(), realRobot.forceSensors()[i].parentBody());
    auto fsensor = realRobot.forceSensors()[i].worldWrenchWithoutGravity(realRobot);
    Eigen::MatrixXd Jac = jacobian.bodyJacobian(realRobot.mb(), realRobot.mbc());
    Eigen::MatrixXd fullJac(6, dofNumber);
    jacobian.fullJacobian(realRobot.mb(), Jac, fullJac);
    Eigen::MatrixXd Jfb = fullJac.transpose() - FT * I_c_0_inv;
    torque_sum += Jfb * realRobot.posW().dualMul(fsensor).vector();
    std::cout << "Jac = \n" << Jac.format(format) << std::endl;
    std::cout << "FT * I_c_0_inv = \n" << (FT * I_c_0_inv).format(format) << std::endl;
    std::cout << "fsensor = \n" << fsensor << std::endl;
    std::cout << "Jfb = \n" << Jfb.format(format) << std::endl;
  }
  std::cout << "torque_sum = \n" << torque_sum << std::endl;
  std::cout << "Hfbd*qdot_joint = \n" << Hfbd * qdot_joint << std::endl;
  std::cout << "Cfb = \n" << Cfb << std::endl;
  std::cout << "Hfb*qdot_joint = \n" << Hfb * qdot_joint << std::endl;

  integralTermIntern += (tau_joint + torque_sum + Hfbd * qdot_joint - Cfb + internResidual) * ctl.timestep();
  internResidual = residualGains * (Hfb * qdot_joint - integralTermIntern);
  integralTermExtern +=
      (I_c_0d * qdot_fb + Fd * qdot_joint - coriolisGravityTerm.head<6>() + fsum + externalResidual) * ctl.timestep();
  externResidual = residualGains * (Ic0 * qdot_fb + F * qdot_joint - integralTermExtern);
  std::cout << "integralTermIntern = \n" << integralTermIntern << std::endl;
  std::cout << "internResidual = \n" << internResidual << std::endl;
  std::cout << "integralTermExtern = \n" << integralTermExtern << std::endl;
  std::cout << "externResidual = \n" << externResidual << std::endl;

  Eigen::VectorXd residual(dofNumber);
  residual.head<6>() = externResidual;
  residual.tail(dofNumber - 6) = internResidual;

  externalTorques = residual;

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
                                         }));

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
}

void ExternalForcesEstimator::addLog(mc_control::MCGlobalController & controller)
{
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
