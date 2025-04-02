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
  auto & realRobot = ctl.realRobot(ctl.robots()[0].name());
  auto & rjo = robot.refJointOrder();

  dt = ctl.timestep();

  jointNumber = ctl.robot(ctl.robots()[0].name()).refJointOrder().size();

  if(!ctl.controller().datastore().has("ros_spin"))
  {
     ctl.controller().datastore().make<bool>("ros_spin", false);
  }

  if(!ctl.controller().datastore().has("extTorquePlugin"))
  {
     ctl.controller().datastore().make_initializer<std::vector<std::string>>("extTorquePlugin", "");
  }

  if(!robot.hasDevice<mc_rbdyn::ExternalTorqueSensor>("externalTorqueSensor"))
  {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[ExternalForcesEstimator][Init] No \"ExternalTorqueSensor\" with the name \"externalTorqueSensor\" found in "
        "the robot module, please add one to the robot's RobotModule.");
  }
  extTorqueSensor = &robot.device<mc_rbdyn::ExternalTorqueSensor>("externalTorqueSensor");

  if(!robot.hasDevice<mc_rbdyn::VirtualTorqueSensor>("virtualTorqueSensor"))
  {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[ExternalForcesEstimator][Init] No \"VirtualTorqueSensor\" with the name \"virtualTorqueSensor\" found in the "
        "robot module, please add one to the robot's RobotModule.");
  }
  virtTorqueSensor = &robot.device<mc_rbdyn::VirtualTorqueSensor>("virtualTorqueSensor");

  Eigen::VectorXd qdot(jointNumber);
  for(size_t i = 0; i < jointNumber; i++)
  {
    qdot[i] = robot.alpha()[robot.jointIndexByName(rjo[i])][0];
  }

  // load config
  residualGains = config("residual_gain", 0.0);
  referenceFrame = config("reference_frame", (std::string) "");
  verbose = config("verbose", false);
  use_force_sensor_ = config("use_force_sensor", false);
  ros_force_sensor_ = config("ros_force_sensor", false);
  use_cmd_torque_ = config("use_commanded_torque", false);
  residualSpeedGain = config("residual_speed_gain", 100.0);
  // config loaded

  jac = rbd::Jacobian(robot.mb(), referenceFrame);
  coriolis = new rbd::Coriolis(robot.mb());
  forwardDynamics = rbd::ForwardDynamics(robot.mb());

  forwardDynamics.computeH(robot.mb(), robot.mbc());
  auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  pzero = inertiaMatrix * qdot;

  integralTerm = Eigen::VectorXd::Zero(jointNumber);
  integralTermWithRotorInertia = Eigen::VectorXd::Zero(jointNumber);
  residual = Eigen::VectorXd::Zero(jointNumber);
  residualWithRotorInertia = Eigen::VectorXd::Zero(jointNumber);
  FTSensorTorques = Eigen::VectorXd::Zero(jointNumber);
  filteredFTSensorTorques = Eigen::VectorXd::Zero(jointNumber);
  newExternalTorques = Eigen::VectorXd::Zero(jointNumber);
  filteredExternalTorques = Eigen::VectorXd::Zero(jointNumber);
  externalForces = sva::ForceVecd::Zero();
  externalForcesResidual = sva::ForceVecd::Zero();
  externalForcesFT = Eigen::Vector6d::Zero();

  integralTermSpeed = Eigen::VectorXd::Zero(jointNumber);
  residualSpeed = Eigen::VectorXd::Zero(jointNumber);

  counter = 0;

  format = Eigen::IOFormat(2, 0, " ", "\n", " ", " ", "[", "]");

  // Create datastore's entries to change modify parameters from code
  ctl.controller().datastore().make_call("EF_Estimator::isActive", [this]() { return this->isActive; });
  ctl.controller().datastore().make_call("EF_Estimator::toggleActive", [this]() { this->isActive = !this->isActive; });
  ctl.controller().datastore().make_call("EF_Estimator::useForceSensor", [this]() { return this->use_force_sensor_; });
  ctl.controller().datastore().make_call("EF_Estimator::toggleForceSensor",
                                         [this]() { this->use_force_sensor_ = !this->use_force_sensor_; });
  ctl.controller().datastore().make_call("EF_Estimator::setGain",
                                         [this](double gain)
                                         {
                                           this->integralTerm.setZero();
                                           this->residual.setZero();
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

  // mc_rtc::log::info("[ExternalForceEstimator][ROS] Force: {} | Couple: {}",
  //   wrench_sub_.data().value().force().transpose(),
  //   wrench_sub_.data().value().couple().transpose()
  // );

  auto & robot = ctl.robot();
  auto & realRobot = ctl.realRobot(ctl.robots()[0].name());

  auto & rjo = realRobot.refJointOrder();

  Eigen::VectorXd qdot(jointNumber), tau(jointNumber);
  rbd::paramToVector(realRobot.alpha(), qdot);
  if(use_cmd_torque_)
  {
    rbd::paramToVector(robot.jointTorque(), tau);
  }
  else
  {
    tau = Eigen::VectorXd::Map(realRobot.jointTorques().data(), realRobot.jointTorques().size());
  }

  auto R = controller.robot().bodyPosW(referenceFrame).rotation();

  // rbd::forwardKinematics(realRobot.mb(), realRobot.mbc());
  // rbd::forwardVelocity(realRobot.mb(), realRobot.mbc());
  // rbd::forwardAcceleration(realRobot.mb(), realRobot.mbc());
  forwardDynamics.computeC(realRobot.mb(), realRobot.mbc());
  forwardDynamics.computeH(realRobot.mb(), realRobot.mbc());
  auto coriolisMatrix = coriolis->coriolis(realRobot.mb(), realRobot.mbc());
  auto coriolisGravityTerm = forwardDynamics.C();

  // std::cout << "==================== Coriolis matrix ====================" << std::endl;
  // std::cout << coriolisMatrix.format(format) << std::endl;
  // std::cout << "==================== Coriolis Term ====================" << std::endl;
  // std::cout << coriolisGravityTerm.format(format) << std::endl;

  integralTerm += (tau + (coriolisMatrix + coriolisMatrix.transpose()) * qdot - coriolisGravityTerm
                   + virtTorqueSensor->torques() + residual)
                  * ctl.timestep();
  auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  auto pt = inertiaMatrix * qdot;

  residual = residualGains * (pt - integralTerm + pzero);

  auto inertiaMatrixWithRotorInertia = forwardDynamics.H();
  auto ptWithRotorInertia = inertiaMatrixWithRotorInertia * qdot;
  integralTermWithRotorInertia += (tau + (coriolisMatrix + coriolisMatrix.transpose()) * qdot - coriolisGravityTerm
                   + virtTorqueSensor->torques() + residualWithRotorInertia)
                  * ctl.timestep();
  residualWithRotorInertia = residualGains * (ptWithRotorInertia - integralTermWithRotorInertia + pzero);
  
  // Residual speed observer
  integralTermSpeed += (tau + (coriolisMatrix + coriolisMatrix.transpose()) * qdot - coriolisGravityTerm
                    + residualSpeed) * ctl.timestep();
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
  Eigen::VectorXd FR = jTranspose.completeOrthogonalDecomposition().solve(residual);
  externalForcesResidual = sva::ForceVecd(FR);
  externalForcesResidual.force() = R * externalForcesResidual.force();
  externalForcesResidual.couple() = R * externalForcesResidual.couple();
  // mc_rtc::log::info("===== {}", jTranspose.completeOrthogonalDecomposition().pseudoInverse()*jTranspose);

  auto sva_EF_FT = realRobot.forceSensor("EEForceSensor").wrenchWithoutGravity(realRobot);
  if(!ros_force_sensor_) sva_EF_FT = sva_EF_FT.Zero();
  externalForcesFT = sva_EF_FT.vector();
  // Applying some rotation so it match the same world as the residual
  externalForces.force() = R.transpose() * sva_EF_FT.force();
  externalForces.couple() = R.transpose() * sva_EF_FT.couple();
  FTSensorTorques = jac.jacobian(realRobot.mb(), realRobot.mbc()).transpose() * (externalForces.vector());
  double alpha = 1 - exp(-dt * residualGains);
  filteredFTSensorTorques += alpha * (FTSensorTorques - filteredFTSensorTorques);
  newExternalTorques = residual + (FTSensorTorques - filteredFTSensorTorques);
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
    externalTorques = residual;
  }

  externalForces = sva::ForceVecd(jac.jacobian(realRobot.mb(), realRobot.mbc())
                                      .transpose()
                                      .completeOrthogonalDecomposition()
                                      .solve(externalTorques));
  externalForces.force() = R * externalForces.force();
  externalForces.couple() = R * externalForces.couple();

  counter++;


  std::vector<std::string> & extTorquePlugin = ctl.controller().datastore().get<std::vector<std::string>>("extTorquePlugin");

  if(isActive)
  {
    extTorquePlugin.push_back("ResidualEstimator");
  }
  else
  {
    extTorquePlugin.erase(std::remove(extTorquePlugin.begin(), extTorquePlugin.end(), "ResidualEstimator"), extTorquePlugin.end());
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
        if (verbose) mc_rtc::log::info("[ExternalForcesEstimator] Another plugin is active: {}, the last plugin sets the external torques.", pluginName);
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
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(jointNumber);
    extTorqueSensor->torques(zero);
    if(counter == 1) mc_rtc::log::warning("External force feedback inactive");
  }

  

  // mc_rtc::log::info("ExternalForcesEstimator::before");
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

  ctl.controller().gui()->addElement({"Plugins", "External forces estimator"},
                                     mc_rtc::gui::Checkbox("Is estimation feedback active", isActive),
                                     mc_rtc::gui::Checkbox("Use force sensor", use_force_sensor_),
                                     mc_rtc::gui::Checkbox("Use force sensor over ROS", ros_force_sensor_),
                                     mc_rtc::gui::Checkbox("Use commanded torque", use_cmd_torque_),
                                     mc_rtc::gui::NumberInput(
                                         "Gain", [this]() { return this->residualGains; },
                                         [this](double gain)
                                         {
                                           if(gain != residualGains)
                                           {
                                             integralTerm.setZero();
                                             residual.setZero();
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
                                           auto transform = controller.robot().bodyPosW(referenceFrame);
                                           return transform;
                                         }));

  fConf.color = mc_rtc::gui::Color::Yellow;

  ctl.controller().gui()->addElement({"Plugins", "External forces estimator"},
                                     mc_rtc::gui::Force(
                                         "EndEffector Residual", fConf,
                                         [this]() { return this->externalForcesResidual; },
                                         [this, &controller]()
                                         {
                                           auto transform = controller.robot().bodyPosW(referenceFrame);
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
            auto transform = controller.robot().bodyPosW(referenceFrame);
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
                                               [&, this]() { return this->residual; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_residual_wrench",
                                               [&, this]() { return this->externalForcesResidual; });
  controller.controller().logger().addLogEntry("ExternalForceEstimator_integralTerm",
                                               [&, this]() { return this->integralTerm; });
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
