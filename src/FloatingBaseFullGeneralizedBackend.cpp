#include "EstimatorMathUtils.h"
#include "ExternalForcesEstimator.h"

#include <mc_control/mc_global_controller.h>

namespace mc_plugin
{

namespace
{

ExternalForcesEstimator::ForceEffectsData computeFullGeneralizedForceFusion(
    const mc_rbdyn::Robot & robot,
    const rbd::MultiBodyConfig & mbc,
    rbd::Jacobian & jac,
    const std::vector<int> & activeJointIndices,
    int actuatedDofNumber,
    const std::string & referenceFrame,
    const Eigen::VectorXd & activeResidual)
{
  ExternalForcesEstimator::ForceEffectsData fusion;
  fusion.sensorTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  fusion.filteredSensorTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  const auto R = robot.bodyPosW(robot.frame(referenceFrame).body()).rotation();

  auto jTranspose = jac.jacobian(robot.mb(), mbc);
  jTranspose.transposeInPlace();
  auto jTransposeActive = mc_plugin::detail::selectRows(jTranspose, activeJointIndices);
  fusion.residualWrench = sva::ForceVecd(jTransposeActive.completeOrthogonalDecomposition().solve(activeResidual));
  fusion.residualWrench.force() = R * fusion.residualWrench.force();
  fusion.residualWrench.couple() = R * fusion.residualWrench.couple();
  fusion.fusedWrench = fusion.residualWrench;
  fusion.filteredPublishedTorques = activeResidual;
  fusion.fusedTorques = activeResidual;
  fusion.unfilteredWrench = fusion.residualWrench;
  return fusion;
}

struct FloatingBaseFullGeneralizedBackend final : EstimatorBackend
{
  const char * name() const override { return "FloatingBaseFullGeneralized"; }
  void addToGui(ExternalForcesEstimator::EstimatorData & data,
                mc_control::MCGlobalController & controller) override;
  void addToLogger(ExternalForcesEstimator::EstimatorData & data,
                   mc_control::MCGlobalController & controller) override;

  ExternalForcesEstimator::EstimatorResult run(ExternalForcesEstimator::EstimatorData & data,
                                               mc_control::MCGlobalController & controller) override
  {
    return data.owner->computeForFloatingBaseFullGeneralized(data, controller);
  }
};

} // namespace

std::unique_ptr<EstimatorBackend> makeFloatingBaseFullGeneralizedBackend()
{
  return std::make_unique<FloatingBaseFullGeneralizedBackend>();
}

void FloatingBaseFullGeneralizedBackend::addToGui(ExternalForcesEstimator::EstimatorData & data,
                                                  mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);
  auto fConf = mc_rtc::gui::ForceConfig();
  fConf.force_scale = 0.01;

  fConf.color = mc_rtc::gui::Color::Blue;
  ctl.controller().gui()->addElement({"Plugins", "External forces estimator"},
                                     mc_rtc::gui::Force(
                                         "EndEffector", fConf,
                                         [&data]() { return data.forceEffects->fusedWrench; },
                                         [&controller, &data]()
                                         {
                                           auto transform = controller.robot().bodyPosW(
                                               controller.robot().frame(*data.referenceFrame).body());
                                           return transform;
                                         }));

  fConf.color = mc_rtc::gui::Color::Yellow;
  ctl.controller().gui()->addElement({"Plugins", "External forces estimator"},
                                     mc_rtc::gui::Force(
                                         "EndEffector Residual", fConf,
                                         [&data]() { return data.forceEffects->residualWrench; },
                                         [&controller, &data]()
                                         {
                                           auto transform = controller.robot().bodyPosW(
                                               controller.robot().frame(*data.referenceFrame).body());
                                           return transform;
                                         }));

  fConf.color = mc_rtc::gui::Color::Red;
  ctl.controller().gui()->addElement({"Plugins", "External forces estimator"},
                                     mc_rtc::gui::Force(
                                         "EndEffector F/T sensor", fConf,
                                         [&data]()
                                         {
                                           const auto & sensor = data.forceEffects->sensorWrench;
                                           return sva::ForceVecd(sensor.segment(0, 3), sensor.segment(3, 3));
                                         },
                                         [&controller, &data]()
                                         {
                                           auto transform = controller.robot().bodyPosW(
                                               controller.robot().frame(*data.referenceFrame).body());
                                           return transform;
                                         }));

  fConf.color = mc_rtc::gui::Color::Blue;
  size_t fsi = 0;
  for(const auto & sensor : ctl.robot().forceSensors())
  {
    ctl.controller().gui()->addElement({"Plugins", "External forces estimator"},
                                       mc_rtc::gui::Force(
                                           fmt::format("Estimation at {}", sensor.name()), fConf,
                                           [&data, fsi]() { return data.forceEffects->sensorForceEstimations[fsi]; },
                                           [&controller, sensor]()
                                           { return controller.realRobot().bodyPosW(sensor.parent()); }));
    fsi++;
  }
}

void FloatingBaseFullGeneralizedBackend::addToLogger(ExternalForcesEstimator::EstimatorData & data,
                                                     mc_control::MCGlobalController & controller)
{
  auto & logger = controller.controller().logger();
  logger.addLogEntry("ExternalForceEstimator_wrench", data.owner,
                     [&data]() { return data.forceEffects->fusedWrench; });
  logger.addLogEntry("ExternalForceEstimator_non_filtered_wrench", data.owner,
                     [&data]() { return data.forceEffects->unfilteredWrench; });
  logger.addLogEntry("ExternalForceEstimator_residual_joint_torque", data.owner,
                     [&data]() { return data.residuals->jointResidual; });
  logger.addLogEntry("ExternalForceEstimator_external_residual_joint_torque", data.owner,
                     [&data]() -> Eigen::Vector6d { return data.residuals->baseResidual; });
  logger.addLogEntry("ExternalForceEstimator_residual_wrench", data.owner,
                     [&data]() { return data.forceEffects->residualWrench; });
  logger.addLogEntry("ExternalForceEstimator_integralTerm", data.owner,
                     [&data]() { return data.residuals->integralFull; });
  logger.addLogEntry("ExternalForceEstimator_FTSensor_filtered_torque", data.owner,
                     [&data]() { return data.forceEffects->filteredSensorTorques; });
  logger.addLogEntry("ExternalForceEstimator_FTSensor_filtered_wrench", data.owner,
                     [&data]() { return data.forceEffects->filteredSensorWrench; });
  logger.addLogEntry("ExternalForceEstimator_FTSensor_torque", data.owner,
                     [&data]() { return data.forceEffects->sensorTorques; });
  logger.addLogEntry("ExternalForceEstimator_FTSensor_wrench", data.owner,
                     [&data]() { return data.forceEffects->sensorWrench; });
  logger.addLogEntry("ExternalForceEstimator_non_filtered_torque_value", data.owner,
                     [&data]() { return data.forceEffects->fusedTorques; });
  logger.addLogEntry("ExternalForceEstimator_torque_value", data.owner,
                     [&data]() { return data.forceEffects->publishedTorques; });
  logger.addLogEntry("ExternalForceEstimator_residualSpeed", data.owner,
                     [&data]() { return data.speedResidual->residual; });
}

void ExternalForcesEstimator::updateFullGeneralizedResidualObserver(const Eigen::VectorXd & tau,
                                                                    const Eigen::VectorXd & qdot,
                                                                    const Eigen::VectorXd & coriolisGravityTerm,
                                                                    const Eigen::MatrixXd & coriolisMatrix,
                                                                    const Eigen::MatrixXd & inertiaMatrix,
                                                                    double timestep)
{
  residuals_.integralFull +=
      (tau + (coriolisMatrix + coriolisMatrix.transpose()) * qdot - coriolisGravityTerm + residuals_.residualFull)
      * timestep;
  residuals_.residualFull = residualGain * (inertiaMatrix * qdot - residuals_.integralFull);
}

ExternalForcesEstimator::EstimatorResult
ExternalForcesEstimator::computeForFloatingBaseFullGeneralized(EstimatorData & data,
                                                                mc_control::MCGlobalController & controller)
{
  static_cast<void>(data);
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);
  auto inputs = buildEstimatorInputs(ctl, 6, false, false);
  updateDiagnostics(inputs);

  const auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  const auto coriolisGravityTerm = forwardDynamics.C();

  updateFullGeneralizedResidualObserver(inputs.tau, inputs.qdot, coriolisGravityTerm, inputs.coriolisMatrix,
                                        inertiaMatrix, ctl.timestep());
  const auto activeResidual = detail::selectEntries(residuals_.residualFull, activeJointIndices);

  EstimatorResult result;
  result.preservedPrefix = inputs.preservedPrefix;
  result.warnWhenInactive = inputs.warnWhenInactive;
  result.logPluginState = inputs.logPluginState;
  forceEffects_ = computeFullGeneralizedForceFusion(*inputs.robot, inputs.mbc, jac, activeJointIndices,
                                                    actuatedDofNumber, referenceFrame, activeResidual);
  forceEffects_.sensorForceEstimations.assign(static_cast<size_t>(inputs.robot->forceSensors().size()),
                                              sva::ForceVecd::Zero());
  result.torques = residuals_.residualFull;
  result.accelerations = inertiaMatrix.ldlt().solve(result.torques);
  forceEffects_.publishedTorques = result.torques;
  return result;
}

} // namespace mc_plugin
