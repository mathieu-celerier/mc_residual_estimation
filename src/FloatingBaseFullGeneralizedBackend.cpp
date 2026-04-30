#include "EstimatorMathUtils.h"
#include "ExternalForcesEstimator.h"

#include <mc_control/mc_global_controller.h>

namespace mc_plugin
{

namespace
{

struct FloatingBaseFullGeneralizedBackend final : EstimatorBackend
{
  const char * name() const override { return "FloatingBaseFullGeneralized"; }
  void addToGui(ExternalForcesEstimator & estimator, mc_control::MCGlobalController & controller) override;
  void addToLogger(ExternalForcesEstimator & estimator, mc_control::MCGlobalController & controller) override;

  ExternalForcesEstimator::EstimatorResult run(ExternalForcesEstimator & estimator,
                                               mc_control::MCGlobalController & controller) override
  {
    return estimator.computeForFloatingBaseFullGeneralized(controller);
  }
};

} // namespace

std::unique_ptr<EstimatorBackend> makeFloatingBaseFullGeneralizedBackend()
{
  return std::make_unique<FloatingBaseFullGeneralizedBackend>();
}

void FloatingBaseFullGeneralizedBackend::addToGui(ExternalForcesEstimator & estimator,
                                                  mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);
  auto fConf = mc_rtc::gui::ForceConfig();
  fConf.force_scale = 0.01;

  fConf.color = mc_rtc::gui::Color::Blue;
  ctl.controller().gui()->addElement({"Plugins", "External forces estimator"},
                                     mc_rtc::gui::Force(
                                         "EndEffector", fConf,
                                         [&estimator]() { return estimator.forceFusionState().fusedWrench; },
                                         [&controller, &estimator]()
                                         {
                                           auto transform = controller.robot().bodyPosW(
                                               controller.robot().frame(estimator.referenceFrameName()).body());
                                           return transform;
                                         }));

  fConf.color = mc_rtc::gui::Color::Yellow;
  ctl.controller().gui()->addElement({"Plugins", "External forces estimator"},
                                     mc_rtc::gui::Force(
                                         "EndEffector Residual", fConf,
                                         [&estimator]() { return estimator.forceFusionState().residualWrench; },
                                         [&controller, &estimator]()
                                         {
                                           auto transform = controller.robot().bodyPosW(
                                               controller.robot().frame(estimator.referenceFrameName()).body());
                                           return transform;
                                         }));

  fConf.color = mc_rtc::gui::Color::Red;
  ctl.controller().gui()->addElement({"Plugins", "External forces estimator"},
                                     mc_rtc::gui::Force(
                                         "EndEffector F/T sensor", fConf,
                                         [&estimator]()
                                         {
                                           const auto & sensor = estimator.forceFusionState().sensorWrench;
                                           return sva::ForceVecd(sensor.segment(0, 3), sensor.segment(3, 3));
                                         },
                                         [&controller, &estimator]()
                                         {
                                           auto transform = controller.robot().bodyPosW(
                                               controller.robot().frame(estimator.referenceFrameName()).body());
                                           return transform;
                                         }));

  fConf.color = mc_rtc::gui::Color::Blue;
  size_t fsi = 0;
  for(const auto & sensor : ctl.robot().forceSensors())
  {
    ctl.controller().gui()->addElement({"Plugins", "External forces estimator"},
                                       mc_rtc::gui::Force(
                                           fmt::format("Estimation at {}", sensor.name()), fConf,
                                           [&estimator, fsi]() { return estimator.forceSensorEstimations()[fsi]; },
                                           [&controller, sensor]()
                                           { return controller.realRobot().bodyPosW(sensor.parent()); }));
    fsi++;
  }
}

void FloatingBaseFullGeneralizedBackend::addToLogger(ExternalForcesEstimator & estimator,
                                                     mc_control::MCGlobalController & controller)
{
  auto & logger = controller.controller().logger();
  logger.addLogEntry("ExternalForceEstimator_wrench", &estimator,
                     [&estimator]() { return estimator.forceFusionState().fusedWrench; });
  logger.addLogEntry("ExternalForceEstimator_non_filtered_wrench", &estimator,
                     [&estimator]() { return estimator.forceFusionState().unfilteredWrench; });
  logger.addLogEntry("ExternalForceEstimator_residual_joint_torque", &estimator,
                     [&estimator]() { return estimator.residualObserverState().jointResidual; });
  logger.addLogEntry("ExternalForceEstimator_external_residual_joint_torque", &estimator,
                     [&estimator]() -> Eigen::Vector6d { return estimator.residualObserverState().baseResidual; });
  logger.addLogEntry("ExternalForceEstimator_residual_wrench", &estimator,
                     [&estimator]() { return estimator.forceFusionState().residualWrench; });
  logger.addLogEntry("ExternalForceEstimator_integralTerm", &estimator,
                     [&estimator]() { return estimator.residualObserverState().integralFull; });
  logger.addLogEntry("ExternalForceEstimator_FTSensor_filtered_torque", &estimator,
                     [&estimator]() { return estimator.forceFusionState().filteredSensorTorques; });
  logger.addLogEntry("ExternalForceEstimator_FTSensor_filtered_wrench", &estimator,
                     [&estimator]() { return estimator.forceFusionState().filteredSensorWrench; });
  logger.addLogEntry("ExternalForceEstimator_FTSensor_torque", &estimator,
                     [&estimator]() { return estimator.forceFusionState().sensorTorques; });
  logger.addLogEntry("ExternalForceEstimator_FTSensor_wrench", &estimator,
                     [&estimator]() { return estimator.forceFusionState().sensorWrench; });
  logger.addLogEntry("ExternalForceEstimator_non_filtered_torque_value", &estimator,
                     [&estimator]() { return estimator.forceFusionState().fusedTorques; });
  logger.addLogEntry("ExternalForceEstimator_torque_value", &estimator,
                     [&estimator]() { return estimator.forceFusionState().publishedTorques; });
  logger.addLogEntry("ExternalForceEstimator_residualSpeed", &estimator,
                     [&estimator]() { return estimator.speedObserverState().residual; });
}

void ExternalForcesEstimator::updateFullGeneralizedResidualObserver(const Eigen::VectorXd & tau,
                                                                    const Eigen::VectorXd & qdot,
                                                                    const Eigen::VectorXd & coriolisGravityTerm,
                                                                    const Eigen::MatrixXd & coriolisMatrix,
                                                                    const Eigen::MatrixXd & inertiaMatrix,
                                                                    double timestep)
{
  residualObserver_.integralFull +=
      (tau + (coriolisMatrix + coriolisMatrix.transpose()) * qdot - coriolisGravityTerm + residualObserver_.residualFull)
      * timestep;
  residualObserver_.residualFull = residualGain * (inertiaMatrix * qdot - residualObserver_.integralFull);
}

ExternalForcesEstimator::EstimatorResult
ExternalForcesEstimator::computeForFloatingBase(mc_control::MCGlobalController & controller)
{
  return backend_->run(*this, controller);
}

ExternalForcesEstimator::EstimatorResult
ExternalForcesEstimator::computeForFloatingBaseFullGeneralized(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);
  auto inputs = buildEstimatorInputs(ctl, 6, false, false);
  updateDiagnostics(inputs);

  const auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  const auto coriolisGravityTerm = forwardDynamics.C();

  updateFullGeneralizedResidualObserver(inputs.tau, inputs.qdot, coriolisGravityTerm, inputs.coriolisMatrix,
                                        inertiaMatrix, ctl.timestep());
  const auto activeResidual = detail::selectEntries(residualObserver_.residualFull, activeJointIndices);

  EstimatorResult result;
  result.preservedPrefix = inputs.preservedPrefix;
  result.warnWhenInactive = inputs.warnWhenInactive;
  result.logPluginState = inputs.logPluginState;
  forceFusion_ = computeFullGeneralizedForceFusion(*inputs.robot, inputs.mbc, activeResidual);
  EstimationAtFTSensors.assign(static_cast<size_t>(inputs.robot->forceSensors().size()), sva::ForceVecd::Zero());
  result.torques = residualObserver_.residualFull;
  result.accelerations = inertiaMatrix.ldlt().solve(result.torques);
  forceFusion_.publishedTorques = result.torques;
  return result;
}

} // namespace mc_plugin
