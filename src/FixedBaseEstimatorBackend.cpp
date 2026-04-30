#include "EstimatorMathUtils.h"
#include "ExternalForcesEstimator.h"

#include <mc_control/mc_global_controller.h>

namespace mc_plugin
{

namespace
{

struct FixedBaseEstimatorBackend final : EstimatorBackend
{
  const char * name() const override { return "FixedBase"; }
  void addToGui(ExternalForcesEstimator & estimator, mc_control::MCGlobalController & controller) override;
  void addToLogger(ExternalForcesEstimator & estimator, mc_control::MCGlobalController & controller) override;

  ExternalForcesEstimator::EstimatorResult run(ExternalForcesEstimator & estimator,
                                               mc_control::MCGlobalController & controller) override
  {
    return estimator.computeForFixedBase(controller);
  }
};

} // namespace

std::unique_ptr<EstimatorBackend> makeFixedBaseEstimatorBackend()
{
  return std::make_unique<FixedBaseEstimatorBackend>();
}

void FixedBaseEstimatorBackend::addToGui(ExternalForcesEstimator & estimator, mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);
  auto fConf = mc_rtc::gui::ForceConfig();
  fConf.force_scale = 0.01;

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

void FixedBaseEstimatorBackend::addToLogger(ExternalForcesEstimator & estimator,
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
                     [&estimator]() { return estimator.residualObserverState().integralJoint; });
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
  logger.addLogEntry("ExternalForceEstimator_residualWithRotorInertia", &estimator,
                     [&estimator]() { return estimator.residualObserverState().rotorInertiaResidual; });
  logger.addLogEntry("ExternalForceEstimator_residualSpeed", &estimator,
                     [&estimator]() { return estimator.speedObserverState().residual; });
}

void ExternalForcesEstimator::updateFixedBaseResidualObserver(const Eigen::VectorXd & tauActive,
                                                              const Eigen::VectorXd & qdotActive,
                                                              const Eigen::VectorXd & coriolisGravityTerm,
                                                              const Eigen::MatrixXd & coriolisMatrixActive,
                                                              const Eigen::MatrixXd & inertiaMatrixActive,
                                                              double timestep)
{
  residualObserver_.integralJoint +=
      (tauActive + coriolisMatrixActive * qdotActive - coriolisGravityTerm + residualObserver_.jointResidual) * timestep;
  const auto momentum = inertiaMatrixActive * qdotActive;
  residualObserver_.jointResidual = residualGain * (momentum - residualObserver_.integralJoint + pzero);
}

void ExternalForcesEstimator::updateRotorInertiaResidual(const Eigen::VectorXd & tauActive,
                                                         const Eigen::VectorXd & qdotActive,
                                                         const Eigen::VectorXd & coriolisGravityTerm,
                                                         const Eigen::MatrixXd & coriolisMatrixActive,
                                                         const Eigen::MatrixXd & inertiaMatrixWithRotorInertia,
                                                         double timestep)
{
  const auto momentumWithRotorInertia = inertiaMatrixWithRotorInertia * qdotActive;
  residualObserver_.rotorInertiaIntegral +=
      (tauActive + coriolisMatrixActive * qdotActive - coriolisGravityTerm + residualObserver_.rotorInertiaResidual)
      * timestep;
  residualObserver_.rotorInertiaResidual =
      residualGain * (momentumWithRotorInertia - residualObserver_.rotorInertiaIntegral + pzero);
}

void ExternalForcesEstimator::updateSpeedResidualObserver(const Eigen::VectorXd & tauActive,
                                                          const Eigen::VectorXd & qdotActive,
                                                          const Eigen::VectorXd & coriolisGravityTerm,
                                                          const Eigen::MatrixXd & coriolisMatrixActive,
                                                          const Eigen::MatrixXd & inertiaMatrixActive,
                                                          double timestep)
{
  speedObserver_.integral +=
      (tauActive + coriolisMatrixActive * qdotActive - coriolisGravityTerm + speedObserver_.residual) * timestep;
  const auto momentum = inertiaMatrixActive * qdotActive;
  speedObserver_.residual = residualSpeedGain * (momentum - speedObserver_.integral + pzero);
}

ExternalForcesEstimator::EstimatorResult
ExternalForcesEstimator::computeForFixedBase(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);
  auto inputs = buildEstimatorInputs(ctl, 0, true, true);
  updateDiagnostics(inputs);

  const auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  const auto inertiaMatrixActive = detail::selectSubmatrix(inertiaMatrix, activeJointIndices);
  const auto qdotActive = detail::selectEntries(inputs.qdot, activeJointIndices);
  const auto tauActive = detail::selectEntries(inputs.tau, activeJointIndices);
  const auto coriolisGravityTerm = detail::selectEntries(forwardDynamics.C(), activeJointIndices);
  const auto coriolisMatrixActive =
      detail::selectSubmatrix(inputs.coriolisMatrix + inputs.coriolisMatrix.transpose(), activeJointIndices);

  updateFixedBaseResidualObserver(tauActive, qdotActive, coriolisGravityTerm, coriolisMatrixActive, inertiaMatrixActive,
                                  ctl.timestep());
  ctl.controller().datastore().assign<Eigen::VectorXd>("EF_Estimator::getResidualOnly", residualObserver_.jointResidual);

  const auto inertiaMatrixWithRotorInertia = detail::selectSubmatrix(forwardDynamics.H(), activeJointIndices);
  updateRotorInertiaResidual(tauActive, qdotActive, coriolisGravityTerm, coriolisMatrixActive,
                             inertiaMatrixWithRotorInertia, ctl.timestep());
  updateSpeedResidualObserver(tauActive, qdotActive, coriolisGravityTerm, coriolisMatrixActive, inertiaMatrixActive,
                              ctl.timestep());
  updateSpeedResidualDatastore(ctl);

  EstimatorResult result;
  result.preservedPrefix = inputs.preservedPrefix;
  result.warnWhenInactive = inputs.warnWhenInactive;
  result.logPluginState = inputs.logPluginState;
  forceFusion_ = computeFixedBaseForceFusion(*inputs.robot, *inputs.realRobot, inputs.mbc, residualObserver_.jointResidual);
  EstimationAtFTSensors.assign(static_cast<size_t>(inputs.robot->forceSensors().size()), sva::ForceVecd::Zero());
  result.torques = forceFusion_.publishedTorques;
  result.accelerations = forwardDynamics.H().ldlt().solve(result.torques);
  return result;
}

} // namespace mc_plugin
