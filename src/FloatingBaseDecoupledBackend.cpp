#include "EstimatorMathUtils.h"
#include "ExternalForcesEstimator.h"

#include <mc_control/mc_global_controller.h>

namespace mc_plugin
{

namespace
{

struct FloatingBaseDecoupledBackend final : EstimatorBackend
{
  const char * name() const override { return "FloatingBaseDecoupled"; }
  void addToGui(ExternalForcesEstimator & estimator, mc_control::MCGlobalController & controller) override;
  void addToLogger(ExternalForcesEstimator & estimator, mc_control::MCGlobalController & controller) override;

  ExternalForcesEstimator::EstimatorResult run(ExternalForcesEstimator & estimator,
                                               mc_control::MCGlobalController & controller) override
  {
    return estimator.computeForFloatingBaseDecoupled(controller);
  }
};

} // namespace

std::unique_ptr<EstimatorBackend> makeFloatingBaseDecoupledBackend()
{
  return std::make_unique<FloatingBaseDecoupledBackend>();
}

void FloatingBaseDecoupledBackend::addToGui(ExternalForcesEstimator & estimator,
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

void FloatingBaseDecoupledBackend::addToLogger(ExternalForcesEstimator & estimator,
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

ExternalForcesEstimator::FloatingBaseCouplingTerms
ExternalForcesEstimator::computeFloatingBaseCouplingTerms(const Eigen::VectorXd & coriolisGravityTerm,
                                                          const Eigen::MatrixXd & inertiaMatrix,
                                                          const Eigen::MatrixXd & inertiaRateMatrix) const
{
  FloatingBaseCouplingTerms terms;
  terms.F = detail::selectCols(inertiaMatrix.topRows(6), activeJointIndices);
  terms.FT = terms.F.transpose();
  terms.Ic0 = inertiaMatrix.topLeftCorner(6, 6);
  terms.I_c_0_inv = terms.Ic0.inverse();

  const auto Hsub = detail::selectSubmatrix(inertiaMatrix, activeJointIndices);
  const auto Fd = detail::selectCols(inertiaRateMatrix.topRows(6), activeJointIndices);
  const auto FdT = Fd.transpose();
  const auto I_c_0d = inertiaRateMatrix.topLeftCorner(6, 6);
  const auto Hdsub = detail::selectSubmatrix(inertiaRateMatrix, activeJointIndices);

  terms.Hfb = Hsub - terms.FT * terms.I_c_0_inv * terms.F;
  terms.Cfb =
      detail::selectEntries(coriolisGravityTerm, activeJointIndices) - terms.FT * terms.I_c_0_inv * coriolisGravityTerm.head(6);
  terms.Hfbd = Hdsub - FdT * terms.I_c_0_inv * terms.F - terms.FT * terms.I_c_0_inv * Fd
               - terms.FT * (-terms.I_c_0_inv * I_c_0d * terms.I_c_0_inv) * terms.F;
  return terms;
}

void ExternalForcesEstimator::updateDecoupledResidualObservers(const FloatingBaseCouplingTerms & couplingTerms,
                                                               const Eigen::VectorXd & tau,
                                                               const Eigen::VectorXd & tauJoint,
                                                               const Eigen::VectorXd & qdot,
                                                               const Eigen::VectorXd & qdotBase,
                                                               const Eigen::VectorXd & qdotJoint,
                                                               const Eigen::VectorXd & coriolisGravityTerm,
                                                               double timestep)
{
  residualObserver_.integralFull +=
      (tau + Hd * qdot - coriolisGravityTerm + residualObserver_.residualFull) * timestep;
  residualObserver_.residualFull = residualGain * (H * qdot - residualObserver_.integralFull);

  residualObserver_.integralJoint +=
      (tauJoint + couplingTerms.Hfbd * qdotJoint - couplingTerms.Cfb + residualObserver_.jointResidual) * timestep;
  residualObserver_.jointResidual = residualGain * (couplingTerms.Hfb * qdotJoint - residualObserver_.integralJoint);

  const auto inertiaRateMatrix = Hd.topLeftCorner(6, 6);
  const auto forceRateMatrix = detail::selectCols(Hd.topRows(6), activeJointIndices);
  residualObserver_.integralBase +=
      (inertiaRateMatrix * qdotBase + forceRateMatrix * qdotJoint - coriolisGravityTerm.head(6) + residualObserver_.baseResidual)
      * timestep;
  residualObserver_.baseResidual =
      residualGain * (couplingTerms.Ic0 * qdotBase + couplingTerms.F * qdotJoint - residualObserver_.integralBase);
}

ExternalForcesEstimator::EstimatorResult
ExternalForcesEstimator::computeForFloatingBaseDecoupled(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);
  auto inputs = buildEstimatorInputs(ctl, 6, false, false);
  updateDiagnostics(inputs);

  const auto tauJoint = detail::selectEntries(inputs.tau, activeJointIndices);
  const auto qdotBase = inputs.qdot.head(6);
  const auto qdotJoint = detail::selectEntries(inputs.qdot, activeJointIndices);

  Eigen::VectorXd coriolisGravityTerm;

  if(forward_dynamics_mode_ == ForwardDynamicsMode::Flacco)
  {
    detail::computeForwardDynamicFlacco(*inputs.robot, inputs.mbc, H, Hd);
  }
  else
  {
    H = forwardDynamics.H() - forwardDynamics.HIr();
    Hd = inputs.coriolisMatrix + inputs.coriolisMatrix.transpose();
  }

  if(bias_term_mode_ == BiasTermMode::Flacco)
  {
    coriolisGravityTerm = detail::computeCHatPc0HatFlacco(*inputs.robot, inputs.mbc);
  }
  else
  {
    coriolisGravityTerm = forwardDynamics.C();
  }

  const auto couplingTerms = computeFloatingBaseCouplingTerms(coriolisGravityTerm, H, Hd);

  updateDecoupledResidualObservers(couplingTerms, inputs.tau, tauJoint, inputs.qdot, qdotBase, qdotJoint,
                                   coriolisGravityTerm, ctl.timestep());

  Eigen::VectorXd residualFB(6 + actuatedDofNumber);
  residualFB.head(6) = residualObserver_.baseResidual;
  residualFB.tail(actuatedDofNumber) = residualObserver_.jointResidual;

  Eigen::VectorXd residual = Eigen::VectorXd::Zero(dofNumber);
  residual.head(6) = residualObserver_.baseResidual;
  residual += detail::scatterEntries(residualObserver_.jointResidual
                                         + couplingTerms.FT * couplingTerms.I_c_0_inv * residualObserver_.baseResidual,
                                     activeJointIndices, dofNumber);

  EstimatorResult result;
  result.preservedPrefix = inputs.preservedPrefix;
  result.warnWhenInactive = inputs.warnWhenInactive;
  result.logPluginState = inputs.logPluginState;
  EstimationAtFTSensors =
      estimateFloatingBaseSensorWrenches(*inputs.robot, *inputs.realRobot, inputs.mbc, couplingTerms.FT,
                                         couplingTerms.I_c_0_inv, residualFB);
  result.torques = residual;
  result.accelerations = H.ldlt().solve(result.torques);
  forceFusion_.sensorTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  forceFusion_.filteredSensorTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  forceFusion_.fusedTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  forceFusion_.filteredPublishedTorques = Eigen::VectorXd::Zero(actuatedDofNumber);
  forceFusion_.publishedTorques = result.torques;
  return result;
}

} // namespace mc_plugin
