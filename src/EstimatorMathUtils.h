/*
 * Copyright 2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include "ExternalForcesEstimator.h"

#include <mc_rtc/logging.h>

#include <algorithm>

namespace mc_plugin::detail
{

inline Eigen::VectorXd selectEntries(const Eigen::VectorXd & vector, const std::vector<int> & indices)
{
  Eigen::VectorXd out(indices.size());
  for(size_t i = 0; i < indices.size(); ++i)
  {
    out(static_cast<Eigen::Index>(i)) = vector(indices[i]);
  }
  return out;
}

inline Eigen::VectorXd scatterEntries(const Eigen::VectorXd & vector, const std::vector<int> & indices, int fullSize)
{
  Eigen::VectorXd out = Eigen::VectorXd::Zero(fullSize);
  for(size_t i = 0; i < indices.size(); ++i)
  {
    out(indices[i]) = vector(static_cast<Eigen::Index>(i));
  }
  return out;
}

inline void zeroInactiveEntries(Eigen::VectorXd & vector, const std::vector<int> & activeIndices, int preservedPrefix)
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

inline Eigen::MatrixXd selectRows(const Eigen::MatrixXd & matrix, const std::vector<int> & indices)
{
  Eigen::MatrixXd out(indices.size(), matrix.cols());
  for(size_t i = 0; i < indices.size(); ++i)
  {
    out.row(static_cast<Eigen::Index>(i)) = matrix.row(indices[i]);
  }
  return out;
}

inline Eigen::MatrixXd selectCols(const Eigen::MatrixXd & matrix, const std::vector<int> & indices)
{
  Eigen::MatrixXd out(matrix.rows(), indices.size());
  for(size_t i = 0; i < indices.size(); ++i)
  {
    out.col(static_cast<Eigen::Index>(i)) = matrix.col(indices[i]);
  }
  return out;
}

inline Eigen::MatrixXd selectSubmatrix(const Eigen::MatrixXd & matrix, const std::vector<int> & indices)
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

inline int jointDofOffset(const rbd::MultiBody & mb, int jointIndex)
{
  int offset = 0;
  for(int i = 0; i < jointIndex; ++i)
  {
    offset += mb.joint(i).dof();
  }
  return offset;
}

inline Eigen::VectorXd mapFullDofByJointName(const mc_rbdyn::Robot & sourceRobot,
                                             const Eigen::VectorXd & raw,
                                             const mc_rbdyn::Robot & targetRobot,
                                             int fullSize)
{
  Eigen::VectorXd out = Eigen::VectorXd::Zero(fullSize);
  const bool sourceFloating = sourceRobot.mb().nrJoints() > 0 && sourceRobot.mb().joint(0).type() == rbd::Joint::Free;
  const bool targetFloating = targetRobot.mb().nrJoints() > 0 && targetRobot.mb().joint(0).type() == rbd::Joint::Free;
  if(sourceFloating && targetFloating)
  {
    out.head(std::min<int>(6, std::min<int>(raw.size(), fullSize))) =
        raw.head(std::min<int>(6, std::min<int>(raw.size(), fullSize)));
  }
  for(int jIndex = sourceFloating ? 1 : 0; jIndex < sourceRobot.mb().nrJoints(); ++jIndex)
  {
    const auto & sourceJoint = sourceRobot.mb().joint(jIndex);
    if(sourceJoint.dof() != 1 || !targetRobot.hasJoint(sourceJoint.name()))
    {
      continue;
    }
    const auto targetIndex = targetRobot.mb().jointIndexByName(sourceJoint.name());
    const auto & targetJoint = targetRobot.mb().joint(targetIndex);
    if(targetJoint.dof() != 1)
    {
      continue;
    }
    const auto sourceOffset = jointDofOffset(sourceRobot.mb(), jIndex);
    const auto targetOffset = jointDofOffset(targetRobot.mb(), targetIndex);
    if(sourceOffset < raw.size() && targetOffset < fullSize)
    {
      out(targetOffset) = raw(sourceOffset);
    }
  }
  return out;
}

inline Eigen::VectorXd refJointOrderToFullDof(const mc_rbdyn::Robot & sourceRobot,
                                              const Eigen::VectorXd & raw,
                                              const mc_rbdyn::Robot & targetRobot,
                                              int fullSize)
{
  Eigen::VectorXd out = Eigen::VectorXd::Zero(fullSize);
  for(Eigen::Index i = 0; i < raw.size(); ++i)
  {
    if(i >= static_cast<Eigen::Index>(sourceRobot.refJointOrder().size()))
    {
      continue;
    }
    const auto & jointName = sourceRobot.refJointOrder()[static_cast<size_t>(i)];
    if(!targetRobot.hasJoint(jointName))
    {
      continue;
    }
    const auto jointIndex = targetRobot.mb().jointIndexByName(jointName);
    const auto & joint = targetRobot.mb().joint(jointIndex);
    if(joint.dof() != 1)
    {
      continue;
    }
    out(jointDofOffset(targetRobot.mb(), jointIndex)) = raw(i);
  }
  return out;
}

inline Eigen::VectorXd sanitizeTorqueInput(const mc_rbdyn::Robot & sourceRobot,
                                           const mc_rbdyn::Robot & targetRobot,
                                           const Eigen::VectorXd & raw,
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
  if(raw.size() == static_cast<Eigen::Index>(sourceRobot.mb().nrDof()))
  {
    auto out = mapFullDofByJointName(sourceRobot, raw, targetRobot, fullSize);
    zeroInactiveEntries(out, activeIndices, preservedPrefix);
    return out;
  }
  if(raw.size() == static_cast<Eigen::Index>(sourceRobot.refJointOrder().size()))
  {
    auto out = refJointOrderToFullDof(sourceRobot, raw, targetRobot, fullSize);
    zeroInactiveEntries(out, activeIndices, preservedPrefix);
    return out;
  }
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[ExternalForcesEstimator] Unexpected torque vector size {}, expected {}, {}, {} or {}", raw.size(), fullSize,
      activeIndices.size(), sourceRobot.mb().nrDof(), sourceRobot.refJointOrder().size());
}

inline void computeForwardDynamicFlacco(const mc_rbdyn::Robot & robot,
                                        const rbd::MultiBodyConfig & mbc,
                                        Eigen::MatrixXd & H,
                                        Eigen::MatrixXd & Hd)
{
  const auto & mb = robot.mb();
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

inline Eigen::VectorXd computeCHatPc0HatFlacco(const mc_rbdyn::Robot & robot, const rbd::MultiBodyConfig & mbc)
{
  const auto & mb = robot.mb();
  Eigen::VectorXd c_hat = Eigen::VectorXd::Zero(mb.nrDof());
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
  return c_hat;
}

} // namespace mc_plugin::detail
