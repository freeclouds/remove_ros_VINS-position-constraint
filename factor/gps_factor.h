//
// Created by grn on 11/30/18.
//

#ifndef VINS_GPS_FACTOR_H
#define VINS_GPS_FACTOR_H

#include <iostream>
#include <eigen3/Eigen/Dense>

#include <ceres/ceres.h>


struct gps_struct
{
    double gpspos[3]={0.0};
    double gpscov[9]={0.0};
    double time;
};

class GPSFactor : public ceres::SizedCostFunction< 7 , 7 >
{
  public:
    GPSFactor()=delete;
    GPSFactor(gps_struct _gpsdata):gpsdata(_gpsdata){}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d gps(gpsdata.gpspos[0],gpsdata.gpspos[1],gpsdata.gpspos[2]);
        Eigen::Matrix<double, 7, 7> gpscovariance;
        gpscovariance.setZero();
        gpscovariance(0,0)=gpsdata.gpscov[0];gpscovariance(1,1)=gpsdata.gpscov[4];gpscovariance(2,2)=gpsdata.gpscov[8];
        gpscovariance(3,3)=1000000000;gpscovariance(4,4)=1000000000;gpscovariance(5,5)=1000000000;gpscovariance(6,6)=1000000000;
        Eigen::Map<Eigen::Matrix<double, 7, 1>> residual(residuals);
        residual.setZero();
        residual.block<3, 1>(0, 0)=Pi-gps;
        Eigen::Matrix<double, 7, 7> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 7, 7>>(gpscovariance.inverse()).matrixL().transpose();
        sqrt_info=sqrt_info*100000000;
        residual = sqrt_info * residual;
        cout<<sqrt_info<<endl;
        if(jacobians)
        {
            Eigen::Map<Eigen::Matrix<double, 7, 7, Eigen::RowMajor>> jacobian(jacobians[0]);
            jacobian.setZero();
            jacobian.block<3, 3>(0, 0)=Eigen::Matrix3d::Identity();
            jacobian = sqrt_info * jacobian;
        }

        return true;
    }

    gps_struct gpsdata;

};




#endif //VINS_GPS_FACTOR_H
