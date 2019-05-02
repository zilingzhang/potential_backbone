#include <scalorPotential.h>
#include <scalorPotential.cpp>

#include <iostream>
#include <chrono>

#include <Eigen/Dense>

int main(){
	auto start = std::chrono::system_clock::now();
	//define a source list
	std::vector<ScalorPotential::srcStruct> src_list;

	// Create a 10m x 2m x 2m box with 8000 point source on the surface
	for(int i=-100;i<100;i++){
		for(int j=-19;j<19;j++){
			double x=i*0.05+0.025;
			double y=-1.0;
			double z=j*0.05+0.025;
			
			ScalorPotential::srcStruct source;
			source.B_Coeff.push_back(ScalorPotential::srcCoeff(1,0));
			source.srcPosition = Eigen::Vector3d(x,y,z); // x,y,z value of voxel mean
			src_list.push_back(source);

			y=1.0;
			source.srcPosition = Eigen::Vector3d(x,y,z); // x,y,z value of voxel mean
			src_list.push_back(source);
		}
	}
	for(int i=-100;i<100;i++){
		for(int j=-20;j<20;j++){
			double x=i*0.05+0.025;
			double y=j*0.05+0.025;
			double z=-1.0;
			ScalorPotential::srcStruct source;
			source.B_Coeff.push_back(ScalorPotential::srcCoeff(1,0));
			source.srcPosition = Eigen::Vector3d(x,y,z); // x,y,z value of voxel mean
			src_list.push_back(source);
			z=1.0;
			source.srcPosition = Eigen::Vector3d(x,y,z); // x,y,z value of voxel mean
			src_list.push_back(source);
		}
	}

	// ScalorPotential::srcStruct source;
	// source.B_Coeff.push_back(ScalorPotential::srcCoeff(1,0));
	// source.srcPosition = Eigen::Vector3d(1.1,2.0,3.0); // x,y,z value of voxel mean
	// src_list.push_back(source);

	//Define the tunnel Scalar Potential
	ScalorPotential tunnel(src_list);
	
	//Query for the potential and jacobian at a point
	Eigen::Vector3d point(-0.5,0,0);

	// ScalorPotentialState potential_state = tunnel.getState(point);
	// Eigen::Vector3d delta_point = potential_state.secondSpatialDerivative.colPivHouseholderQr().solve(potential_state.firstSpatialDerivative);
	// std::cout << "delta pos:\n " << delta_point << std::endl;
	// std::cout << "old value: " << potential_state.value << "\tnew value: " << tunnel.getState(point-delta_point).value << std::endl;
	// std::cout << tunnel.getState(point,-1).value << std::endl;
	// std::cout << tunnel.getState(point,-1).firstSpatialDerivative << std::endl << "----------" << std::endl;
	// std::cout << tunnel.getState(point,-1).secondSpatialDerivative << std::endl;

	//Iterate for 10 times

	// for(int i=0;i<10;i++){
	// 	std::cout << point << std::endl;
	// 	ScalorPotentialState potential_state = tunnel.getState(point);
	// 	Eigen::Vector3d delta_point = potential_state.secondSpatialDerivative.colPivHouseholderQr().solve(potential_state.firstSpatialDerivative);
	// 	point = point - delta_point;
	// }

	//Get states
	Eigen::Vector3d test_point(-2,0,0);
	//std::cout << tunnel.getState(test_point).value << std::endl;

	// auto start = std::chrono::system_clock::now();
	for(int i=0;i<10;i++){
        
		std::cout
		<< "---" << std::endl << test_point << std::endl << "---" << std::endl
		<< tunnel.getState(test_point).value << std::endl;
		Eigen::Vector3d delta(0.5,0,0);
		test_point = test_point + delta;
        
                      
	}
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;  
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
}