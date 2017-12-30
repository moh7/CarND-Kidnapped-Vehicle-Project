/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#define EPS 0.00001


using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;

    // Create a normal (Gaussian) distribution for x, y, and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

    default_random_engine gen; // "gen" is the random engine initialized earlier.

    for (int p = 0; p < num_particles; p++){

        Particle particle;
        particle.id = p;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1.0;

        particles.push_back(particle);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    double vel_dt = velocity * delta_t;
	double yaw_rate_dt = yaw_rate * delta_t;
	double vel_yawrate = velocity / yaw_rate;

    default_random_engine gen;
    normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	for (int p = 0; p < num_particles; p++){

        if (fabs(yaw_rate) < EPS){
            particles[p].x += vel_dt * cos(particles[p].theta);
            particles[p].y += vel_dt * sin(particles[p].theta);
        // particles[i].theta unchanged if yaw_rate is too small
        }
        else{
            double new_theta = particles[p].theta + yaw_rate_dt;
            particles[p].x += vel_yawrate *(sin(new_theta) - sin(particles[p].theta));
            particles[p].y += vel_yawrate *(cos(particles[p].theta) - cos(new_theta));
            particles[p].theta = new_theta;
        }

        // Add random Gaussian noise
        particles[p].x += dist_x(gen);
        particles[p].y += dist_y(gen);
        particles[p].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  for (unsigned int i = 0; i < observations.size(); i++) { // for each observation

    // Initialize minimum distance to a very large number.
    double min_distance = 10000000.0;

    // Initialize the found map in something not possible.
    int mapId = -1;

    for (unsigned j = 0; j < predicted.size(); j++) { // for each prediction

      double x_distance = observations[i].x - predicted[j].x;
      double y_distance = observations[i].y - predicted[j].y;

      double distance = x_distance * x_distance + y_distance * y_distance;

      // If the "distance" is less than min, stored the id and update min.
      if (distance < min_distance) {
        min_distance = distance;
        mapId = predicted[j].id;
      }
    }

    // Update the observation identifier.
    observations[i].id = mapId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// transform to map x coordinate
  double stdLandmarkRange = std_landmark[0];
  double stdLandmarkBearing = std_landmark[1];
  double sum_w = 0;

  for (int i = 0; i < num_particles; i++) {

    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    // Find landmarks in particle's range.
    double sensor_range_2 = sensor_range * sensor_range;
    vector<LandmarkObs> inRangeLandmarks;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
        float landmarkX = map_landmarks.landmark_list[j].x_f;
        float landmarkY = map_landmarks.landmark_list[j].y_f;
        int id = map_landmarks.landmark_list[j].id_i;
        double dX = x - landmarkX;
        double dY = y - landmarkY;

        if (dX*dX + dY*dY <= sensor_range_2) {
        inRangeLandmarks.push_back(LandmarkObs{id, landmarkX, landmarkY});
        }
    }

    // Transform observation coordinates.
    vector<LandmarkObs> mappedObservations;
    for(int j = 0; j < observations.size(); j++) {
        double xx = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
        double yy = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
        mappedObservations.push_back(LandmarkObs{observations[j].id, xx, yy});
    }

    // Observation association to landmark.
    dataAssociation(inRangeLandmarks, mappedObservations);

    // Reseting weight.
    particles[i].weight = 1.0;
    // Calculate weights.
    for(int j = 0; j < mappedObservations.size(); j++) {
      double observationX = mappedObservations[j].x;
      double observationY = mappedObservations[j].y;
      int landmarkId = mappedObservations[j].id;

      double landmarkX, landmarkY;
      int k = 0;
      bool found = false;
      while(!found && k < inRangeLandmarks.size()) {
        if (inRangeLandmarks[k].id == landmarkId) {
          found = true;
          landmarkX = inRangeLandmarks[k].x;
          landmarkY = inRangeLandmarks[k].y;
        }
        k++;
      }

      // Calculating weight.
      double dX = observationX - landmarkX;
      double dY = observationY - landmarkY;

      double weight = ( 1/(2*M_PI*stdLandmarkRange*stdLandmarkBearing)) * exp( -( dX*dX/(2*stdLandmarkRange*stdLandmarkRange) + (dY*dY/(2*stdLandmarkBearing*stdLandmarkBearing)) ) );
      if (weight == 0) {
        particles[i].weight *= EPS;
        sum_w += particles[i].weight;

      } else {
        particles[i].weight *= weight;
        sum_w += particles[i].weight;

      }
    }
  }
  	// Weights normalization to sum(weights)=1
	for (int i = 0; i < num_particles; i++){
		particles[i].weight /= sum_w;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<double> weights;
    for(int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }

    static default_random_engine gen;
    gen.seed(123);
    discrete_distribution<> dist_particles(weights.begin(), weights.end());
    vector<Particle> new_particles;
    new_particles.resize(num_particles);
    for (int i = 0; i < num_particles; i++) {
        new_particles[i] = particles[dist_particles(gen)];
    }
    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
