#include <stdio.h>
#include <cfloat>
#include <math.h>
#include <time.h>

#include <algorithm>
#include <vector>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include "opencv2/opencv.hpp"

#include <iostream>
// #include "opencv2/core/matx.hpp"
#include "opencv2/core/core.hpp"
// #include <torch/script.h>
#include <torch/extension.h>
#include "torch/script.h"
#include <torch/script.h>
#include <torch/torch.h>

#define VERTEX_CHANNELS 3
using namespace torch;
// typedef Eigen::Matrix<float,10,1,Eigen::DontAlign> Vec;
using namespace at;

int clamp(int val, int min_val, int max_val)
{
  return std::max(min_val, std::min(max_val, val));
}

// // Hough Voting layer main function
// std::vector<cv::Vec<float, 14>> Forward(const Tensor& label, const Tensor& vertex, const Tensor& extents, //const Tensor& meta_data, const Tensor& gt, int is_train);
//std::vector<std::vector<float>> Forward(const Tensor& label, const Tensor& vertex, const Tensor& extents, //const Tensor& meta_data, const Tensor& gt, int is_train);
torch::Tensor Forward(const Tensor& label, const Tensor& vertex, const Tensor& extents, const Tensor& meta_data, const Tensor& gt, int is_train);

    
// Get ground truth model 3D geometry
void getBb3Ds(const Tensor& extents, std::vector<std::vector<cv::Point3f>>& bb3Ds, int num_classes);

// // Get ground truth model 3D bounding box
inline std::vector<cv::Point3f> getBB3D(const cv::Vec<float, 3>& extent);

// // Projected 2D Bounding box
// inline cv::Rect getBB2D(int imageWidth, int imageHeight, const std::vector<cv::Point3f>& bb3D, const cv::Mat& camMat, const cv::Mat& rvec, const cv::Mat& tvec);

inline float angle_distance(cv::Point2f x, cv::Point2f n, cv::Point2f p);

void projectPoints(std::vector<cv::Point3f> bb3Ds, float& bb_distance, Eigen::MatrixXf camMat, std::vector<cv::Point2f>& bb2D);
// Hough voting functionality
void hough_voting(const Tensor& v_label, const Tensor& v_vertex, const int labelmap, const int vertmap, std::vector<std::vector<cv::Point3f>> bb3Ds, int batch, int height, int width, int num_classes, int is_train,float fx, float fy, float px, float py, std::vector<cv::Vec<float, 14> >& outputs);

// // Find better bb2D geometry
// inline void compute_width_height(const Tensor& label, const Tensor& vertex, const int labelmap, const int vertmap, cv::Point2f center, std::vector<std::vector<cv::Point3f>> bb3Ds, cv::Mat camMat, float inlierThreshold, int height, int width, int channel, int num_classes, int & bb_width, int & bb_height, float & bb_distance);
inline void compute_width_height(const Tensor& v_label, const Tensor& v_vertex, const int labelmap, const float vertmap, cv::Point2f center, std::vector<std::vector<cv::Point3f>> bb3Ds, Eigen::Matrix3f camMat,float inlierThreshold, int height, int width, int channel, int num_classes, int & bb_width, int & bb_height, float & bb_distance);

// ///////////////////////////////

// std::vector<cv::Vec<float, 14>> Forward(const Tensor& label, const Tensor& vertex, const Tensor& extents, //const Tensor& meta_data, const Tensor& gt, int is_train) 
//std::vector<std::vector<float>> Forward(const Tensor& label, const Tensor& vertex, const Tensor& extents, //const Tensor& meta_data, const Tensor& gt, int is_train) 
torch::Tensor Forward(const Tensor& label, const Tensor& vertex, const Tensor& extents, const Tensor& meta_data, const Tensor& gt, int is_train) 

{
    // Grab the input tensor
    ///////////////////////////////
    // format of the meta_data
    // intrinsic matrix: meta_data[0 ~ 8]
    // inverse intrinsic matrix: meta_data[9 ~ 17]
    // pose_world2live: meta_data[18 ~ 29]
    // pose_live2world: meta_data[30 ~ 41]
    // voxel step size: meta_data[42, 43, 44]
    // voxel min value: meta_data[45, 46, 47]
    auto v_meta_data = meta_data.view(-1);

    // const float* v_gt = gt.flat<float>().data();
    // const float* v_gt = &gt.view(-1);
    auto v_gt = gt.view(-1);
    // batch size
    int batch_size = label.size(0);
    // height
    int height = label.size(1);
    // width
    int width = label.size(2);
    
    auto v_label = label.contiguous().view(-1);
    auto v_vertex = vertex.contiguous().view(-1);
    // num of classes
    int num_classes = vertex.size(3) / VERTEX_CHANNELS;
    int num_meta_data = meta_data.size(1);
    // int num_gt = gt.size(0);
    
    std::vector<cv::Vec<float, 14> > outputs;
    auto v_extents = extents.view(-1);
    std::vector<std::vector<cv::Point3f>> bb3Ds;
    
    getBb3Ds(v_extents, bb3Ds, num_classes);

    int index_meta_data = 0;
    float fx, fy, px, py;
    auto acc_v_meta_data = v_meta_data.accessor<float,1>();
    
    for (int n = 0; n < batch_size; n++)
    {
      // these map are the starting index
      const int labelmap = n * height * width;
      const int vertmap = n * height * width * VERTEX_CHANNELS * num_classes;
      // find camera parameters
      fx = acc_v_meta_data[index_meta_data + 0];
      fy = acc_v_meta_data[index_meta_data + 4];
      px = acc_v_meta_data[index_meta_data + 2];
      py = acc_v_meta_data[index_meta_data + 5];
      
      hough_voting(v_label, v_vertex, labelmap, vertmap, bb3Ds, n, height, width, num_classes, is_train, fx, fy, px, py, outputs);
        
      index_meta_data += num_meta_data;
    }
    if (outputs.size() == 0)
    {
      std::cout << "no detection" << std::endl;
      // add a dummy detection to the output
      cv::Vec<float, 14> roi;
      roi(0) = 0;
      roi(1) = -1;
      outputs.push_back(roi);
    }
    // to change the datatype from vector<cv::Vec> to tensor
    int n_output = outputs.size();
    int size_single_roi = outputs[0].rows;
//     std::cout<<"Size_single_roi: "<<size_single_roi<<std::endl;
//     int size_single_roi = 15;
    
//     auto output_tensor = //torch::CUDA(torch::kFloat32).tensorFromBlob(outputs.data{n_output,size_single_roi});

//     std::cout<<"Type of the output: "<<typeid(output_tensor).name()<<std::endl;
//     Tensor output_tensor = at::zeros({n, size_single_roi}, Kfloat32) 
//     for (int i =0; i< n_output;i++){
//     }
    
//     std::vector<std::vector<float>> result(n_output, std::vector<float>(size_single_roi));
//     for (int i = 0; i<n_output; i++){
        
//         test[i].resize(size_single_roi);
//     }
//     return test;
//     torch::tensor output_tensor = torch::zeros({n_output, size_single_roi}, torch::kFloat32);
    at::Tensor output_tensor = torch::zeros({n_output, size_single_roi}, torch::kFloat32);
    for (int i = 0; i<n_output; i++){
        for (int j =0; j< size_single_roi; j++){
             output_tensor[i][j] = outputs[i][j];
        }
    }
    return output_tensor;
//     return outputs;
//     return output_tensor;
}






void hough_voting(const Tensor& v_label, const Tensor& v_vertex, const int labelmap, const int vertmap, std::vector<std::vector<cv::Point3f>> bb3Ds, int batch, int height, int width, int num_classes, int is_train, float fx, float fy, float px, float py, std::vector<cv::Vec<float, 14> >& outputs){
  
  float inlierThreshold = 0.9;
  int votingThreshold = 50;

  // camera intrinsic matrix 3 X 3
//   cv::Mat camMat=cv::Mat::zeros(3,3,CV_32F); 
//   int sz[] = {3,3};
//   cv::Mat camMat;
//   camMat.create(2,sz,CV_32FC1);
//   camMat = Scalar(0);
//   std::vector<std::vector<float> > camMat(3, std::vector<float>(0));
//   camMat.at([0][0]) = fx;
    
    
//   cv::Mat_<float> camMat = cv::Mat_<float>::zeros(3, 3);
//   camMat(0, 0) = fx;
//   camMat(1, 1) = fy;
//   camMat(2, 2) = 1.f;
//   camMat(0, 2) = px;
//   camMat(1, 2) = py;
  
    
   // Xu Ning
  Eigen::Matrix3f camMat;
  // camMat << fx, 0.0, px.
  //           0.0, fy, py,
  //           0.0, 0.0, 1.0;
  camMat(0,0) = fx;
  camMat(1,1) = fy;
  camMat(2,2) = 1.f;
  camMat(0,2) = px;
  camMat(1,2) = py; 
  camMat(0,1) = 0;
  camMat(1,0) = 0;
  camMat(2,0) = 0;
  camMat(2,1) = 0;
    
    
  // initialize hough space
  // H X W X N integer
  int* hough_space = (int*)malloc(sizeof(int) * height * width * num_classes);
  // Initialize all values to 0
  memset(hough_space, 0, height * width * num_classes);
  // N integer
  int* flags = (int*)malloc(sizeof(int) * num_classes);
  // Initialize all values in memory space to 0
  memset(flags, 0, num_classes);
  auto acc_label = v_label.accessor<int,1>();
  auto acc_vertex = v_vertex.accessor<float,1>();
  // for each pixel
  for (int x = 0; x < width; x++)
  {
    for (int y = 0; y < height; y++)
    {
      // here need to understand the value of label map
      int c = acc_label[labelmap+y * width + x]; // label map is one dimension array contains pixel wise image label map, map to class 1-13 etc..  
      if (c > 0) // this pixel is in this object class
      {
        flags[c] = 1; // this is a flag of whether there is this object in this image. 
        // read the predict center direction
        int offset = VERTEX_CHANNELS * c + VERTEX_CHANNELS * num_classes * (y * width + x); // Don't understand this 
        float u = acc_vertex[vertmap+offset];
        float v = acc_vertex[vertmap+offset + 1];
        float norm = sqrt(u * u + v * v); // u and v here are the delta_x and delta_y
        u /= norm;
        v /= norm;// (u,v) is the unit vector indicates center direction

        // voting
	    float delta = 1.0 / fabs(u);
        float cx = x;
        float cy = y;
        while(1)
        {
          cx += delta * u;
          cy += delta * v;
          int center_x = int(cx);
          int center_y = int(cy);
          if (center_x >= 0 && center_x < width && center_y >= 0 && center_y < height)
          {
            offset = c + num_classes * (center_y * width + center_x);
            hough_space[offset] += 1;
          }
          else
            break;
        }
      }
    }
  }
  // find the maximum in hough space
  for (int c = 1; c < num_classes; c++)
  {
    if (flags[c])
    {
      int max_vote = 0;
      int max_x, max_y;
      for (int x = 0; x < width; x++)
      {
        for (int y = 0; y < height; y++)
        {
          int offset = c + num_classes * (y * width + x);
          if (hough_space[offset] > max_vote)
          {
            max_vote = hough_space[offset];
            max_x = x;
            max_y = y;
          }
        }
      }
      if (max_vote < votingThreshold)
        continue;

      // center
      cv::Point2f center(max_x, max_y);
      int bb_width, bb_height;
      float bb_distance;
      
      compute_width_height(v_label, v_vertex, labelmap, vertmap, center, bb3Ds, camMat, inlierThreshold, height, width, c, num_classes, bb_width, bb_height, bb_distance);
      
      // construct output
      cv::Vec<float, 14> roi;
      roi(0) = batch; //batch number index 0 to batchsize -1
      roi(1) = c; //cls number index 1 to 13

      // bounding box
      float scale = 0.05;
      roi(2) = center.x - bb_width * (0.5 + scale);
      roi(3) = center.y - bb_height * (0.5 + scale);
      roi(4) = center.x + bb_width * (0.5 + scale);
      roi(5) = center.y + bb_height * (0.5 + scale);
      // score
      roi(6) = max_vote;

      // pose
      float rx = (center.x - px) / fx;
      float ry = (center.y - py) / fy;
      roi(7) = 1;
      roi(8) = 0;
      roi(9) = 0;
      roi(10) = 0;
      roi(11) = rx * bb_distance;
      roi(12) = ry * bb_distance;
      roi(13) = bb_distance;

      outputs.push_back(roi);
//     /////////////
//     // TODO 
//       if (is_train)
//       {
//         // add jittering rois
//         float x1 = roi(2);
//         float y1 = roi(3);
//         float x2 = roi(4);
//         float y2 = roi(5);
//         float ww = x2 - x1;
//         float hh = y2 - y1;

//         // (-1, -1)
//         roi(2) = x1 - 0.05 * ww;
//         roi(3) = y1 - 0.05 * hh;
//         roi(4) = roi(2) + ww;
//         roi(5) = roi(3) + hh;
//         outputs.push_back(roi);

//         // (+1, -1)
//         roi(2) = x1 + 0.05 * ww;
//         roi(3) = y1 - 0.05 * hh;
//         roi(4) = roi(2) + ww;
//         roi(5) = roi(3) + hh;
//         outputs.push_back(roi);

//         // (-1, +1)
//         roi(2) = x1 - 0.05 * ww;
//         roi(3) = y1 + 0.05 * hh;
//         roi(4) = roi(2) + ww;
//         roi(5) = roi(3) + hh;
//         outputs.push_back(roi);

//         // (+1, +1)
//         roi(2) = x1 + 0.05 * ww;
//         roi(3) = y1 + 0.05 * hh;
//         roi(4) = roi(2) + ww;
//         roi(5) = roi(3) + hh;
//         outputs.push_back(roi);

//         // (0, -1)
//         roi(2) = x1;
//         roi(3) = y1 - 0.05 * hh;
//         roi(4) = roi(2) + ww;
//         roi(5) = roi(3) + hh;
//         outputs.push_back(roi);

//         // (-1, 0)
//         roi(2) = x1 - 0.05 * ww;
//         roi(3) = y1;
//         roi(4) = roi(2) + ww;
//         roi(5) = roi(3) + hh;
//         outputs.push_back(roi);

//         // (0, +1)
//         roi(2) = x1;
//         roi(3) = y1 + 0.05 * hh;
//         roi(4) = roi(2) + ww;
//         roi(5) = roi(3) + hh;
//         outputs.push_back(roi);

//         // (+1, 0)
//         roi(2) = x1 + 0.05 * ww;
//         roi(3) = y1;
//         roi(4) = roi(2) + ww;
//         roi(5) = roi(3) + hh;
//         outputs.push_back(roi);
//       }
    }
  }
}


// get 3D bounding boxes
void getBb3Ds(const Tensor& extents, std::vector<std::vector<cv::Point3f>>& bb3Ds, int num_classes)
{
  // for each object
  auto acc_extents = extents.packed_accessor<float,1>();
  for (int i = 1; i < num_classes; i++)
  {
    cv::Vec<float, 3> extent;

    extent(0) = acc_extents[i * 3];
    extent(1) = acc_extents[i * 3 + 1];
    extent(2) = acc_extents[i * 3 + 2];
    bb3Ds.push_back(getBB3D(extent));
  }
}


inline std::vector<cv::Point3f> getBB3D(const cv::Vec<float, 3>& extent)
{
  std::vector<cv::Point3f> bb;  
  float xHalf = extent[0] * 0.5;
  float yHalf = extent[1] * 0.5;
  float zHalf = extent[2] * 0.5;

  bb.push_back(cv::Point3f(xHalf, yHalf, zHalf));
  bb.push_back(cv::Point3f(-xHalf, yHalf, zHalf));
  bb.push_back(cv::Point3f(xHalf, -yHalf, zHalf));
  bb.push_back(cv::Point3f(-xHalf, -yHalf, zHalf));
  
  bb.push_back(cv::Point3f(xHalf, yHalf, -zHalf));
  bb.push_back(cv::Point3f(-xHalf, yHalf, -zHalf));
  bb.push_back(cv::Point3f(xHalf, -yHalf, -zHalf));
  bb.push_back(cv::Point3f(-xHalf, -yHalf, -zHalf));

  return bb;
}


// inline cv::Rect getBB2D(int imageWidth, int imageHeight, const std::vector<cv::Point3f>& bb3D, const cv::Mat& camMat, const cv::Mat& rvec, const cv::Mat& tvec)
// {    
//   // project 3D bounding box vertices into the image
//   std::vector<cv::Point2f> bb2D;
//   cv::projectPoints(bb3D, rvec, tvec, camMat, cv::Mat(), bb2D);
    
//   // get min-max of projected vertices
//   int minX = imageWidth - 1;
//   int maxX = 0;
//   int minY = imageHeight - 1;
//   int maxY = 0;
    
//   for(unsigned j = 0; j < bb2D.size(); j++)
//   {
//     minX = std::min((float) minX, bb2D[j].x);
//     minY = std::min((float) minY, bb2D[j].y);
//     maxX = std::max((float) maxX, bb2D[j].x);
//     maxY = std::max((float) maxY, bb2D[j].y);
//   }
    
//   // clamp at image border
//   minX = clamp(minX, 0, imageWidth - 1);
//   maxX = clamp(maxX, 0, imageWidth - 1);
//   minY = clamp(minY, 0, imageHeight - 1);
//   maxY = clamp(maxY, 0, imageHeight - 1);
    
//   return cv::Rect(minX, minY, (maxX - minX + 1), (maxY - minY + 1));
// }

inline void compute_width_height(const Tensor& v_label, const Tensor& v_vertex, const int labelmap, const float vertmap, cv::Point2f center, std::vector<std::vector<cv::Point3f>> bb3Ds, Eigen::Matrix3f camMat, float inlierThreshold, int height, int width, int channel, int num_classes, int & bb_width, int & bb_height, float & bb_distance)
{
  float d = 0;
  int count = 0;

  // for each pixel
  std::vector<float> dx;
  std::vector<float> dy;
  auto acc_label = v_label.accessor<int,1>();
  auto acc_vertex = v_vertex.accessor<float,1>();
  for (int x = 0; x < width; x++)
  {
    for (int y = 0; y < height; y++)
    {
      if (acc_label[labelmap+y * width + x] == channel)
      {
        cv::Point2f point(x, y);
        int offset = VERTEX_CHANNELS * channel + VERTEX_CHANNELS * num_classes * (y * width + x);
        float u = acc_vertex[vertmap+offset];
        float v = acc_vertex[vertmap+offset + 1];
        float distance = exp(acc_vertex[vertmap+offset + 2]/1000.0);
        float norm = sqrt(u * u + v * v);
        u /= norm;
        v /= norm;
        cv::Point2f direction(u, v);

        // inlier check
        if(angle_distance(center, direction, point) > inlierThreshold)
        {
          dx.push_back(fabs(point.x - center.x));
          dy.push_back(fabs(point.y - center.y));
          d += distance;
          count++;
        }
      }
    }
  }
  bb_distance = d / count;
  // estimate a projection
//   cv::Mat tvec(3, 1, CV_64F);
//   cv::Mat rvec(3, 1, CV_64F);
//   for(int i = 0; i < 3; i++)
//   {
//     tvec.at<double>(i, 0) = 0.0;
//     rvec.at<double>(i, 0) = 0.0;
//   }  
//   tvec.at<double>(2, 0) = bb_distance;
// //   jp::cv_trans_t pose(rvec, tvec);

//   std::vector<cv::Point2f> bb2D;
// //   cv::projectPoints(bb3Ds[objID-1], pose.first, pose.second, camMat, cv::Mat(), bb2D);
//   cv::projectPoints(bb3Ds[channel-1], rvec, tvec,  camMat, cv::Mat(), bb2D);
    
  //Xu Ning
  Eigen::MatrixXf tvec(3,1);
  Eigen::MatrixXf rvec(3,1);
  for(int i = 0; i < 3; i++)
  {
    tvec(i,0) = 0.0;
    rvec(i,0) = 0.0;
  }    
  tvec(2, 0) = bb_distance;
  std::vector<cv::Point2f> bb2D;
  projectPoints(bb3Ds[channel-1], bb_distance, camMat, bb2D);
 
    
    // get min-max of projected vertices
  int minX = 1e8;
  int maxX = -1e8;
  int minY = 1e8;
  int maxY = -1e8;
  for(unsigned int i = 0; i < bb2D.size(); i++)
  {
    minX = std::min((float) minX, bb2D[i].x);
    minY = std::min((float) minY, bb2D[i].y);
    maxX = std::max((float) maxX, bb2D[i].x);
    maxY = std::max((float) maxY, bb2D[i].y);
  }
  cv::Rect bb = cv::Rect(0, 0, (maxX - minX + 1), (maxY - minY + 1));
  std::vector<float>::iterator it;
  it = std::remove_if(dx.begin(), dx.end(), std::bind2nd(std::greater<float>(), std::max(bb.width, bb.height) ));
  dx.erase(it, dx.end()); 

  it = std::remove_if(dy.begin(), dy.end(), std::bind2nd(std::greater<float>(), std::max(bb.width, bb.height) ));
  dy.erase(it, dy.end()); 
  std::sort(dx.begin(), dx.end());
  std::sort(dy.begin(), dy.end());
  int index1 = int(dx.size() * 0.95);
  int index2 = int(dy.size() * 0.95);
  if (dx.size() == 0 || dy.size() == 0){
     bb_width = 2;
     bb_height = 2;
  }else{
     bb_width = 2 * dx[index1];
     bb_height = 2 * dy[index2];
  }
}

void projectPoints(std::vector<cv::Point3f> bb3Ds, float& bb_distance, Eigen::MatrixXf camMat, std::vector<cv::Point2f>& bb2D){
    Eigen::MatrixXf extrinsic = Eigen::MatrixXf::Zero(3,4);
    Eigen::MatrixXf intrinsic(3,3);
    // extrinsic << 1,0,0,0,0,1,0,0,0,0,1,bb_distance;
    extrinsic(0,0) = 1;
    extrinsic(1,1) = 1;
    extrinsic(2,2) = 1;
    extrinsic(2,3) = bb_distance;
    intrinsic = camMat;
    // Eigen::Matrix2d mat;  
    // mat << 1, 2,  
            // 3, 4;  
    // Eigen::Vector2d u(-1,1), v(2,0);  
    // std::cout << "Here is mat*mat:\n" << mat*u << std::endl;  

    for (auto bb3D:bb3Ds){
        cv::Point2f pt(0,0);
        Eigen::Vector4f vecpt(bb3D.x, bb3D.y, bb3D.z, 1);
        // extrinsic*vecpt;
        Eigen::Vector3f pt_vec = intrinsic * extrinsic * vecpt;
        pt.x = pt_vec[0]/pt_vec[2];
        pt.y = pt_vec[1]/pt_vec[2];
        bb2D.push_back(pt);
    }
}

inline float angle_distance(cv::Point2f x, cv::Point2f n, cv::Point2f p)
{
    return n.dot(x - p) / (cv::norm(n) * cv::norm(x - p));
}

PYBIND11_MODULE(HoughVoting, m) {
  m.def("forward", &Forward, "HoughVoting forward");
  // m.def("backward", &Backward, "HoughVoting backward");
}


