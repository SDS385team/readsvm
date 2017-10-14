#define ARMA_64BIT_WORD 1
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::plugins(cpp11)]]

#include <RcppArmadillo.h>
#include <fstream>
#include <sstream>
#include <string>
#include <list>
#include <iterator>
#include <boost/algorithm/string/split.hpp> // Include for boost::split
#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of
  
using namespace Rcpp;
using namespace arma;

//' @title Read SVM Light into Sparse
//' @description Convenience function to read svm light data format
//' @param path File : to data in .svm format
//' @param min_col_index : integer indicating if stored data indicated columns indexing starting from 0 or 1
//' @param return_transpose : logical indicating if feature matrix should be returned transposed (see details)
//' @return an object of class dgCMatrix from the Matrix package 
//' @details Most sparse matrix libraries have better support for column-compressed sparse data format. So it is recommended to use return_transpose = true for efficiency
//' @export
// [[Rcpp::export]] 
List read_sparse_svm(std::string path, int min_col_index = 1, bool return_tranpose = false) {
  std::vector<int> row, col;
  std::vector<int> response;
  std::vector<double> raw_values;
 
  std::ifstream read(path.c_str());
  std::string tmp_string;
  
  int row_counter = 0;

  while (getline(read, tmp_string)) {
    std::vector<std::string> tmp_words;
    boost::split(tmp_words, tmp_string, boost::is_any_of(" :"), boost::token_compress_on);
    auto it = tmp_words.begin();
    response.push_back(stoi(*it++));
    while (it != tmp_words.end()) {
      col.push_back(stod(*it++) - min_col_index); 
      raw_values.push_back(stof(*it++));
      row.push_back(row_counter);
    }
    row_counter++;
  }
  
  arma::umat location(2, row.size());
  arma::mat values(raw_values);
  
  bool sort;
  if (!return_tranpose) {
    location.row(1) = conv_to<urowvec>::from(row);
    location.row(0) = conv_to<urowvec>::from(col); 
    sort = false;
  } else {
    location.row(0) = conv_to<urowvec>::from(row);
    location.row(1) = conv_to<urowvec>::from(col); 
    sort = true;
  }
  
  arma::sp_mat features(location, values, sort);
  
  return List::create(Named("response") = response,
                      Named("features") = features);
}

