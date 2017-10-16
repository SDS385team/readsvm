#define ARMA_64BIT_WORD 1

#include <RcppArmadillo.h>
#include <fstream>
#include <sstream>
#include <string>
#include <list>
#include <iterator>
#include <boost/algorithm/string/split.hpp> // Include for boost::split
#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::plugins(cpp11)]]


using namespace Rcpp;
using namespace arma;

//' @title Read SVM Light sparse data
//' @description Convenience function to read svm light data format
//' @param path : Path of file saved in .svm format
//' @param zero_indexing : integer indicating if stored data indicated columns indexing starting from 0 or 1
//' @param zero_one_response : teh default output for the response variable is 1, -1; set to true to change to 1, 0.
//' @param transpose : logical indicating if feature matrix should be returned transposed (see details).
//' @param nfeatures : optional non negative integer indicating the total number of features in the matrix, if the value is zero (by default)
//' the number of features is taken from the observed data. Use when the number of features is known and you want to run data in batches.
//' @return an object of class dgCMatrix from the Matrix package
//' @details Most sparse matrix libraries have better support for column-compressed sparse data format while algorithms 
//' require to read information by row. The default return_transpose = true to make this easier.
//' @export
// [[Rcpp::export]]
List read_svm(std::string path,
              const bool zero_indexing = true,
              const bool zero_one_response = false,
              const bool transpose = true,
              const std::size_t nfeatures = 0) {
  
  //  Define list for saving values of rows, columns and values
  std::vector<uword> row, col;
  std::vector<int> response;
  std::vector<double> raw_values;
  
  // Make conection to file
  std::ifstream read(path.c_str()); 
  std::string tmp_string;
  
  // Initializer counters for rows and columns
  int row_counter = 0, col_counter;

  // Should response be in 1,0 format?
  const uword base_col = zero_one_response ? 1 : 0;
  
  try {
    while (getline(read, tmp_string)) {
      std::vector<std::string> tmp_words;
      boost::split(tmp_words, tmp_string, boost::is_any_of(" :"), boost::token_compress_on);
      auto it = tmp_words.begin();
      response.push_back(stoi(*it++));
      while (it != tmp_words.end()) {
        col_counter = stod(*it++) - base_col;
        col.push_back(col_counter); 
        raw_values.push_back(stof(*it++));
        row.push_back(row_counter);
      }
      row_counter++;
    }
  }
  catch(std::exception const& e){
    cout << e.what() << endl;
  }
   
  // Create ingredients of sparse matrix
  arma::umat location(2, row.size());
  // arma::vec values(&raw_values[0], raw_values.size(), false, true); // copy from memory
  arma::mat values(raw_values);
  
  // How many columns are there ?
  col_counter = (nfeatures > 0) ? nfeatures : col.back();
  
  // Define field of sparse matrix depending on transposed or not
  int nrow, ncol;// [[Rcpp::export]]
  bool sort;
  if (transpose) {
    location.row(1) = conv_to<urowvec>::from(row);
    location.row(0) = conv_to<urowvec>::from(col);
    nrow = row_counter;
    ncol = col_counter;
    sort = false;
  } else {
    location.row(0) = conv_to<urowvec>::from(row);
    location.row(1) = conv_to<urowvec>::from(col);
    nrow = row_counter;
    ncol = col_counter;
    sort = true;
  }
  
  // Buld sparse matrix
  sp_mat features(location, values, sort);
  
  // Output
  return List::create(Named("response") = response,
                      Named("features") = features);
}
