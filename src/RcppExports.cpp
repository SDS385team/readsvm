// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// read_svm
List read_svm(std::string path, const bool zero_indexing, const bool zero_one_response, const bool transpose, const std::size_t nfeatures);
RcppExport SEXP _readsvm_read_svm(SEXP pathSEXP, SEXP zero_indexingSEXP, SEXP zero_one_responseSEXP, SEXP transposeSEXP, SEXP nfeaturesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type path(pathSEXP);
    Rcpp::traits::input_parameter< const bool >::type zero_indexing(zero_indexingSEXP);
    Rcpp::traits::input_parameter< const bool >::type zero_one_response(zero_one_responseSEXP);
    Rcpp::traits::input_parameter< const bool >::type transpose(transposeSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type nfeatures(nfeaturesSEXP);
    rcpp_result_gen = Rcpp::wrap(read_svm(path, zero_indexing, zero_one_response, transpose, nfeatures));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_readsvm_read_svm", (DL_FUNC) &_readsvm_read_svm, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_readsvm(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
