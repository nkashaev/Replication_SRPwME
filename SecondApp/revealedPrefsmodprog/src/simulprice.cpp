////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////  Simulation functions //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*  
Copyright 2014 Julien Boelaert.

This file is part of revealedPrefs.

revealedPrefs is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

revealedPrefs is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with revealedPrefs.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <RcppArmadillo.h>
#include <vector>
using namespace Rcpp;

////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

// Recursive depth-first GARP 
// (used by SimAxiom, defined in garp.cpp)
std::vector<unsigned> RecGarp(unsigned cur_obs, std::vector<bool> *tabu, 
                              unsigned *n_tabu, 
                              std::vector<unsigned> *hist_tabu,
                              std::vector<unsigned> ascendence, 
                              std::vector<bool> strict_asc, 
                              arma::mat *x, arma::mat *p, 
                              double afriat_par);
                              
// Recursive depth-first SARP 
// (used by SimAxiom, defined in sarp.cpp)
std::vector<unsigned> RecSarp(unsigned cur_obs, std::vector<bool> * tabu, 
                              unsigned *n_tabu,
                              std::vector<unsigned> ascendence, 
                              arma::mat *x, arma::mat *p, 
                              double afriat_par);


////////////////////////////////////////////////////////////////////////////////
// Generate simulated GARP-consistent data
RcppExport SEXP SimAxiomPrice(SEXP nobs, SEXP ngoods, SEXP afriat, SEXP maxit, 
                         SEXP pmin, SEXP pmax, SEXP qmin, SEXP qmax, 
                         SEXP axiom, SEXP p) { try {
  // import arguments
  double afriat_par= as<double>(afriat);
  //change
  double p_min= as<double>(pmin);
  double p_max= as<double>(pmax);
  double q_min= as<double>(qmin);
  double q_max= as<double>(qmax);
  unsigned n_obs= as<unsigned>(nobs);
  unsigned n_goods= as<unsigned>(ngoods);
  unsigned max_it= as<unsigned>(maxit);  
  CharacterVector r_axiom(axiom);
  char *the_axiom= r_axiom[0];
  //change
  NumericMatrix r_p(p);
  //unsigned n_obs= r_p.nrow();
  arma::mat pmat(r_p.begin(), r_p.nrow(), r_p.ncol());
  
  
  arma::mat mat_q= q_min + (q_max - q_min) * arma::randu(1, n_goods);
  //arma::mat mat_p= p_min + (p_max - p_min) * arma::randu(1, n_goods);
  //change
  //passing row_zero
  arma::mat mat_p= pmat.row(0);
  arma::mat cur_mat_q(mat_q);
  arma::mat cur_mat_p(mat_p);
    
  bool b_violation;
  unsigned cur_length= 1;
  unsigned cur_it= 0;
  // Variables for depth-first search (GARP or SARP)
  unsigned n_tabu= 0;
  std::vector<bool> tabu;
  std::vector<unsigned> hist_tabu; // history of tabu obs, ie. utility ordering
  std::vector<unsigned> df_search;
  while ((cur_length < n_obs) & (cur_it < max_it)) {
    b_violation= false;
    cur_it++;
    
    // Generate a random candidate observation
    cur_mat_q.insert_rows(cur_length, (q_max - q_min) * arma::randu(1, n_goods));
    //cur_mat_p.insert_rows(0, (p_max - p_min) * arma::randu(1, n_goods));
	//change
	cur_mat_p.insert_rows(cur_length, pmat.row(cur_length));
    
    if (strcmp(the_axiom, "WARP")==0) {
      // WARP : test new observation against the previous ones
      for (unsigned i_row= 1; i_row < cur_length + 1; i_row++) {
        if (afriat_par * arma::dot(cur_mat_p.row(0), cur_mat_q.row(0)) >=
              arma::dot(cur_mat_p.row(0), cur_mat_q.row(i_row))) {
          if (afriat_par * arma::dot(cur_mat_p.row(i_row), cur_mat_q.row(i_row)) >=
                arma::dot(cur_mat_p.row(i_row), cur_mat_q.row(0))) {
            if (arma::accu(arma::abs(cur_mat_p.row(i_row) - cur_mat_q.row(0))) != 0)
            { // if p_i x_i >= p_i x_k AND p_k x_k >= p_k x_i AND x_i != x_k
              b_violation= true;
              break;
            }
          }
        }
      }
    } else {
      // GARP or SARP : launch recursive search from the generated observation
      tabu= std::vector<bool>(cur_mat_q.n_rows + 1, false);
      n_tabu= 0;
      hist_tabu.clear();
      df_search.clear();
      if (strcmp(the_axiom, "GARP")==0) {
        df_search= RecGarp(0, &tabu, &n_tabu, &hist_tabu,
                           std::vector<unsigned>(0), 
                           std::vector<bool>(0),
                           &cur_mat_q, &cur_mat_p,
                           afriat_par);
      } else {
        df_search= RecSarp(0, &tabu, &n_tabu,
                           std::vector<unsigned>(0), 
                           &cur_mat_q, &cur_mat_p,
                           afriat_par);
      }
      if (df_search.size())  b_violation= true;
    }
    
    if (b_violation) {
      // violation
      cur_mat_q.shed_row(cur_length);
      cur_mat_p.shed_row(cur_length);
    } else {
      mat_q= cur_mat_q;
      mat_p= cur_mat_p;
      cur_length++;
    }
  }
    
  return List::create(Named("x", wrap(mat_q)), 
                      Named("p", wrap(mat_p)),
                      Named("iter", wrap(cur_it)), 
                      Named("nobs", wrap(cur_length)));
} catch(std::exception &ex) {  
  forward_exception_to_r(ex);
} catch(...) { 
  ::Rf_error("c++ exception (unknown reason)"); 
}
  // return to avoid CRAN warning:
  return wrap("ok");
}
