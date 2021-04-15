/////////////////////////////////////////////////////////////////////
// Class and methods for developing and evaluating a logistic      //
// regression model.                                               //
//                                                                 //
// Source: Saishruthi Swaminathan (Logistic Regression--Detailed   //
//   Overview, https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc, //
//   posted Mar 15, 2018.                                          //
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
// -- HISTORY ---------------------------------------------------- //
// 04/08/2021 - Brennan Young                                      //
// - created.                                                      //
// 04/13/2021 - Brennan Young                                      //
// - significantly simplified and corrected.                       //
// 04/15/2021 - Brennan Young                                      //
// - there can be an occasional issue computing cost with zero or  //
//   near-zero inverse predictions. This is avoided by adding a    //
//   very small number to the inverse predictions.                 //
// - corrected an error where forcing binary outputs would go out  //
//   of bounds.
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
// -- TO-DO ------------------------------------------------------ //
// Might be less efficient, but enable each iteration to be called //
// independently, and remove all attempts to report during         //
// iterations (let the calling program handle that -- also permits //
// the calling program to terminate if convergence doesn't occur). //
/////////////////////////////////////////////////////////////////////

#ifndef YOUNG_LOGISTICREGRESSION_20210408
#define YOUNG_LOGISTICREGRESSION_20210408

#include <iostream>

#include "Matrix.hpp"

class LogisticRegressionModel {
private:
    // helpers
    Matrix<double> sigmoid(const Matrix<double>&) const;
    double costFunc (const Matrix<double>&, const Matrix<double>&,
        const Matrix<double>&) const;
public:
    double learn_rate;
    double cost_threshold;
    int n_iter;
    int print_output;
    
    Matrix<double> weights;
    double intercept;
    double cost;
    double dCost;
    int iteration;
    
    // constructors, destructor
    LogisticRegressionModel(double, double, int);
    LogisticRegressionModel(const LogisticRegressionModel&);
    ~LogisticRegressionModel();
    
    // operators
    LogisticRegressionModel& operator=(
        const LogisticRegressionModel&);
    
    // operations
    void train(const Matrix<double>&, const Matrix<double>&);
    Matrix<double> predict(const Matrix<double>&, bool) const;
    Matrix<double> predict(const Matrix<double>&) const;
}; // Logistic Model


// CONSTRUCTORS / DESTRUCTOR ////////////////////////////////////////

LogisticRegressionModel::LogisticRegressionModel (
    double lrate=0.0001, double threshold=0.0000001, int n=1000000 )
: learn_rate(lrate), cost_threshold(threshold), n_iter(n),
    print_output(0), intercept(0), cost(0), dCost(0), iteration(0)
{}

LogisticRegressionModel::LogisticRegressionModel (
    const LogisticRegressionModel& model )
: learn_rate(model.learn_rate), cost_threshold(model.cost_threshold),
    n_iter(model.n_iter), print_output(model.print_output),
    weights(model.weights), intercept(model.intercept),
    cost(model.cost), dCost(model.dCost), iteration(model.iteration)
{}

LogisticRegressionModel::~LogisticRegressionModel () {}


// OPERATORS ////////////////////////////////////////////////////////

LogisticRegressionModel& LogisticRegressionModel::operator= (
    const LogisticRegressionModel& model )
{
    if ( &model == this ) return *this;
    
    learn_rate = model.learn_rate;
    cost_threshold = model.cost_threshold;
    n_iter = model.n_iter;
    print_output = model.print_output;
    
    weights = model.weights;
    intercept = model.intercept;
    cost = model.cost;
    dCost = model.dCost;
    iteration = model.iteration;
    
    return *this;
}

// OPERATIONS ///////////////////////////////////////////////////////

// Activate with the sigmoid function.
// -- Arguments --
// Y : vector of values to transform.
Matrix<double> LogisticRegressionModel::sigmoid (
    const Matrix<double>& Y ) const
{
    return 1.0 / (1.0 + elemExp(-Y));
}

// Compute the cost of the current model.
// -- Arguments --
// X  : independent variable matrix.
// Y  : training outcome vector.
// Yp : predicted outcome vector.
double LogisticRegressionModel::costFunc (
    const Matrix<double>& X, const Matrix<double>& Y,
    const Matrix<double>& Yp ) const
{
    Matrix<double> Yt = Y.transpose();
    Matrix<double> Ypt = Yp.transpose();
    return (-1.0 / X.ncols())
        * sum((Yt * elemLog(Ypt))
        + ((1.0 - Yt) * (elemLog(1.0 - Ypt + 0.0000001))));
}

// Train the logistic regression model.
// -- Arguments --
// X : independent variable training samples, with a column for each
//     predictor variable and a row for each sample.
// Y : dependent variable training samples.
void LogisticRegressionModel::train (
    const Matrix<double>& X, const Matrix<double>& Y )
{
    // initialize coefficients
    weights = randMatrix(1, X.ncols());
    intercept = 0.0;
    
    // perform the regression
    Matrix<double> Xt = X.transpose();
    Matrix<double> dw;
    double db;
    
    double prevCost = 0;
    dCost = cost_threshold + 1;
    
    for ( iteration = 0;
            iteration < n_iter && dCost > cost_threshold;
            ++iteration ) {
        // predict
        Matrix<double> Y_predicted = sigmoid(
            (weights * Xt) + intercept);
        
        // compute cost
        cost = costFunc(X, Y, Y_predicted);
        dCost = fabs(cost - prevCost);
        prevCost = cost;
        
        // compute gradients
        Matrix<double> gradient = Y_predicted - Y.transpose();
            dw = (1.0 / X.nrows())
                * (Xt * gradient.transpose());
            db = (1.0 / X.nrows()) * sum(gradient);
        
        // update weight
        weights = weights - (learn_rate * dw.transpose());
        intercept = intercept - (learn_rate * db);
        
        if ( print_output > 0 && iteration % print_output == 0 ) {
            std::cout << "epoch " << iteration
                      << " cost " << cost << " :"
                      << " " << intercept;
            for ( int j = 0; j < weights.size(); ++j )
                std::cout << " + " << weights[j] << "x" << (j+1);
            std::cout << "\n";
        }
    }
}

// Predict the model outcome.
// -- Arguments --
// X : independent variable matrix, with a column for each predictor
//     and a row for each sample.
// b : (default=false) true to force all outputs to 0 or 1; false
//     for real outputs in the interval [0,1].
// -- Returns --
// Solution vector.
Matrix<double> LogisticRegressionModel::predict (
    const Matrix<double>& X, bool b ) const
{
    Matrix<double> Y = sigmoid(
        (weights * X.transpose()) + intercept);
    for ( int i = 0; b && i < Y.size(); ++i )
        Y[i] = Y[i] < 0.5 ? 0.0 : 1.0;
    return Y;
}
Matrix<double> LogisticRegressionModel::predict (
    const Matrix<double>& X ) const
{
    return predict(X, false);
}


#endif // YOUNG_LOGISTICREGRESSION_20210408