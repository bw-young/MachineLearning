#include <iostream>

#include "LogisticRegressionModel.hpp"

void test1D ( Matrix<double>& X, Matrix<double>& Y )
{
    X = Matrix<double> (20, 1);
    Y = Matrix<double> (20, 1);
    
    int i = 0; X(i,0) = 0.50; Y(i,0) = 0;
    ++i;       X(i,0) = 0.75; Y(i,0) = 0;
    ++i;       X(i,0) = 1.00; Y(i,0) = 0;
    ++i;       X(i,0) = 1.25; Y(i,0) = 0;
    ++i;       X(i,0) = 1.50; Y(i,0) = 0;
    ++i;       X(i,0) = 1.75; Y(i,0) = 0;
    ++i;       X(i,0) = 1.75; Y(i,0) = 1;
    ++i;       X(i,0) = 2.00; Y(i,0) = 0;
    ++i;       X(i,0) = 2.25; Y(i,0) = 1;
    ++i;       X(i,0) = 2.50; Y(i,0) = 0;
    ++i;       X(i,0) = 2.75; Y(i,0) = 1;
    ++i;       X(i,0) = 3.00; Y(i,0) = 0;
    ++i;       X(i,0) = 3.25; Y(i,0) = 1;
    ++i;       X(i,0) = 3.50; Y(i,0) = 0;
    ++i;       X(i,0) = 4.00; Y(i,0) = 1;
    ++i;       X(i,0) = 4.25; Y(i,0) = 1;
    ++i;       X(i,0) = 4.50; Y(i,0) = 1;
    ++i;       X(i,0) = 4.75; Y(i,0) = 1;
    ++i;       X(i,0) = 5.00; Y(i,0) = 1;
    ++i;       X(i,0) = 5.50; Y(i,0) = 1;
}

void test2D ( Matrix<double>& X, Matrix<double>& Y )
{
    X = Matrix<double> (20, 2);
    Y = Matrix<double> (20, 1);
    
    int i = 0; X(i,0) = 0.50; X(i,1) = 1; Y(i,0) = 0;
    ++i;       X(i,0) = 0.75; X(i,1) = 1; Y(i,0) = 0;
    ++i;       X(i,0) = 1.00; X(i,1) = 0; Y(i,0) = 0;
    ++i;       X(i,0) = 1.25; X(i,1) = 1; Y(i,0) = 0;
    ++i;       X(i,0) = 1.50; X(i,1) = 1; Y(i,0) = 0;
    ++i;       X(i,0) = 1.75; X(i,1) = 0; Y(i,0) = 0;
    ++i;       X(i,0) = 1.75; X(i,1) = 0; Y(i,0) = 1;
    ++i;       X(i,0) = 2.00; X(i,1) = 1; Y(i,0) = 0;
    ++i;       X(i,0) = 2.25; X(i,1) = 1; Y(i,0) = 1;
    ++i;       X(i,0) = 2.50; X(i,1) = 0; Y(i,0) = 0;
    ++i;       X(i,0) = 2.75; X(i,1) = 0; Y(i,0) = 1;
    ++i;       X(i,0) = 3.00; X(i,1) = 0; Y(i,0) = 0;
    ++i;       X(i,0) = 3.25; X(i,1) = 0; Y(i,0) = 1;
    ++i;       X(i,0) = 3.50; X(i,1) = 1; Y(i,0) = 0;
    ++i;       X(i,0) = 4.00; X(i,1) = 0; Y(i,0) = 1;
    ++i;       X(i,0) = 4.25; X(i,1) = 0; Y(i,0) = 1;
    ++i;       X(i,0) = 4.50; X(i,1) = 0; Y(i,0) = 1;
    ++i;       X(i,0) = 4.75; X(i,1) = 1; Y(i,0) = 1;
    ++i;       X(i,0) = 5.00; X(i,1) = 0; Y(i,0) = 1;
    ++i;       X(i,0) = 5.50; X(i,1) = 0; Y(i,0) = 1;
}

int main ()
{
    Matrix<double> X, Y;
    //test1D(X, Y);
    test2D(X, Y);
    
    std::cout << "begin\n";
    LogisticRegressionModel model;
    model.n_iter = 50000000;
    model.print_output = 10000;
    std::cout << "training...\n";
    model.train(X, Y);
    std::cout << "training complete after " << model.iteration
              << " iterations\n";
    
    std::cout << "model weights "; printMatrix(model.weights);
    std::cout << "  " << model.intercept << "\n";
    std::cout << "model cost " << model.cost
              << " (" << model.dCost << ")"
              << "\n";
    
    for ( int i = 0; i < X.nrows(); ++i ) {
        Matrix<double> Yp = model.predict(X.sub(i,i+1,0,2));
        Matrix<double> Ypp = model.predict(X.sub(i,i+1,0,2), true);
        std::cout << X(i,0);
        for ( int j = 1; j < X.ncols(); ++j )
            std::cout << " " << X(i,j);
        std::cout << " " << Y[i]
                  << " " << Yp[0] << " " << Ypp[0]
                  << " " << (fabs(Ypp[0] - Y[i]) < 0.0000001 ? "" : "***")
                  << "\n";
    }
    
    std::cout << "done\n";
    
    return 0;
}