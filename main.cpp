#include <iostream>

#include "InputData.hpp"
#include <gurobi_c++.h>

using namespace std;

int main()
{
    cout << "test" << endl;
    InputData inputData;
    inputData.Read_booking();
    try
    {

        GRBEnv env = GRBEnv(true);
        env.set("LogFile", "mip1.log");
        env.start();
    }
    catch (GRBException e)
    {
        cout << "Error during optimization" << endl;
        cout << e.getMessage() << endl;
    }
}