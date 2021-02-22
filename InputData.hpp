#pragma once

using namespace std;
#include <string>
#include <vector>

class InputData
{
private:
    vector<int> T;
    vector<int> L;
    vector<int> D;
    vector<int> Check_port;

public:
    void Read_booking();
};