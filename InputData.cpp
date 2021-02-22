#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include "InputData.hpp"

using namespace std;

void InputData::Read_booking()
{
  cout << "setInputData" << endl;
  string BookingFileName = "book/exp_height.csv";
  vector<vector<string>> vv;
  ifstream ifs(BookingFileName);
  for (string value; getline(ifs, value);)
  {
    vv.push_back(vector<string>());
    for (stringstream ss(value); getline(ss, value, ',');)
    {
      vv[vv.size() - 1].push_back(value);
    }
  }

  int lport_idx, dport_idx;
  int i;
  vector<string> lport_name;
  vector<string> dport_name;
  for (i = 0; i < vv[0].size(); i++)
  {
    if (vv[0][i] == "PORT_L")
    {
      lport_idx = i;
    }
    if (vv[0][i] == "PORT_D")
    {
      dport_idx = i;
    }
  }
  for (i = 1; i < vv.size(); i++)
  {
    if (vv[i][lport_idx] != "")
    {
      lport_name.push_back(vv[i][lport_idx]);
    }
    if (vv[i][dport_idx] != "")
    {
      dport_name.push_back(vv[i][dport_idx]);
    }
  }
  for (i = 0; i < lport_name.size(); i++)
  {
    this->L.push_back(i);
    this->T.push_back(i);
  }
  for (i = 0; i < dport_name.size(); i++)
  {
    this->D.push_back(i + this->L.size());
    this->T.push_back(i + this->L.size());
  }
  // for (i = 0; i < this->T.size(); i++)
  // {
  //   cout << this->T[i] << " ";
  // }
  // cout << endl;
  // for (i = 0; i < this->L.size(); i++)
  // {
  //   cout << this->L[i] << " ";
  // }
  // cout << endl;
  // for (i = 0; i < this->D.size(); i++)
  // {
  //   cout << this->D[i] << " ";
  // }
  // cout << endl;
  // T,L,Dまでok
}