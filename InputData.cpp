#include "InputData.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

void InputData::Read_booking() {
    cout << "setInputData" << endl;
    string BookingFileName = "book/exp.csv";
    vector<vector<string>> vv;
    ifstream ifs(BookingFileName);
    for (string value; getline(ifs, value);) {
        vv.push_back(vector<string>());
        for (stringstream ss(value); getline(ss, value, ',');) {
            vv[vv.size() - 1].push_back(value);
        }
    }

    int lport_idx, dport_idx;
    int unit_idx;
    int i, j;
    int tmp;
    vector<string> lport_name;
    vector<string> dport_name;
    cout << vv.size() << endl;
    // カラム名のインデックスを取得するforループ
    for (i = 0; i < vv[0].size(); i++) {
        if (vv[0][i] == "PORT_L") {
            lport_idx = i;
        }
        if (vv[0][i] == "PORT_D") {
            dport_idx = i;
        }
        if (vv[0][i] == "Units") {
            unit_idx = i;
        }
    }

    for (i = 1; i < vv.size(); i++) {
        if (vv[i][lport_idx] != "") {
            lport_name.push_back(vv[i][lport_idx]);
        }
        if (vv[i][dport_idx] != "") {
            dport_name.push_back(vv[i][dport_idx]);
        }
    }
    for (i = 0; i < lport_name.size(); i++) {
        this->L.push_back(i);
        this->T.push_back(i);
    }
    for (i = 0; i < dport_name.size(); i++) {
        this->D.push_back(i + this->L.size());
        this->T.push_back(i + this->L.size());
    }

    int cport_idx = vv[0].size() - 1;
    this->Check_port.push_back(lport_name.size() - 1);
    // TODO 複雑な航海ではcheck_portの実装が必要

    for (i = 0; i < vv.size(); i++) {
        for (j = 0; j < vv[i].size(); j++) {
            for (int k = 0; k < lport_name.size(); k++) {
                if (vv[i][j] == lport_name[k]) {
                    vv[i][j] = to_string(this->L[k]);
                }
            }
            for (int k = 0; k < dport_name.size(); k++) {
                if (vv[i][j] == dport_name[k]) {
                    vv[i][j] = to_string(this->D[k]);
                }
            }
        }
    }

    vector<int> divided_j;
    vector<vector<int>> divided_dic;
    vector<int> tmpVec;
    tmpVec.push_back(0);
    tmpVec.push_back(0);
    vector<vector<int>> divide_df;
    for (i = 1; i < vv.size(); i++) {
        tmp = stoi(vv[i][unit_idx]);
        int u_num_1, u_num_2;
        if ((500 < tmp) && (tmp <= 1000)) {
            if (tmp % 2 == 0) {
                u_num_1 = tmp / 2;
                u_num_2 = tmp / 2;
            } else {
                u_num_1 = (tmp / 2) + 1;
                u_num_2 = (tmp / 2);
            }
            divided_j.push_back(i);
            tmpVec[0] = i;
            tmpVec[1] = tmp;
            divided_dic.push_back(tmpVec);

            vector<string> divided;
            for (j = 0; j < vv[i].size(); j++) {
                divided.push_back(vv[i][j]);
            }
            divided[unit_idx] = to_string(u_num_1);
            vv.push_back(divided);
            divided[unit_idx] = to_string(u_num_2);
            vv.push_back(divided);
        }
    }
    for (i = divided_j.size() - 1; i >= 0; i--) {
        vv.erase(vv.begin() + divided_j[i]);
    }
    cout << vv.size() << endl;
    for (i = 0; i < vv.size(); i++) {
        cout << vv[i][0] << endl;
    }
}