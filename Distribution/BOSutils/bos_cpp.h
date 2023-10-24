#pragma once

#include <numeric>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;


#include <vector>
// 
// template<typename T>
// class Cube {
// public:
//   Cube(size_t n1, size_t n2, size_t n3)
//     : data(n1, std::vector<std::vector<T>>(n2, std::vector<T>(n3))) {}
//   
//   std::vector<std::vector<T>>& operator[](size_t index) {
//     return data[index];
//   }
//   
//   const std::vector<std::vector<T>>& operator[](size_t index) const {
//     return data[index];
//   }
//   
// private:
//   std::vector<std::vector<std::vector<T>>> data;
// };

struct Cube {
  std::vector<std::vector<std::vector<double>>> data;
  size_t n1, n2, n3;
  
  Cube(size_t n1, size_t n2, size_t n3)
    : data(n1, std::vector<std::vector<double>>(n2, std::vector<double>(n3))),
      n1(n1), n2(n2), n3(n3) {}
  
  std::vector<std::vector<double>>& operator[](size_t index) {
    return data[index];
  }
  
  const std::vector<std::vector<double>>& operator[](size_t index) const {
    return data[index];
  }
};




Cube gettabpej(
  int& _m
) {
  Cube result(_m, _m, _m);
  for (size_t i = 0; i < _m; ++i) {
    for (size_t j = 0; j < _m; ++j) {
      for (size_t k = 0; k < _m; ++k) {
        result[i][j][k] = 1.0;
      }
    }
  }
  
  if (_m == 2) {
    for (size_t i = 0; i < _m; ++i) {
      for (size_t j = 0; j < _m; ++j) {
        for (size_t k = 0; k < _m; ++k) {
          result[i][j][k] /= 2.0;
        }
      }
    }
    result[0][1][1] = -0.5;
    result[1][0][1] = -0.5;
  }
  
  if (_m == 3) {
    for (size_t i = 0; i < _m; ++i) {
      for (size_t j = 0; j < _m; ++j) {
        for (size_t k = 0; k < _m; ++k) {
          result[i][j][k] /= 3.0;
        }
      }
    }
    
    result[1] = {
      { 11.0 / 18.0, -5.0 / 18.0, -8.0 / 18.0 },
      { -1.0 / 6.0, 10.0 / 18.0, -1.0 / 6.0 },
      { -8.0 / 18.0, -5.0 / 18.0, 11.0 / 18.0 }
    };
    
    result[2] = {
      { 1.0 / 18.0, -1.0 / 18.0, 1.0 / 9.0 },
      { -1.0 / 6.0, 1.0 / 9.0, -1.0 / 6.0 },
      { 1.0 / 9.0, -1.0 / 18.0, 1.0 / 18.0 }
    };
  }
  
  if (_m == 4) {
    for (size_t i = 0; i < _m; ++i) {
      for (size_t j = 0; j < _m; ++j) {
        for (size_t k = 0; k < _m; ++k) {
          result[i][j][k] /= 4.0;
        }
     }
   }
    
    result[1] = {
      { 61.0 / 96.0, -5.0 / 32.0, -5.0 / 16.0, -19.0 / 48.0},
      { -1.0 / 48.0, 55.0 / 96.0, -5.0 / 48.0, -7.0 / 32.0},
      { -7.0 / 32.0, -5.0 / 48.0, 55.0 / 96.0, -1.0 / 48.0},
      { -19.0 / 48.0, -5.0 / 16.0, -5.0 / 32.0, 61.0 / 96.0}
   };
    
    result[2] = {
      { 1.0 / 9.0, -13.0 / 144.0, 1.0 / 18.0, 1.0 / 6.0 },
      { -5.0 / 24.0, 1.0 / 6.0, -19.0 / 144.0, -5.0 / 72.0},
      { -5.0 / 72.0, -19.0 / 144.0, 1.0 / 6.0, -5.0 / 24.0},
      { 1.0 / 6.0, 1.0 / 18.0, -13.0 / 144.0, 1.0 / 9.0}
    };
    
    result[3] = {
      { 1.0 / 288.0, -1.0 / 288.0, 1.0 / 144.0, -1.0 / 48.0},
      { -1.0 / 48.0, 1.0 / 96.0, -1.0 / 72.0, 11.0 / 288.0},
      { 11.0 / 288.0, -1.0 / 72.0, 1.0 / 96.0, -1.0 / 48.0},
      { -1.0 / 48.0, 1.0 / 144.0, -1.0 / 288.0, 1.0 / 288.0}
   };
  }
  
  if (_m == 5) {
    for (size_t i = 0; i < _m; ++i) {
      for (size_t j = 0; j < _m; ++j) {
        for (size_t k = 0; k < _m; ++k) {
          result[i][j][k] /= 5.0;
        }
      }
    }
    
    result[1] = {
      { 379.0 / 600.0, -49.0 / 600.0, -17.0 / 75.0, -23.0 / 75.0, -107.0 / 300.0},
      { 11.0 / 200.0, 17.0 / 30.0, -11.0 / 200.0, -33.0 / 200.0, -137.0 / 600.0},
      { -61.0 / 600.0, -1.0 / 75.0, 169.0 / 300.0, -1.0 / 75.0, -61.0 / 600.0},
      { -137.0 / 600.0, -33.0 / 200.0, -11.0 / 200.0, 17.0 / 30.0, 11.0 / 200.0},
      { -107.0 / 300.0, -23.0 / 75.0, -17.0 / 75.0, -49.0 / 600.0, 379.0 / 600.0}
    };
    
    result[2] = {
      { 127.0 / 800.0, -263.0 / 2400.0, 17.0 / 1200.0, 47.0 / 400.0, 59.0 / 300.0},
      { -761.0 / 3600.0, 379.0 / 1800.0, -859.0 / 7200.0, -457.0 / 7200.0, -19.0 / 3600.0},
      { -111.0 / 800.0, -31.0 / 200.0, 757.0 / 3600.0, -31.0 / 200.0, -111.0 / 800.0},
      { -19.0 / 3600.0, -457.0 / 7200.0, -859.0 / 7200.0, 379.0 / 1800.0, -761.0 / 3600.0},
      { 59.0 / 300.0, 47.0 / 400.0, 17.0 / 1200.0, -263.0 / 2400.0, 127.0 / 800.0}
    };
    
    result[3] = {
      { 17.0 / 1800.0, -31.0 / 3600.0, 11.0 / 900.0, -1.0 / 100.0, -13.0 / 300.0},
      { -19.0 / 450.0, 1.0 / 45.0, -1.0 / 40.0, 2.0 / 75.0, 73.0 / 1800.0},
      { 8.0 / 225.0, -109.0 / 3600.0, 23.0 / 900.0, -109.0 / 3600.0, 8.0 / 225.0},
      { 73.0 / 1800.0, 2.0 / 75.0, -1.0 / 40.0, 1.0 / 45.0, -19.0 / 450.0},
      { -13.0 / 300.0, -1.0 / 100.0, 11.0 / 900.0, -31.0 / 3600.0, 17.0 / 1800.0}
    };
    
    result[4] = {
      { 1.0 / 7200.0, -1.0 / 7200.0, 1.0 / 3600.0, -1.0 / 1200.0, 1.0 / 300.0},
      { -1.0 / 720.0, 1.0 / 1800.0, -1.0 / 1440.0, 13.0 / 7200.0, -1.0 / 144.0},
      { 7.0 / 1440.0, -1.0 / 720.0, 1.0 / 1200.0, -1.0 / 720.0, 7.0 / 1440.0},
      { -1.0 / 144.0, 13.0 / 7200.0, -1.0 / 1440.0, 1.0 / 1800.0, -1.0 / 720.0},
      { 1.0 / 300.0, -1.0 / 1200.0, 1.0 / 3600.0, -1.0 / 7200.0, 1.0 / 7200.0}
    };
  }
  
  if (_m == 6) {
    for (size_t i = 0; i < _m; ++i) {
      for (size_t j = 0; j < _m; ++j) {
        for (size_t k = 0; k < _m; ++k) {
          result[i][j][k] /= 6.0;
        }
      }
    }
    
    result[1] = {
      { 667.0 / 1080.0, -7.0 / 216.0, -361.0 / 2160.0, -35.0 / 144.0, -7.0 / 24.0, -13.0 / 40.0},
      { 71.0 / 720.0, 397.0 / 720.0, -7.0 / 360.0, -133.0 / 1080.0, -133.0 / 720.0, -9.0 / 40.0},
      { -1.0 / 30.0, 7.0 / 180.0, 1183.0 / 2160.0, 11.0 / 2160.0, -11.0 / 135.0, -287.0 / 2160.0},
      { -287.0 / 2160.0, -11.0 / 135.0, 11.0 / 2160.0, 1183.0 / 2160.0, 7.0 / 180.0, -1.0 / 30.0},
      { -9.0 / 40.0, -133.0 / 720.0, -133.0 / 1080.0, -7.0 / 360.0, 397.0 / 720.0, 71.0 / 720.0},
      { -13.0 / 40.0, -7.0 / 24.0, -35.0 / 144.0, -361.0 / 2160.0, -7.0 / 216.0, 667.0 / 1080.0}
    };
    
    result[2] = {
      { 51413.0 / 259200.0, -30983.0 / 259200.0, -2011.0 / 129600.0, 1129.0 / 14400.0, 1663.0 / 10800.0, 461.0 / 2160.0},
      { -52139.0 / 259200.0, 63773.0 / 259200.0, -7231.0 / 64800.0, -3389.0 / 51840.0, -113.0 / 10800.0, 1607.0 / 43200.0},
      { -42767.0 / 259200.0, -4049.0 / 25920.0, 31543.0 / 129600.0, -33517.0 / 259200.0, -295.0 / 2592.0, -21469.0 / 259200.0},
      { -21469.0 / 259200.0, -295.0 / 2592.0, -33517.0 / 259200.0, 31543.0 / 129600.0, -4049.0 / 25920.0, -42767.0 / 259200.0},
      { 1607.0 / 43200.0, -113.0 / 10800.0, -3389.0 / 51840.0, -7231.0 / 64800.0, 63773.0 / 259200.0, -52139.0 / 259200.0},
      { 461.0 / 2160.0, 1663.0 / 10800.0, 1129.0 / 14400.0, -2011.0 / 129600.0, -30983.0 / 259200.0, 51413.0 / 259200.0}
    };
    
    result[3] = {
      { 487.0 / 28800.0, -137.0 / 9600.0, 73.0 / 4800.0, -19.0 / 43200.0, -331.0 / 10800.0, -137.0 / 2160.0},
      { -15617.0 / 259200.0, 8899.0 / 259200.0, -4379.0 / 129600.0, 949.0 / 51840.0, 269.0 / 8100.0, 4103.0 / 129600.0},
      { 5771.0 / 259200.0, -1481.0 / 32400.0, 23.0 / 576.0, -2033.0 / 51840.0, 187.0 / 8100.0, 13697.0 / 259200.0},
      { 13697.0 / 259200.0, 187.0 / 8100.0, -2033.0 / 51840.0, 23.0 / 576.0, -1481.0 / 32400.0, 5771.0 / 259200.0},
      { 4103.0 / 129600.0, 269.0 / 8100.0, 949.0 / 51840.0, -4379.0 / 129600.0, 8899.0 / 259200.0, -15617.0 / 259200.0},
      { -137.0 / 2160.0, -331.0 / 10800.0, -19.0 / 43200.0, 73.0 / 4800.0, -137.0 / 9600.0, 487.0 / 28800}
   };
    
    result[4] = {
      { 41.0 / 86400.0, -13.0 / 28800.0, 11.0 / 14400.0, -67.0 / 43200.0, 17.0 / 10800.0, 19.0 / 2160.0},
      { -989.0 / 259200.0, 403.0 / 259200.0, -59.0 / 32400.0, 181.0 / 51840.0, -29.0 / 6480.0, -1501.0 / 129600.0},
      { 2351.0 / 259200.0, -461.0 / 129600.0, 11.0 / 4800.0, -823.0 / 259200.0, 347.0 / 64800.0, -763.0 / 259200.0},
      { -763.0 / 259200.0, 347.0 / 64800.0, -823.0 / 259200.0, 11.0 / 4800.0, -461.0 / 129600.0, 2351.0 / 259200.0},
      { -1501.0 / 129600.0, -29.0 / 6480.0, 181.0 / 51840.0, -59.0 / 32400.0, 403.0 / 259200.0, -989.0 / 259200.0},
      { 19.0 / 2160.0, 17.0 / 10800.0, -67.0 / 43200.0, 11.0 / 14400.0, -13.0 / 28800.0, 41.0 / 86400}
    };
    result[5] = {
      { 1.0 / 259200.0, -1.0 / 259200.0, 1.0 / 129600.0, -1.0 / 43200.0, 1.0 / 10800.0, -1.0 / 2160.0},
      { -1.0 / 17280.0, 1.0 / 51840.0, -1.0 / 43200.0, 1.0 / 17280.0, -7.0 / 32400.0, 137.0 / 129600.0},
      { 17.0 / 51840.0, -1.0 / 12960.0, 1.0 / 25920.0, -1.0 / 17280.0, 1.0 / 5400.0, -1.0 / 1152.0},
      { -1.0 / 1152.0, 1.0 / 5400.0, -1.0 / 17280.0, 1.0 / 25920.0, -1.0 / 12960.0, 17.0 / 51840.0},
      { 137.0 / 129600.0, -7.0 / 32400.0, 1.0 / 17280.0, -1.0 / 43200.0, 1.0 / 51840.0, -1.0 / 17280.0},
      { -1.0 / 2160.0, 1.0 / 10800.0, -1.0 / 43200.0, 1.0 / 129600.0, -1.0 / 259200.0, 1.0 / 259200.0}
    };
}
  
  
  if (_m == 7) {
    for (size_t i = 0; i < _m; ++i) {
      for (size_t j = 0; j < _m; ++j) {
        for (size_t k = 0; k < _m; ++k) {
          result[i][j][k] /= 7.0;
        }
      }
    }
    
    result[1] = {
      { 529.0 / 882.0, 4.0 / 2205.0, -437.0 / 3528.0, -1151.0 / 5880.0, -713.0 / 2940.0, -809.0 / 2940.0, -293.0 / 980.0 },
      { 221.0 / 1764.0, 2351.0 / 4410.0, 113.0 / 17640.0, -107.0 / 1176.0, -3.0 / 20.0, -557.0 / 2940.0, -213.0 / 980.0 },
      { 29.0 / 2940.0, 209.0 / 2940.0, 1166.0 / 2205.0, 193.0 / 8820.0, -215.0 / 3528.0, -281.0 / 2520.0, -641.0 / 4410.0 },
      { -323.0 / 4410.0, -527.0 / 17640.0, 743.0 / 17640.0, 1168.0 / 2205.0, 743.0 / 17640.0, -527.0 / 17640.0, -323.0 / 4410.0 },
      { -641.0 / 4410.0, -281.0 / 2520.0, -215.0 / 3528.0, 193.0 / 8820.0, 1166.0 / 2205.0, 209.0 / 2940.0, 29.0 / 2940.0 },
      { -213.0 / 980.0, -557.0 / 2940.0, -3.0 / 20.0, -107.0 / 1176.0, 113.0 / 17640.0, 2351.0 / 4410.0, 221.0 / 1764.0 },
      { -293.0 / 980.0, -809.0 / 2940.0, -713.0 / 2940.0, -1151.0 / 5880.0, -437.0 / 3528.0, 4.0 / 2205.0, 529.0 / 882.0}
    };
    
    result[2] = {
      { 163103.0 / 705600.0, -262159.0 / 2116800.0, -38933.0 / 1058400.0, 16901.0 / 352800.0, 5237.0 / 44100.0, 311.0 / 1764.0, 3929.0 / 17640.0 },
      { -65759.0 / 352800.0, 72697.0 / 264600.0, -222569.0 / 2116800.0, -48341.0 / 705600.0, -6523.0 / 352800.0, 3223.0 / 117600.0, 2593.0 / 39200.0 },
      { -9139.0 / 52920.0, -314971.0 / 2116800.0, 63683.0 / 235200.0, -239119.0 / 2116800.0, -4387.0 / 43200.0, -151427.0 / 2116800.0, -87853.0 / 2116800.0 },
      { -63211.0 / 529200.0, -284233.0 / 2116800.0, -269987.0 / 2116800.0, 17671.0 / 66150.0, -269987.0 / 2116800.0, -284233.0 / 2116800.0, -63211.0 / 529200.0 },
      { -87853.0 / 2116800.0, -151427.0 / 2116800.0, -4387.0 / 43200.0, -239119.0 / 2116800.0, 63683.0 / 235200.0, -314971.0 / 2116800.0, -9139.0 / 52920.0 },
      { 2593.0 / 39200.0, 3223.0 / 117600.0, -6523.0 / 352800.0, -48341.0 / 705600.0, -222569.0 / 2116800.0, 72697.0 / 264600.0, -65759.0 / 352800.0 },
      { 3929.0 / 17640.0, 311.0 / 1764.0, 5237.0 / 44100.0, 16901.0 / 352800.0, -38933.0 / 1058400.0, -262159.0 / 2116800.0, 163103.0 / 705600.0}
    };
    
    result[3] = {
      {  319847.0 / 12700800.0, -50509.0 / 2540160.0, 104207.0 / 6350400.0, 991.0 / 141120.0, -289.0 / 15120.0, -5293.0 / 105840.0, -71.0 / 882.0 },
      { -37949.0 / 508032.0, 58843.0 / 1270080.0, -19211.0 / 470400.0, 49337.0 / 4233600.0, 59933.0 / 2116800.0, 58621.0 / 2116800.0, 10361.0 / 529200.0 },
      { 92443.0 / 12700800.0, -745111.0 / 12700800.0, 677017.0 / 12700800.0, -590339.0 / 12700800.0, 33293.0 / 2540160.0, 534701.0 / 12700800.0, 12829.0 / 235200.0 },
      { 123481.0 / 2540160.0, 17551.0 / 1411200.0, -216679.0 / 4233600.0, 176569.0 / 3175200.0, -216679.0 / 4233600.0, 17551.0 / 1411200.0, 123481.0 / 2540160.0 },
      { 12829.0 / 235200.0, 534701.0 / 12700800.0, 33293.0 / 2540160.0, -590339.0 / 12700800.0, 677017.0 / 12700800.0, -745111.0 / 12700800.0, 92443.0 / 12700800.0 },
      { 10361.0 / 529200.0, 58621.0 / 2116800.0, 59933.0 / 2116800.0, 49337.0 / 4233600.0, -19211.0 / 470400.0, 58843.0 / 1270080.0, -37949.0 / 508032.0 },
      { -71.0 / 882.0, -5293.0 / 105840.0, -289.0 / 15120.0, 991.0 / 141120.0, 104207.0 / 6350400.0, -50509.0 / 2540160.0, 319847.0 / 12700800.0}
    };
    
    result[4] = {
      { 1433.0 / 1411200.0, -29.0 / 31360.0, 319.0 / 235200.0, -17.0 / 8640.0, -17.0 / 105840.0, 661.0 / 105840.0, 3.0 / 196.0 },
      { -86999.0 / 12700800.0, 2627.0 / 907200.0, -40651.0 / 12700800.0, 61169.0 / 12700800.0, -14227.0 / 6350400.0, -58231.0 / 6350400.0, -1783.0 / 132300.0 },
      { 21431.0 / 1814400.0, -26429.0 / 4233600.0, 10547.0 / 2540160.0, -65033.0 / 12700800.0, 15503.0 / 2540160.0, -4559.0 / 4233600.0, -13289.0 / 1270080.0 },
      { 33743.0 / 12700800.0, 105073.0 / 12700800.0, -76331.0 / 12700800.0, 229.0 / 50400.0, -76331.0 / 12700800.0, 105073.0 / 12700800.0, 33743.0 / 12700800.0 },
      { -13289.0 / 1270080.0, -4559.0 / 4233600.0, 15503.0 / 2540160.0, -65033.0 / 12700800.0, 10547.0 / 2540160.0, -26429.0 / 4233600.0, 21431.0 / 1814400.0 },
      { -1783.0 / 132300.0, -58231.0 / 6350400.0, -14227.0 / 6350400.0, 61169.0 / 12700800.0, -40651.0 / 12700800.0, 2627.0 / 907200.0, -86999.0 / 12700800.0 },
      { 3.0 / 196.0, 661.0 / 105840.0, -17.0 / 105840.0, -17.0 / 8640.0, 319.0 / 235200.0, -29.0 / 31360.0, 1433.0 / 1411200.0}
    };
    result[5] = {
      { 67.0 / 4233600.0, -13.0 / 846720.0, 59.0 / 2116800.0, -29.0 / 423360.0, 19.0 / 105840.0, -23.0 / 105840.0, -13.0 / 8820.0 },
      { -2531.0 / 12700800.0, 17.0 / 254016.0, -991.0 / 12700800.0, 2141.0 / 12700800.0, -2767.0 / 6350400.0, 4097.0 / 6350400.0, 1259.0 / 529200.0 },
      { 2209.0 / 2540160.0, -1007.0 / 4233600.0, 1591.0 / 12700800.0, -2213.0 / 12700800.0, 1067.0 / 2540160.0, -383.0 / 470400.0, -193.0 / 907200.0 },
      { -17509.0 / 12700800.0, 1039.0 / 1814400.0, -607.0 / 2540160.0, 157.0 / 1058400.0, -607.0 / 2540160.0, 1039.0 / 1814400.0, -17509.0 / 12700800.0 },
      { -193.0 / 907200.0, -383.0 / 470400.0, 1067.0 / 2540160.0, -2213.0 / 12700800.0, 1591.0 / 12700800.0, -1007.0 / 4233600.0, 2209.0 / 2540160.0 },
      { 1259.0 / 529200.0, 4097.0 / 6350400.0, -2767.0 / 6350400.0, 2141.0 / 12700800.0, -991.0 / 12700800.0, 17.0 / 254016.0, -2531.0 / 12700800.0 },
      { -13.0 / 8820.0, -23.0 / 105840.0, 19.0 / 105840.0, -29.0 / 423360.0, 59.0 / 2116800.0, -13.0 / 846720.0, 67.0 / 4233600.0 }
    };
    result[6] = {
      { 1.0 / 12700800.0, -1.0 / 12700800.0, 1.0 / 6350400.0, -1.0 / 2116800.0, 1.0 / 529200.0, -1.0 / 105840.0, 1.0 / 17640.0 },
      { -1.0 / 604800.0, 1.0 / 2116800.0, -1.0 / 1814400.0, 17.0 / 12700800.0, -31.0 / 6350400.0, 149.0 / 6350400.0, -1.0 / 7200.0 },
      { 1.0 / 72576.0, -1.0 / 362880.0, 1.0 / 846720.0, -1.0 / 604800.0, 1.0 / 201600.0, -281.0 / 12700800.0, 29.0 / 226800.0 },
      { -1.0 / 17280.0, 19.0 / 1814400.0, -1.0 / 362880.0, 1.0 / 635040.0, -1.0 / 362880.0, 19.0 / 1814400.0, -1.0 / 17280.0 },
      { 29.0 / 226800.0, -281.0 / 12700800.0, 1.0 / 201600.0, -1.0 / 604800.0, 1.0 / 846720.0, -1.0 / 362880.0, 1.0 / 72576.0 },
      { -1.0 / 7200.0, 149.0 / 6350400.0, -31.0 / 6350400.0, 17.0 / 12700800.0, -1.0 / 1814400.0, 1.0 / 2116800.0, -1.0 / 604800.0 },
      { 1.0 / 17640.0, -1.0 / 105840.0, 1.0 / 529200.0, -1.0 / 2116800.0, 1.0 / 6350400.0, -1.0 / 12700800.0, 1.0 / 12700800.0}
    };
  }
  
  return result;
}


bool compare_vec(std::vector<int>& vec1, std::vector<int>& vec2) {
  if (vec1.size() != vec2.size()) {
    return false;
  }
  
  for (int i = 0; i < vec1.size(); ++i) {
    if (vec1[i] != vec2[i]) {
      return false;
    }
  }
  
  return true;
}

// std::vector<std::vector<int>> allej(int j, int m) {
//   std::vector<std::vector< int>> result;
//   
//   if (j == 1) {
//     result.push_back({0, static_cast<int>(m - 1)});
//     return result;
//   }
//   
//   if (j > m) {
//     return result;
//   }
//   
//   for (int sizeej = 0; sizeej <= m - j; ++sizeej) {
//     for (int binf = 0; binf <= m - sizeej; ++binf) {
//       int bsup = binf + sizeej - 1;
//       result.push_back({static_cast<int>(binf), static_cast<int>(bsup)});
//     }
//   }
//   
//   return result;
// }

std::vector<std::vector<int>> allej(int j, int m) {
  std::vector<std::vector<int>> result;
  
  if (j == 1) {
    result.push_back({1, m});
    result[0][0] -= 1;
    result[0][1] -= 1;
    return result;
  }
  
  if (j > m) {
    return result;
  }
  
  std::vector<int> indicesi;
  for (int i = 1; i <= m - j + 1; ++i) {
    indicesi.push_back(i);
  }
  
  for (int sizeej : indicesi) {
    std::vector<int> indicesj;
    for (int j = 1; j <= m - sizeej + 1; ++j) {
      indicesj.push_back(j);
    }
    
    for (int binf : indicesj) {
      int bsup = binf + sizeej - 1;
      result.push_back({binf, bsup});
      result.back()[0] -= 1;
      result.back()[1] -= 1;
    }
  }
  
  return result;
}


double pejp1_yjej(std::vector<int>& ejp1, int yj, std::vector<int>& ej, int mu, double p) {
  double proba = 0.0;
  std::vector<int> ejminus = {ej[0], yj - 1};
  std::vector<int> ejequal = {yj, yj};
  std::vector<int> ejplus = {yj + 1, ej[1]};
  
  // pejp1_yjejzj0
  double pejp1_yjejzj0 = 0.0;
  if (compare_vec(ejp1, ejminus) || compare_vec(ejp1, ejequal) || compare_vec(ejp1, ejplus)) {
    pejp1_yjejzj0 = static_cast<double>(ejp1[1] - ejp1[0] + 1) / (ej[1] - ej[0] + 1);
  }
  
  // pejp1_yjejzj1
  double dmuejminus = 0.0;
  if (ejminus[0] > ejminus[1]) {
    dmuejminus = INFINITY;
  } else {
    std::vector<int> ejminusbis = ejminus;
    for (int& val : ejminusbis) {
      val -= mu;
      val = std::abs(val);
    }
    dmuejminus = *std::min_element(ejminusbis.begin(), ejminusbis.end());
  }
  
  double dmuejplus = 0.0;
  if (ejplus[0] > ejplus[1]) {
    dmuejplus = INFINITY;
  } else {
    std::vector<int> ejplusbis = ejplus;
    for (int& val : ejplusbis) {
      val -= mu;
      val = std::abs(val);
    }
    dmuejplus = *std::min_element(ejplusbis.begin(), ejplusbis.end());
  }
  
  std::vector<int> ejequalbis = ejequal;
  for (int& val : ejequalbis) {
    val -= mu;
    val = std::abs(val);
  }
  double dmuejequal = *std::min_element(ejequalbis.begin(), ejequalbis.end());
  
  std::vector<int> ejp1bis = ejp1;
  for (int& val : ejp1bis) {
    val -= mu;
    val = std::abs(val);
  }
  double dmuejp1 = *std::min_element(ejp1bis.begin(), ejp1bis.end());
  
  std::vector<double> all_dmu = {dmuejminus, dmuejequal, dmuejplus};
  int pejp1_yjejzj1 = 0;
  if ((dmuejp1 == *std::min_element(all_dmu.begin(), all_dmu.end())) &&
      (compare_vec(ejp1, ejminus) || compare_vec(ejp1, ejequal) || compare_vec(ejp1, ejplus))) {
    pejp1_yjejzj1 = 1;
  }
  
  proba = p * pejp1_yjejzj1 + (1 - p) * pejp1_yjejzj0;
  return proba;
}

double pejp1zj1_yjej(std::vector<int>& ejp1, unsigned int yj, std::vector<int>& ej, int mu, double p) {
  double proba = 0.0;
  std::vector<int> ejminus = {ej[0], static_cast<int>(yj - 1)};
  std::vector<int> ejequal = {static_cast<int>(yj), static_cast<int>(yj)};
  std::vector<int> ejplus = {static_cast<int>(yj + 1), ej[1]};
  
  double dmuejminus = 0.0;
  double dmuejplus = 0.0;
  if (ejminus[0] > ejminus[1]) {
    dmuejminus = INFINITY;
  }
  else {
    std::vector<int> ejminusbis = ejminus;
    for (int& val : ejminusbis) {
      val -= mu;
      val = std::abs(val);
    }
    dmuejminus = *std::min_element(ejminusbis.begin(), ejminusbis.end());
  }
  
  if (ejplus[0] > ejplus[1]) {
    dmuejplus = INFINITY;
  }
  else {
    std::vector<int> ejplusbis = ejplus;
    for (int& val : ejplusbis) {
      val -= mu;
      val = std::abs(val);
    }
    dmuejplus = *std::min_element(ejplusbis.begin(), ejplusbis.end());
  }
  
  std::vector<int> ejequalbis = ejequal;
  for (int& val : ejequalbis) {
    val -= mu;
    val = std::abs(val);
  }
  double dmuejequal = *std::min_element(ejequalbis.begin(), ejequalbis.end());
  
  std::vector<int> ejp1bis = ejp1;
  for (int& val : ejp1bis) {
    val -= mu;
    val = std::abs(val);
  }
  double dmuejp1 = *std::min_element(ejp1bis.begin(), ejp1bis.end());
  
  std::vector<double> all_dmu = {dmuejminus, dmuejequal, dmuejplus};
  int pejp1_yjejzj1 = 0;
  if (dmuejp1 == *std::min_element(all_dmu.begin(), all_dmu.end()) &&
      (compare_vec(ejp1, ejminus) || compare_vec(ejp1, ejequal) || compare_vec(ejp1, ejplus))) {
    pejp1_yjejzj1 = 1;
  }
  
  proba = p * pejp1_yjejzj1;
  return proba;
}

double pejp1zj1_ej(std::vector<int>& ejp1, std::vector<int>& ej, int mu, double p) {
  double proba = 0.0;
  std::vector<int> ej_bounds;
  for (int i = ej[0]; i <= ej[1]; ++i) {
    ej_bounds.push_back(i);
  }
  for (unsigned int i = 0; i < ej_bounds.size(); ++i) {
    int yj = ej_bounds[i];
    proba += pejp1zj1_yjej(ejp1, yj, ej, mu, p);
  }
  proba /= (ej[1] - ej[0] + 1);
  return proba;
}

double pyj_ej(unsigned int yj, std::vector<int>& ej) {
  double proba = 0.0;
  if (ej[0] <= static_cast<int>(yj) && static_cast<int>(yj) <= ej[1]) {
    proba = 1.0 / (ej[1] - ej[0] + 1);
  }
  else {
    proba = 0.0;
  }
  return proba;
}

double pejp1_ej(std::vector<int>& ejp1, std::vector<int>& ej, int mu, double p) {
  double proba = 0.0;
  std::vector<unsigned int> allyj;
  
  if (ejp1[1] == ejp1[0]) { // |ejp1|=1
    if (ejp1[1] < ej[1] && ejp1[0] > ej[0]) { // ejp1 doesn't touch any bound
      allyj.push_back(ejp1[0]);
    }
    else {
      if (ejp1[1] < ej[1]) { // ejp1 doesn't touch right bound
        allyj.push_back(ejp1[0]);
        allyj.push_back(ejp1[0] + 1);
      }
      else { // ejp1 doesn't touch left bound
        allyj.push_back(ejp1[0] - 1);
        allyj.push_back(ejp1[0]);
      }
    }
  }
  else { // |ejp1|>1
    if (ejp1[1] < ej[1]) { // ejp1 doesn't touch right bound
      allyj.push_back(ejp1[1] + 1);
    }
    else { // ejp1 doesn't touch left bound
      allyj.push_back(ejp1[0] - 1);
    }
  }
  
  for (unsigned int i = 0; i < allyj.size(); ++i) {
    unsigned int yj = allyj[i];
    proba += pejp1_yjej(ejp1, yj, ej, mu, p) * pyj_ej(yj, ej);
  }
  
  return proba;
}

double pej(std::vector<int>& ej, int j, int m, int mu, double p, std::vector<int>& z1tozjm1)
{
  if (j == 1) {
    return 1.0;
  }
  
  if (ej.size() == 1) {
    int ejn = ej[0];
    ej = std::vector<int>(2, ejn);
  }
  
  std::vector<int> z1tozjm2(z1tozjm1.begin(), z1tozjm1.end() - 1);
  int zjm1 = z1tozjm1.back();
  
  double proba = 0.0;
  
  if (zjm1 != 0) { // zjm1 is known
    std::vector<std::vector<int>> tabint = allej(j - 1, m); // Replace with the appropriate implementation of allej()
    int nbtabint = tabint.size();
    
    for (int i = 0; i < nbtabint; ++i) {
      std::vector<int>& ejm1 = tabint[i];
      
      if ((ej[0] >= ejm1[0]) && (ej[1] <= ejm1[1])) { // To accelerate, check if ejm is included in ej
        proba += pejp1zj1_ej(ej, ejm1, mu, p) * pej(ejm1, j - 1, m, mu, p, z1tozjm2);
      }
    }
  }
  
  else { // zjm1 is unknown
    std::vector<std::vector<int>> tabint = allej(j - 1, m); // Replace with the appropriate implementation of allej()
    int nbtabint = tabint.size();
    
    for (int i = 0; i < nbtabint; ++i) {
      std::vector<int>& ejm1 = tabint[i];
      
      if ((ej[0] >= ejm1[0]) && (ej[1] <= ejm1[1])) { // To accelerate, check if ejm is included in ej
        proba += pejp1_ej(ej, ejm1, mu, p) * pej(ejm1, j - 1, m, mu, p, z1tozjm2);
      }
    }
  }
  
  return proba;
}


std::vector<double> pallx(
    int& mu,
    double& p,
    int& _m)
{
  Cube _tab_pejs = gettabpej(_m);
  
  std::vector<double> pallx(_m, 0.0);
  
  for (int i = 0; i < _m; ++i) {
    std::vector<double> tmp(_m, p);
    for (int deg = 0; deg < _m; ++deg) {
      tmp[deg] = std::pow(tmp[deg], deg);
    }
    
    
    std::vector<double> sub_cube;
    for (int slice = 0; slice < _tab_pejs[0][0].size(); ++slice) {
      sub_cube.push_back(_tab_pejs[slice][i][mu]);
    }
    
    std::vector<double> vecprod;
    for (unsigned int j = 0; j < sub_cube.size(); ++j) {
      vecprod.push_back(sub_cube[j] * tmp[j]);
    }
    
    pallx[i] = 0.0;
    for (double prod : vecprod) {
      pallx[i] += prod;
    }
  }
  return pallx;
}



std::vector<double> ordiemCpp(
    std::vector<int>& x,
    std::vector<int>& tabmu0,
    std::vector<double>& tabp0,
    int& _m,
    double eps = 1,
    int iter_max = 100)
{
  double ml = -INFINITY;
  double p_ml = tabp0[0];
  int mu_ml = tabmu0[0];
  
  Cube _tab_pejs = gettabpej(_m);

    int n = x.size();
  std::vector<double> w(n, 1.0);
  
  double ntot = 0.0;
  for (double weight : w) {
    ntot += weight;
  }

  int a = 0;
  for (unsigned int imu = 0; imu < tabmu0.size(); ++imu) {
    int mu = tabmu0[imu];
    double mlold = -INFINITY;
    for (unsigned int ip = 0; ip < tabp0.size(); ++ip) {
      // std::cout << mu << std::endl;
      double p = tabp0[ip];
      // std::cout << p << std::endl;
      int nostop = 1;
      int iter = 0;
      while (nostop) {
        iter++;
        // -- E step ---
        // first: compute px for each modality
        std::vector<double> pallx(_m, 0.0);
        
        std::vector<double> px(n, 0.0);
        
        for (int i = 0; i < _m; ++i) {
          std::vector<double> tmp(_m, p);
          for (int deg = 0; deg < _m; ++deg) {
            tmp[deg] = std::pow(tmp[deg], deg);
          }
          
          
          std::vector<double> sub_cube;
          for (int slice = 0; slice < _tab_pejs[0][0].size(); ++slice) {
            sub_cube.push_back(_tab_pejs[slice][i][imu]);
          }
          
          std::vector<double> vecprod;
          for (unsigned int j = 0; j < sub_cube.size(); ++j) {
            vecprod.push_back(sub_cube[j] * tmp[j]);
          }
          
          pallx[i] = 0.0;
          for (double prod : vecprod) {
            pallx[i] += prod;
          }

          for (unsigned int j = 0; j < x.size(); ++j) {
            if (x[j] == (i)) {
              px[j] = std::max(1e-300, pallx[i]);
            }
          }
        }

        std::vector<double> vecprod;
        for (unsigned int i = 0; i < w.size(); ++i) {
          vecprod.push_back(w[i] * std::log(px[i]));
        }
        
        double mlnew = 0.0;
        for (double prod : vecprod) {
          mlnew += prod;
        }
        
        // first: compute pxz1 for each modality
        std::vector<std::vector<double>> pallxz1(_m, std::vector<double>(_m - 1, 0.0));
        
        for (int i = 0; i < _m; ++i) {
          for (int j = 0; j < (_m - 1); ++j) {
            std::vector<int> z1tozmm1(_m - 1, 0);
            z1tozmm1[j] = 1;
            
            std::vector<int> ivec = { (i) };
            
            double pejv = pej(ivec, _m, _m, mu, p, z1tozmm1); // TODO change that
            pallxz1[i][j] = std::max(1e-300, pejv) / std::max(1e-300, pallx[i]);
          }
        }
        
        // second: affect each modality value to the corresponding units
        std::vector<std::vector<double>> pxz1(n, std::vector<double>(_m - 1, 0.0));
        for (int i = 0; i < _m; ++i) {
          std::vector<int> whereisi;
          for (int j = 0; j < n; ++j) {
            if (x[j] == i) {
              whereisi.push_back(j);
            }
          }
          
          std::vector<int> all_cols_pxz1(pxz1[0].size());
          std::iota(all_cols_pxz1.begin(), all_cols_pxz1.end(), 0);
          
          int sumwhereisi = whereisi.size();
          std::vector<double> matsumwhereisi(sumwhereisi, 1.0);
          std::vector<double> pallxz1i = pallxz1[i];
          std::vector<std::vector<double>> sub_mat(sumwhereisi, std::vector<double>(pallxz1i.size()));
          for (int j = 0; j < sumwhereisi; ++j) {
            for (int k = 0; k < pallxz1i.size(); ++k) {
              sub_mat[j][k] = matsumwhereisi[j] * pallxz1i[k];
            }
          }
          
          for (int j = 0; j < sumwhereisi; ++j) {
            for (int k = 0; k < all_cols_pxz1.size(); ++k) {
              pxz1[whereisi[j]][all_cols_pxz1[k]] = sub_mat[j][k];
            }
          }
        }
        
        std::vector<double> temp1(_m - 1, 1.0); // Initialize vector temp1 with _m - 1 elements, all set to 1.0
        
        std::vector<std::vector<double>> temp2(n, std::vector<double>(_m - 1, 0.0)); // Initialize matrix temp2 with n rows and _m - 1 columns, all set to 0.0
        
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < _m - 1; ++j) {
            temp2[i][j] = w[i] * temp1[j];
          }
        }
        
        for (unsigned int i = 0; i < pxz1.size(); ++i) {
          for (unsigned int j = 0; j < pxz1[i].size(); ++j) {
            pxz1[i][j] *= temp2[i][j];
          }
        }
        
        // ---- M step ----
        double sum = 0.0;
        for (unsigned int i = 0; i < pxz1.size(); ++i) {
          for (unsigned int j = 0; j < pxz1[i].size(); ++j) {
            sum += pxz1[i][j];
          }
        }
        
        double pmean = sum / (ntot * (_m - 1));
        if (!(mlnew == -INFINITY)) {
          if ((std::abs(mlnew - mlold) / ntot < eps) || (iter > (iter_max - 1))) {
            nostop = 0;
            if (mlnew > ml) {
              ml = mlnew;
              p_ml = pmean;
              mu_ml = mu;
            }
          }
        } else {
          if (iter > (iter_max - 1)) {
            nostop = 0;
            if (mlnew > ml) {
              ml = mlnew;
              p_ml = pmean;
              mu_ml = mu;
            }
          }
        }

        mlold = mlnew;
      } // while
    } // p
    ++a;
  } // mu
  
  double p_out = p_ml;
  double mu_out = mu_ml * 1.0; 
  std::vector<double> results(2, 0.0);
  results[0] = p_out;
  results[1] = mu_out; 
  
  return results;
}

