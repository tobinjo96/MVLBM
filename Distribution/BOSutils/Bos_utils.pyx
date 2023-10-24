import numpy as np
cimport numpy as np

from libcpp.vector cimport vector


cdef extern from "bos_cpp.h":
    vector[double] ordiemCpp(vector[int] x, vector[int] tabmu0, vector[double] tabp0, int _m, double eps, int iter_max)

    vector[double] pallx(int mu, double p, int _m)
    
    double pej(vector[int] ej, int j, int m, int mu, double p, vector[int] z1tozjm1)
    
    double pejp1_ej(vector[int] ejp1, vector[int] ej, int mu, double p) 
    
    double pejp1zj1_ej(vector[int] ejp1, vector[int] ej, int mu, double p)

cdef ordiemCpp_np(x, tabmu0, tabp0, _m, eps, iter_max):
    return ordiemCpp(x, tabmu0, tabp0, _m, eps, iter_max)

cdef pallx_np(mu, p, _m):
    return pallx(mu, p, _m)

cdef pej_np(ej, j, m, mu, p, z1tozjm1):
    return pej(ej, j, m, mu, p, z1tozjm1)

cdef pejp1_ej_np(ejp1, ej, mu, p):
    return pejp1_ej_np(ejp1, ej, mu, p)

cdef pejp1zj1_ej_np(ejp1, ej, mu, p):
    return pejp1zj1_ej(ejp1, ej, mu, p)


class Bos_utils:
  def __init__(self):
    pass
  
  def ordiemCpp_run(self,x, tabmu0, tabp0, _m, eps, iter_max):
      return ordiemCpp_np(x, tabmu0, tabp0, _m, eps, iter_max)
  
  def pallx_run(self, mu, p, _m):
      return pallx_np(mu, p, _m)
  
  def pej_run(self, ej, j, m, mu, p, z1tozjm1):
      return pej_np(ej, j, m, mu, p, z1tozjm1)
  
  def pejp1_ej_run(self, ejp1, ej, mu, p):
      return pejp1_ej_np(ejp1, ej, mu, p)

  def pejp1zj1_ej_run(self, ejp1, ej, mu, p):
      return pejp1zj1_ej_np(ejp1, ej, mu, p)
