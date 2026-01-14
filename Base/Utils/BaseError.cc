// (C) Copyright 2019 by Autodesk, Inc.

#include "BaseError.hh"

namespace Base { 

static const char* ERROR_MESSAGE[] =
{
  #define DEFINE_ERROR(CODE, MSG) MSG,
  #include "BaseErrorInc.hh"
  #undef DEFINE_ERROR
};

Error::Error(const int _idx) : Error(_idx, ERROR_MESSAGE[_idx]) {}
const char* Error::message() const { return ERROR_MESSAGE[idx_]; }

}//namespace Base
