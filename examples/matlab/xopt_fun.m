function varargout = xopt_fun(varargin)
%XOPT_FUN Used by Xopt class
%
%[y1,y2,...,isvalid] = xopt_fun(objfun,x1,x2,...)
%  isvalid set to <0 if any objective function variables return nan or inf

nrhs=nargin-1;
nlhs=nargout-1;
% Form function call
fcall="[";
for ilhs=1:nlhs
  fcall=fcall+"varargout{"+ilhs+"}";
  if ilhs<nlhs
    fcall=fcall+",";
  end
end
fcall=fcall+"] = "+varargin{1}+"(";
for irhs=1:nrhs
  fcall=fcall+"X"+irhs;
  if irhs<nrhs
    fcall=fcall+",";
  end
end
fcall=fcall+");";

% Execute function
for iarg=1:nrhs
  eval("X"+iarg+"=varargin{iarg+1};");
end
eval(fcall);
varargout{nargout}=1; % data OK
for iarg=1:nlhs
  if isnan(varargout{iarg}) || isinf(varargout{iarg})
    varargout{nargout}=-1; % data NOT OK
    break
  end
end
if varargout{nargout}<0
  for iarg=1:nlhs
    varargout{iarg}=1000000;
  end
end
% fprintf('xopt_fun: x1: %g x2: %g y1: %g y2: %g c1: %g\n',varargin{2},varargin{3},varargout{1},varargout{2},varargout{3});