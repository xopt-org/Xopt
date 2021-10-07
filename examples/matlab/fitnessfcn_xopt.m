function [y1,y2] = fitnessfcn_xopt(x1,x2)
%FITNESSFCN_XOPT Test function with 2 objective functions and 2 decision variables for use with RUNME.mlx script

f = @(x)[norm(x)^2,0.5*norm(x(:)-[2;-1])^2+2];

output = feval(f,[x1,x2]) ;

y1=output(1); y2=output(2);

% Apply linear constraint by setting NaN's when constraint fails
if x1 + x2 >= 0.5
  y1=nan; y2=nan;
end