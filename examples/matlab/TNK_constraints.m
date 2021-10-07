function [c,ceq] = TNK_constraints(x)

ceq=zeros(size(x));

c(1) =  -x(1).^2-+ x(2).^2 + 1.0 - 0.1 * cos(16 .* atan2(x(1), x(2))) ;
c(2) = (x(1) - 0.5).^2 + (x(2) - 0.5).^2 - 0.5 ;
