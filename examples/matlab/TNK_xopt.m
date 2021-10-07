function [y1,y2] = TNK_xopt(x1,x2)

g1=  -x1.^2-+ x2.^2 + 1.0 - 0.1 * cos(16 .* atan2(x1, x2)) ;
g2 = (x1 - 0.5).^2 + (x2 - 0.5).^2 ;

y1 = x1 ;
y2 = x2 ;

if g1>0 || g2>0.5
  y1=nan; y2=nan;
end