function [y1,y2]=testeval(x1,x2)
y1= x1.^2 + x2.^2 - 1.0 - 0.1 * cos(16 .* atan2(x1, x2)) ;
y2 = (x1 - 0.5).^2 + (x2 - 0.5).^2 ;